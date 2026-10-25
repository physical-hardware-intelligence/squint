"""Phi: two SO-101 arms, each with its own wrist camera, on top of base_random_env's domain randomization.

WristCameraEnv's conventions, once per arm. The agent is ManiSkill's MultiAgent: arm 0 is the left arm, arm 1 the
right arm (seen from behind the robots, both facing +x). The sensors are "left_wrist" then "right_wrist", in that order,
so a flattened observation stacks the left camera's 3 channels before the right camera's, and deploy must stack the
real frames the same way.
"""

import math
from typing import Union

import numpy as np
import sapien
import torch
from sapien.render import RenderBodyComponent

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

from .base_random_env import BaseRandomEnv, RandomizationConfig, WristCameraEnv

ARM_NAMES = ("left", "right")


class BimanualWristCameraEnv(BaseRandomEnv):
    """Two arms, a wrist camera on each, every camera following its own gripper with independent pose noise."""

    # The same mount on both arms until the right arm's camera is fitted and measured (it uses a different printed part).
    WRIST_CAMERA_BASE_POS = WristCameraEnv.WRIST_CAMERA_BASE_POS
    WRIST_CAMERA_BASE_ROT_RAD = WristCameraEnv.WRIST_CAMERA_BASE_ROT_RAD
    WRIST_CAMERA_FOV = WristCameraEnv.WRIST_CAMERA_FOV

    # Optional fixed top camera (B008): off by default, so existing envs keep two cameras
    TOP_CAMERA = False
    TOP_CAMERA_EYE = [0.17, 0.0, 0.90]  # overhead: from behind, the arms hid the bins
    TOP_CAMERA_TARGET = [0.20, 0.0, 0.0]
    TOP_CAMERA_FOV = np.deg2rad(55)
    TOP_CAMERA_INTRINSIC = None  # a 3x3 matrix instead of TOP_CAMERA_FOV (a wide lens squeezed into the square image)

    agent: MultiAgent

    def __init__(
        self,
        *args,
        domain_randomization_config: Union[RandomizationConfig, dict] = RandomizationConfig(),
        **kwargs,
    ):
        super().__init__(*args, domain_randomization_config=domain_randomization_config, **kwargs)

    @property
    def _default_sim_config(self):
        # Two arms per scene hold more contact pairs than the single-arm defaults (ManiSkill's TwoRobot tasks raise these too)
        return SimConfig(
            sim_freq=100,
            control_freq=10,
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**19, max_rigid_contact_count=2**21
            ),
        )

    @property
    def arms(self) -> list:
        """[left, right] SO-101 agents."""
        return self.agent.agents

    def _load_camera_mount(self):
        """One kinematic mount per wrist camera."""
        self.wrist_camera_mounts = []
        for name in ARM_NAMES:
            builder = self.scene.create_actor_builder()
            builder.initial_pose = sapien.Pose()
            self.wrist_camera_mounts.append(builder.build_kinematic(f"{name}_wrist_camera_mount"))

    @property
    def _default_sensor_configs(self):
        config = self.domain_randomization_config
        configs = []
        for name, mount in zip(ARM_NAMES, self.wrist_camera_mounts):
            # FOV noise drawn per env and per camera: the two real cameras are separate parts
            if self.domain_randomization and config.wrist_camera_fov_noise > 0:
                fov_noise = config.wrist_camera_fov_noise * (2 * self._batched_episode_rng.rand() - 1)
            else:
                fov_noise = 0
            configs.append(
                CameraConfig(
                    f"{name}_wrist",
                    pose=sapien.Pose(),
                    width=128,
                    height=128,
                    fov=self.WRIST_CAMERA_FOV + fov_noise,
                    near=0.01,
                    far=100,
                    mount=mount,
                )
            )
        if self.TOP_CAMERA:
            # Fixed, about 90 cm above the table, looking down at both arms, the bar and the bin. No noise: drawing it
            # from the episode generator would shift the per-env object sizes drawn after it.
            lens = dict(intrinsic=self.TOP_CAMERA_INTRINSIC) if self.TOP_CAMERA_INTRINSIC is not None else dict(fov=self.TOP_CAMERA_FOV)
            configs.append(CameraConfig("top", pose=sapien_utils.look_at(self.TOP_CAMERA_EYE, self.TOP_CAMERA_TARGET),
                                        width=128, height=128, near=0.01, far=100, **lens))
        return configs

    def _randomize_robot_color(self):
        """base_random_env's robot colour, applied to both arms."""
        if self.domain_randomization_config.robot_color is None:
            return
        for arm in self.arms:
            for link in arm.robot.links:
                for i, obj in enumerate(link._objs):
                    render_body_component: RenderBodyComponent = obj.entity.find_component_by_type(RenderBodyComponent)
                    if render_body_component is None:
                        continue
                    for render_shape in render_body_component.render_shapes:
                        for part in render_shape.parts:
                            if self.domain_randomization and self.domain_randomization_config.robot_color == "random":
                                color = self._batched_episode_rng[i].uniform(0.0, 1.0, size=(3,)).tolist()
                            else:
                                color = list(self.domain_randomization_config.robot_color)
                            part.material.set_base_color(color + [1])

    def _randomize_gripper_speed(self, env_idx: torch.Tensor):
        """Gripper stiffness and damping per episode, drawn separately for each arm."""
        stiff_lo, stiff_hi = self.domain_randomization_config.gripper_stiffness_range
        damp_lo, damp_hi = self.domain_randomization_config.gripper_damping_range
        if not hasattr(self, "_gripper_stiffness"):
            self._gripper_stiffness = torch.full((len(ARM_NAMES), self.num_envs), (stiff_lo + stiff_hi) / 2, device=self.device)
            self._gripper_damping = torch.full((len(ARM_NAMES), self.num_envs), (damp_lo + damp_hi) / 2, device=self.device)
        if not self.domain_randomization or (stiff_lo == stiff_hi and damp_lo == damp_hi):
            return
        for a, arm in enumerate(self.arms):
            stiffnesses = self._batched_episode_rng[env_idx].uniform(stiff_lo, stiff_hi)
            dampings = self._batched_episode_rng[env_idx].uniform(damp_lo, damp_hi)
            gripper_joint = arm.robot.joints_map["gripper"]
            for i, idx in enumerate(env_idx.tolist()):
                gripper_joint._objs[idx].set_drive_properties(stiffnesses[i], dampings[i], force_limit=100)
                self._gripper_stiffness[a, idx] = stiffnesses[i]
                self._gripper_damping[a, idx] = dampings[i]

    def _random_wrist_offset(self) -> Pose:
        """The camera's pose on its gripper, with this step's noise: WristCameraEnv._update_wrist_camera_pose's math."""
        config = self.domain_randomization_config
        base_x, base_y, base_z = self.WRIST_CAMERA_BASE_POS
        base_roll, base_pitch, base_yaw = self.WRIST_CAMERA_BASE_ROT_RAD
        if self.domain_randomization:
            rand_vals = 2 * torch.rand(self.num_envs, 6, device=self.device) - 1
            pos_offset, rot_noise = config.wrist_camera_pos_noise, config.wrist_camera_rot_noise
            dx, dy, dz = (pos_offset[k] * rand_vals[:, k] for k in range(3))
            d_roll, d_pitch, d_yaw = (rot_noise[k] * rand_vals[:, 3 + k] for k in range(3))
        else:
            dx = dy = dz = d_roll = d_pitch = d_yaw = torch.zeros(self.num_envs, device=self.device)

        px, py, pz = base_x + dx, base_y + dy, base_z + dz
        roll_rad, pitch_rad, yaw_rad = base_roll + d_roll, base_pitch + d_pitch, base_yaw + d_yaw

        cj, sj = torch.cos(pitch_rad / 2), torch.sin(pitch_rad / 2)
        ck, sk = torch.cos(yaw_rad / 2), torch.sin(yaw_rad / 2)
        ci, si = torch.cos(roll_rad / 2), torch.sin(roll_rad / 2)
        q_py_w, q_py_x, q_py_y, q_py_z = cj * ck, sj * sk, sj * ck, cj * sk
        qw = q_py_w * ci - q_py_x * si
        qx = q_py_w * si + q_py_x * ci
        qy = q_py_y * ci + q_py_z * si
        qz = q_py_z * ci - q_py_y * si

        return Pose.create_from_pq(p=torch.stack([px, py, pz], dim=-1), q=torch.stack([qw, qx, qy, qz], dim=-1))

    def _update_wrist_camera_pose(self):
        """Each camera mount follows its own gripper."""
        for arm, mount in zip(self.arms, self.wrist_camera_mounts):
            mount.set_pose(arm.robot.links_map["gripper_link"].pose * self._random_wrist_offset())

    def reset(self, *args, **kwargs):
        """Reset, sync both wrist cameras to the new arm poses, then observe again. The parent's reset renders its
        observation before the cameras move, so its first frame shows the view from the previous episode's last
        gripper pose (WristCameraEnv.reset returns that stale frame; found 2026-10-25, B007)."""
        obs, info = super().reset(*args, **kwargs)
        self._after_reset_before_observe()
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()
        self._update_wrist_camera_pose()
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene._gpu_fetch_all()
        obs = self.get_obs(info)
        self._last_obs = obs
        return obs, info

    def _after_reset_before_observe(self):
        """Hook for subclasses: runs after the parent reset (controllers already reset), before the cameras are synced
        and the returned observation is rendered."""

    def _after_control_step(self):
        """Update both wrist camera poses after the physics step."""
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()
        self._update_wrist_camera_pose()
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()


def arm_yaw_pose(x: torch.Tensor, y: torch.Tensor, yaw: torch.Tensor, z: torch.Tensor = None) -> Pose:
    """Batched poses: position (x, y, z), z = 0 on the table plane by default, rotated by yaw about z."""
    p = torch.stack([x, y, torch.zeros_like(x) if z is None else z], dim=-1)
    q = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], dim=-1)
    return Pose.create_from_pq(p=p, q=q)


def nominal_yaw_quat(yaw: float) -> list:
    """(w, x, y, z) for a rotation of yaw radians about z."""
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]


def lehome_top_intrinsic(size=128, hfov_deg=67.2, vfov_deg=52.9):
    """LeHome's top camera lens (4:3; focal 28.7 mm, aperture 38.11 mm: 67.2° wide, 52.9° tall), its full frame squeezed
    into a square image, as a real 4:3 frame resized to square would be."""
    fx = (size / 2) / math.tan(math.radians(hfov_deg / 2))
    fy = (size / 2) / math.tan(math.radians(vfov_deg / 2))
    return [[fx, 0.0, size / 2], [0.0, fy, size / 2], [0.0, 0.0, 1.0]]


class LeHomeTopCamera:
    """Mixin: the top camera placed as LeHome's (lehome-challenge garment_bi_cfg_v2.py, the competition's rig, arms 46 cm
    apart as ours): centred between the arms, 44 cm in front of their bases, 56 cm above the table, looking down and back
    toward the robots at 19° from vertical. LeHome's README gives ~65 cm for the real rig: measure ours."""

    TOP_CAMERA = True
    TOP_CAMERA_EYE = [0.44, 0.0, 0.56]
    TOP_CAMERA_TARGET = [0.44 - 0.56 * math.tan(math.radians(19.0)), 0.0, 0.0]
    TOP_CAMERA_INTRINSIC = lehome_top_intrinsic()
