"""Phi experiment 2026-10-15: PlaceCube v2, the fixes found on the first real-arm runs (phi/group/squint SYSTEM.md E37, E38).

Each change answers one measured failure of SO101PlaceCube-v1. Scene, item, bin, colours and controller are unchanged.

1. Inside test in the bin's own frame, shrunk by INSET.
   Upstream evaluate() compares the item's world x/y offset with the bin half sizes, but the bin spawns at a random yaw,
   so 15.6% of what it calls "above the bin" is outside the walls (Monte Carlo, uniform yaw). It also accepts a centre
   that is just past the wall line, so the policy learned to open the gripper as the cube crosses the near wall.
2. Success also needs the item resting on the bin floor and still: upstream counts a cube lying on a wall top.
3. Gripper openness reward saturates at OPEN_CAP_DEG: upstream pays openness linearly to the 120 deg joint limit,
   which taught "open as wide as possible over the bin".
4. After success, a penalty on gripper motion (as SO101PlaceCubeHold-v1): upstream pays a flat 9 after success, so
   nothing stops the opening habit.
5. While carrying, a bonus for holding the gripper target SQUEEZE_LEAD_DEG past the cube. The sim gripper (stiffness
   500-2000, force limit 100) holds the cube with a 0.2 deg median lead; the real servo with LeRobot's P=16 does not,
   and real cubes slipped mid-carry. The bonus saturates at the deploy safety cap on gripper goal lead (5 deg).

SO101PlaceCubeV2Far-v1 adds cubes spawned beyond the usual area, so the policy has to reach far or drag the cube closer.
SO101PlaceCubeV2FarTop-v1 adds a fixed top camera to the wrist camera (6-channel image input).
"""
from typing import Any

import numpy as np
import sapien
import torch
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose

from envs.base_random_env import ThirdCameraEnv
from envs.place import PlaceCube

INSET = 0.02            # m. WHY: cube half width 1.25 cm + wall 0.5 cm + 0.25 cm slack: a cube dropped straight down clears the wall
FLOOR_TOL = 0.01        # m above the resting height on the bin floor; a cube on a 2.4-3.6 cm wall is far above this
ITEM_STILL = 2e-2       # m/s, upstream's own is_item_static threshold (place.py evaluate)
OPEN_CAP_DEG = 35.0     # WHY: grasps sit at ~5 deg and the real cube fell free by ~15-18 deg; the start pose is 60 deg
HOLD_PENALTY = 0.5      # same value as SO101PlaceCubeHold-v1 (success still pays >= 8.5)
SQUEEZE_BONUS = 0.3     # WHY: small enough that carrying (<= 5.3) still pays less than reaching the bin (>= ~6)
SQUEEZE_LEAD_DEG = 5.0  # deploy safety cap on gripper goal lead (deploy_utils/safety.py), so sim asks for what real can give


class PlaceCubeV2(PlaceCube):
    def _bin_frame_xy(self, p: torch.Tensor):
        """World positions (N, 3) to x, y in the bin's own frame (the bin is only ever rotated about z)."""
        d = p[:, :2] - self.bin.pose.p[:, :2]
        q = self.bin.pose.q   # wxyz
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        c, s = torch.cos(yaw), torch.sin(yaw)
        return c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]

    def evaluate(self):
        info = super().evaluate()
        x, y = self._bin_frame_xy(self.item.pose.p)
        inside_x = x.abs() < self.bin_half_sizes_x - INSET
        inside_y = y.abs() < self.bin_half_sizes_y - INSET
        above = inside_x & inside_y
        on_floor = self.item.pose.p[:, 2] < self.bin_thickness + self.item_half_sizes + FLOOR_TOL
        still = info["item_vel"] <= ITEM_STILL
        success = (above & on_floor & still & info["is_robot_static"]
                   & ~info["robot_touching_item"] & ~info["robot_touching_bin"])
        info.update(inside_x=inside_x, inside_y=inside_y, is_item_above_bin=above, item_on_bin_floor=on_floor,
                    success=success)
        return info

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        reward = super().compute_dense_reward(obs, action, info)   # upstream terms, on this class's evaluate()
        lo, hi = self.agent.robot.get_qlimits()[0, -1, :]
        q_grip = self.agent.robot.get_qpos()[:, -1]
        openness = (q_grip - lo) / (hi - lo)
        cap = (np.deg2rad(OPEN_CAP_DEG) - lo) / (hi - lo)
        above_only = (info["is_item_above_bin"] & ~info["success"]).float()
        # swap upstream's linear openness term for one that stops paying at the cap
        reward = reward + above_only * (torch.clamp(openness, max=cap) / cap - openness)

        a_grip = action[..., -1].to(reward.device)
        reward = reward - HOLD_PENALTY * a_grip.abs() * info["success"].float()

        target_grip = self.agent.controller._target_qpos[:, -1]
        lead = (q_grip - target_grip) / np.deg2rad(SQUEEZE_LEAD_DEG)   # > 0: target is more closed than the jaw
        carrying = (info["is_item_grasped"] & ~info["is_item_above_bin"]).float()
        return reward + SQUEEZE_BONUS * carrying * torch.clamp(lead, 0, 1)


FAR_PROB = 0.3                # share of episodes with a far cube
FAR_X = (0.42, 0.50)          # m, URDF frame = 38-46 cm from the pan axis (3.88 cm ahead); TCP reach at table height ~42 cm
FAR_Y = 0.10
NEAR_R = 0.42                 # m from the URDF origin: covers the usual spawn square, corners included (0.412 m)
PULL_PENALTY = 0.5


class PlaceCubeV2Far(PlaceCubeV2):
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            xy = torch.stack([torch.rand(b) * (FAR_X[1] - FAR_X[0]) + FAR_X[0], (2 * torch.rand(b) - 1) * FAR_Y], -1)
            p = self.item.pose.p[env_idx].clone()
            clear = torch.linalg.norm(xy - self.bin.pose.p[env_idx, :2], dim=-1) > self.bin_radius[env_idx] + 0.025
            far = (torch.rand(b) < FAR_PROB) & clear
            p[far, :2] = xy[far]
            self.item.set_pose(Pose.create_from_pq(p, self.item.pose.q[env_idx]))

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        reward = super().compute_dense_reward(obs, action, info)
        # WHY: zero inside the usual area, so ordinary episodes are scored exactly as PlaceCubeV2; beyond it, dragging
        # the cube back towards the robot removes the penalty
        r = torch.linalg.norm(self.item.pose.p[:, :2], dim=-1)
        return reward - PULL_PENALTY * torch.tanh(10 * torch.clamp(r - NEAR_R, min=0))


# Top camera, URDF frame: lens 62 cm in front of the base on the centre line, 45 cm above the table, aimed at the
# table 31 cm in front of the base. WHY this framing: the policy sees 16x16, so the view is cropped to the robot base
# plus every spawn (x 12-50 cm), about 2.5 cm of table per pixel, i.e. the cube is ~1 pixel: a coarse "where is it"
# that the wrist camera refines. Randomised every step by ThirdCameraEnv's noise (+-2.5 cm eye, +-5 deg FOV) plus
# +-1.5 cm on the aim point, so a hand-placed real camera only has to be roughly there.
TOP_CAMERA_EYE = [0.62, 0.0, 0.45]
TOP_CAMERA_TARGET = [0.31, 0.0, 0.0]
TOP_CAMERA_FOV = np.deg2rad(50)
TOP_CAMERA_AIM_NOISE = 0.015


class PlaceCubeV2FarTop(PlaceCubeV2Far):
    """Wrist camera ("base_camera", unchanged) plus a fixed top camera ("top_camera"); images stack to 6 channels."""

    def __init__(self, *args, domain_randomization_config=None, **kwargs):
        self.base_camera_settings = dict(pos=TOP_CAMERA_EYE, target=TOP_CAMERA_TARGET)
        cfg = dict(domain_randomization_config or {})
        cfg.setdefault("third_camera_target_noise", TOP_CAMERA_AIM_NOISE)
        super().__init__(*args, domain_randomization_config=cfg, **kwargs)

    @property
    def _default_sensor_configs(self):
        cfg = self.domain_randomization_config
        fov_noise = cfg.third_camera_fov_noise * (2 * self._batched_episode_rng.rand() - 1) if self.domain_randomization else 0
        top = CameraConfig("top_camera", pose=sapien.Pose(), width=128, height=128, fov=TOP_CAMERA_FOV + fov_noise,
                           near=0.01, far=100, mount=self.camera_mount)
        return super()._default_sensor_configs + [top]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        self.camera_mount.set_pose(ThirdCameraEnv.sample_camera_poses(self, n=len(env_idx)))

    def _before_control_step(self):
        if self.domain_randomization:
            self.camera_mount.set_pose(ThirdCameraEnv.sample_camera_poses(self, n=self.num_envs))
            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()


register_env("SO101PlaceCubeV2-v1", max_episode_steps=50)(PlaceCubeV2)
register_env("SO101PlaceCubeV2Far-v1", max_episode_steps=50)(PlaceCubeV2Far)
register_env("SO101PlaceCubeV2FarTop-v1", max_episode_steps=50)(PlaceCubeV2FarTop)
