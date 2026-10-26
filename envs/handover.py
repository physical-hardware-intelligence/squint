"""Phi: two SO-101 arms hand a red bar over in mid-air and put it in a bin (SO101HandoverBar-v1).

Layout, world frame = midway between the two arms' base origins, x forward, y to the robots' left: the left arm's
base at y = +s/2, the right arm's at y = -s/2, both facing +x; s = 46 cm on our rig, pan axis to pan axis.
Each episode one arm is the picker. The bar starts on the picker's outer side, out of the other arm's reach; the bin
stands on the other arm's outer side, out of the picker's reach (an arm reaches at most ~43 cm from its pan axis,
sampled in this simulator 2026-10-24). Success needs a mid-air handover: both grippers on the lifted bar at once,
then the bar lying in the bin with both arms off it and still.

The bar: 25 x 25 x 60 mm, red, designed for this task: 25 mm is the width our gripper already grasps (the cube),
60 mm leaves room for two grippers, one per end, and fits the PlaceCube bin (7-9 x 9-11 cm inside) at any yaw.
"""

from dataclasses import dataclass
from typing import Any, Sequence, Union

import dacite
import numpy as np
import sapien
import torch

from mani_skill.utils import common
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat

from .base_random_env import DefaultRandomizationConfig
from .bimanual_base import ARM_NAMES, BimanualWristCameraEnv, LeHomeTopCamera, arm_yaw_pose, nominal_yaw_quat
from .robot.so101 import SO101

PAN_AXIS_X = 0.0388  # the pan axis sits 3.88 cm in front of the base origin (so101.urdf L439)
CONTACT_LINKS = ("gripper_link", "moving_jaw_so101_v1_link", "wrist_link", "lower_arm_link")  # checked for arm-arm hits
JAW_LINKS = ("gripper_link", "moving_jaw_so101_v1_link")  # reward v3: jaw-to-jaw contact is part of a handover


@dataclass
class HandoverRandomizationConfig(DefaultRandomizationConfig):
    """Domain randomization for the handover, extending the wrist camera randomization."""
    # Noisy joint positions for better sim2real, as in PlaceCube
    robot_qpos_noise_std: float = np.deg2rad(5)
    # The bar: square cross-section and length (half sizes, m)
    bar_half_width_range: Sequence[float] = (0.024 / 2, 0.026 / 2)
    bar_half_length_range: Sequence[float] = (0.058 / 2, 0.062 / 2)
    bar_friction_range: Sequence[float] = (0.2, 0.6)
    bar_density_range: Sequence[float] = (250, 450)  # kg/m^3: a 15%-infill PLA print [estimate until weighed]
    bar_color: Sequence[float] = (1.0, 0.0, 0.0)
    # The bin: as PlaceCube's (half sizes, m)
    bin_half_size_x_range: Sequence[float] = (0.07 / 2, 0.09 / 2)
    bin_half_size_y_range: Sequence[float] = (0.09 / 2, 0.11 / 2)
    bin_half_size_z_range: Sequence[float] = (0.024 / 2, 0.036 / 2)
    # The rig: pan axis to pan axis, and each base's misalignment about z
    arm_separation_range: Sequence[float] = (0.455, 0.465)  # 46 cm nominal [to measure]
    arm_yaw_noise: float = np.deg2rad(1)


class Handover(BimanualWristCameraEnv):
    """
    **Task Description:**
    One arm picks up a bar, hands it to the other arm in mid-air, and the other arm puts it in a bin.

    **Randomizations:**
    - which arm picks (per episode); the bar's position and yaw on the picker's outer side
    - the bin's position and yaw on the other arm's outer side
    - with domain_randomization: bar size, mass, friction; bin size; arm spacing and yaw; cameras, lights, grippers

    **Success Conditions:**
    - both grippers held the lifted bar at the same moment (the handover) earlier in the episode
    - the bar lies in the bin, still, with neither arm touching the bar or the bin, and both arms static
    """

    SUPPORTED_ROBOTS = [("so101", "so101")]
    SUPPORTED_OBS_MODES = ["none", "state", "state_dict", "rgb", "rgb+segmentation", "rgb+state", "rgb+segmentation+state",
                           "rgb+depth+segmentation", "rgb+depth+segmentation+state"]

    # Reward stages (see compute_dense_reward); the maximum normalizes the reward to 1
    MAX_REWARD = 16.0
    PICKER_CLEAR_DISTANCE = 0.06  # reward v4: the picker's release is fully paid 6 cm from its end of the bar
    # A grasp needs the jaws stopped by the bar: is_grasping alone (contact forces up to 110 degrees off the closing
    # direction) also accepts a fully closed gripper pressed against the bar's side, which drops it as soon as the
    # other arm lets go (B018: 255 of 512 such starts; real grips sit at 10-11% open, those at 0%). Off in the envs as run
    GRASP_NEEDS_OPEN_JAWS = False
    MIN_GRASP_OPENNESS = 0.03
    # How the carry is paid for its distance d to the goal: "tanh" = 1 - tanh(5 d), steep only within ~15 cm (B001-B005);
    # "linear" = 1 - d / 40 cm, the same pull at every distance (B006). 40 cm covers the handover point to any goal
    CARRY_DISTANCE_SHAPE = "tanh"
    CARRY_LINEAR_RANGE = 0.40
    # Reward version. 1: B001-B007. 2 (B008): the same up to the handover; after it, scores follow the bar's progress
    # instead of contact flags, so the intended path never dips. 3 (B012): v2, plus a handover step that pays a full
    # point, no 5 cm cliff, and no penalty for jaw-to-jaw contact during the handover (see docs/bimanual/reward-design.md)
    REWARD_VERSION = 1

    def __init__(
        self,
        *args,
        robot_uids=("so101", "so101"),
        control_mode="pd_joint_target_delta_pos",
        domain_randomization_config: Union[HandoverRandomizationConfig, dict] = HandoverRandomizationConfig(),
        domain_randomization=False,
        handover_point=(PAN_AXIS_X + 0.17, 0.0, 0.12),
        **kwargs,
    ):
        self.rest_qpos = SO101.keyframes["start"].qpos.tolist()
        self.handover_point_xyz = handover_point

        self.domain_randomization_config = HandoverRandomizationConfig()
        merged_domain_randomization_config = self.domain_randomization_config.dict()
        if isinstance(domain_randomization_config, dict):
            common.dict_merge(merged_domain_randomization_config, domain_randomization_config)
            self.domain_randomization_config = dacite.from_dict(
                data_class=HandoverRandomizationConfig,
                data=merged_domain_randomization_config,
                config=dacite.Config(strict=True),
            )
        elif isinstance(domain_randomization_config, HandoverRandomizationConfig):
            self.domain_randomization_config = domain_randomization_config

        super().__init__(
            *args,
            robot_uids=robot_uids,
            control_mode=control_mode,
            domain_randomization=domain_randomization,
            domain_randomization_config=self.domain_randomization_config,
            **kwargs,
        )

    # ------------------------------------------------------------------ scene
    def _nominal_separation(self) -> float:
        lo, hi = self.domain_randomization_config.arm_separation_range
        return (lo + hi) / 2

    def _load_agent(self, options: dict):
        s = self._nominal_separation()
        super()._load_agent(
            options,
            [sapien.Pose(p=[0, s / 2, 0], q=nominal_yaw_quat(0)), sapien.Pose(p=[0, -s / 2, 0], q=nominal_yaw_quat(0))],
            build_separate=True
            if self.domain_randomization and self.domain_randomization_config.robot_color == "random"
            else False,
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()
        cfg = self.domain_randomization_config

        def per_env(rng_range):
            lo, hi = rng_range
            if self.domain_randomization:
                return self._batched_episode_rng.uniform(low=lo, high=hi)
            return np.ones(self.num_envs) * (lo + hi) / 2

        # The bar, one per env (sizes, mass and friction are fixed at scene load, like PlaceCube's items)
        half_widths = per_env(cfg.bar_half_width_range)
        half_lengths = per_env(cfg.bar_half_length_range)
        frictions = per_env(cfg.bar_friction_range)
        densities = per_env(cfg.bar_density_range)
        self.bar_half_widths = common.to_tensor(half_widths, device=self.device)
        self.bar_half_lengths = common.to_tensor(half_lengths, device=self.device)
        self.bar_frictions = common.to_tensor(frictions, device=self.device)
        self.bar_densities = common.to_tensor(densities, device=self.device)
        color = list(cfg.bar_color) + [1.0]
        bars = []
        for i in range(self.num_envs):
            half_size = [half_lengths[i], half_widths[i], half_widths[i]]  # long axis = the bar's local x
            material = sapien.pysapien.physx.PhysxMaterial(
                static_friction=frictions[i], dynamic_friction=frictions[i], restitution=0
            )
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=half_size, material=material, density=densities[i])
            builder.add_box_visual(half_size=half_size, material=sapien.render.RenderMaterial(base_color=color))
            builder.initial_pose = sapien.Pose(p=[0.25, 0.3, half_widths[i]])  # away from the bin at creation
            builder.set_scene_idxs([i])
            bar = builder.build(name=f"bar-{i}")
            bars.append(bar)
            self.remove_from_state_dict_registry(bar)
        self.bar = Actor.merge(bars, name="bar")
        self.add_to_state_dict_registry(self.bar)

        # The bin: PlaceCube's construction
        bin_color = sapien.render.RenderMaterial(base_color=[1.0, 1.0, 1.0, 1.0])
        thickness = 0.005
        self.bin_thickness = thickness
        bin_half_sizes_x = per_env(cfg.bin_half_size_x_range)
        bin_half_sizes_y = per_env(cfg.bin_half_size_y_range)
        bin_half_sizes_z = per_env(cfg.bin_half_size_z_range)
        self.bin_half_sizes_x = common.to_tensor(bin_half_sizes_x, device=self.device)
        self.bin_half_sizes_y = common.to_tensor(bin_half_sizes_y, device=self.device)
        self.bin_half_sizes_z = common.to_tensor(bin_half_sizes_z, device=self.device)
        self.bin_dimensions = torch.stack([self.bin_half_sizes_x, self.bin_half_sizes_y, self.bin_half_sizes_z], dim=-1)
        bins = []
        for i in range(self.num_envs):
            bin_half_size = [bin_half_sizes_x[i], bin_half_sizes_y[i], bin_half_sizes_z[i]]
            builder = self.scene.create_actor_builder()
            bin_center_pose = sapien.Pose([0.0, 0.0, thickness / 2])
            bin_center_half_size = [bin_half_size[0], bin_half_size[1], thickness / 2]
            builder.add_box_collision(pose=bin_center_pose, half_size=bin_center_half_size)
            builder.add_box_visual(pose=bin_center_pose, half_size=bin_center_half_size, material=bin_color)
            for j in [-1, 1]:
                y = j * bin_center_half_size[1]
                wall_pose = sapien.Pose([0, y, bin_half_size[2]])
                wall_half_size = [bin_half_size[0], thickness / 2, bin_half_size[2]]
                builder.add_box_collision(pose=wall_pose, half_size=wall_half_size)
                builder.add_box_visual(pose=wall_pose, half_size=wall_half_size, material=bin_color)
                x = j * bin_center_half_size[0]
                wall_pose = sapien.Pose([x, 0, bin_half_size[2]])
                wall_half_size = [thickness / 2, bin_half_size[1], bin_half_size[2]]
                builder.add_box_collision(pose=wall_pose, half_size=wall_half_size)
                builder.add_box_visual(pose=wall_pose, half_size=wall_half_size, material=bin_color)
            builder.initial_pose = sapien.Pose(p=[0.25, -0.3, bin_half_size[2]])  # away from the bar at creation
            builder.set_scene_idxs([i])
            bin_actor = builder.build(name=f"bin-{i}")
            bins.append(bin_actor)
            self.remove_from_state_dict_registry(bin_actor)
        self.bin = Actor.merge(bins, name="bin")
        self.add_to_state_dict_registry(self.bin)

        # Keep both robots, the bar and the bin in the picture; everything else is background
        if self.apply_greenscreen:
            for arm in self.arms:
                self.remove_object_from_greenscreen(arm.robot)
            self.remove_object_from_greenscreen(self.bar)
            self.remove_object_from_greenscreen(self.bin)

        self.rest_qpos = common.to_tensor(self.rest_qpos, device=self.device)
        self.table_pose = Pose.create_from_pq(p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2))
        self.handover_point = common.to_tensor(self.handover_point_xyz, device=self.device)

        self._load_camera_mount()
        self._randomize_robot_color()

        # Per-env task state: which arm picks (0 left, 1 right), and whether the mid-air handover has happened
        self.picker = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.handed_over = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Reward v3: the bar has reached the handover point in the picker's grip this episode (latched)
        self.reached_handover = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        goal_builder = self.scene.create_actor_builder()
        goal_builder.add_sphere_visual(radius=0.01, material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 1]))
        goal_builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        self.goal_site = goal_builder.build_kinematic(name="goal_site")
        self._hidden_objects.append(self.goal_site)

    # ------------------------------------------------------------------ episodes
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        cfg = self.domain_randomization_config
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.table_scene.table.set_pose(self.table_pose)

            # The arms: spacing and yaw drawn per episode with domain randomization
            if self.domain_randomization:
                lo, hi = cfg.arm_separation_range
                sep = torch.rand(b) * (hi - lo) + lo
                yaws = (2 * torch.rand(2, b) - 1) * cfg.arm_yaw_noise
            else:
                sep = torch.full((b,), self._nominal_separation())
                yaws = torch.zeros(2, b)
            pan_axes = []
            for a, (arm, sign) in enumerate(zip(self.arms, (1.0, -1.0))):
                base_y = sign * sep / 2
                arm.robot.set_pose(arm_yaw_pose(torch.zeros(b), base_y, yaws[a]))
                pan_axes.append(torch.stack([PAN_AXIS_X * torch.cos(yaws[a]), base_y + PAN_AXIS_X * torch.sin(yaws[a])], -1))
                arm.robot.set_qpos(
                    self.rest_qpos + torch.randn(size=(b, self.rest_qpos.shape[-1])) * cfg.initial_qpos_noise_scale
                )

            # Roles: the picker's outward side is +y for the left arm, -y for the right arm
            picker = torch.randint(0, 2, (b,))
            self.picker[env_idx] = picker
            self.handed_over[env_idx] = False
            self.reached_handover[env_idx] = False
            outward = torch.where(picker == 0, 1.0, -1.0)
            picker_axis = torch.where((picker == 0)[:, None], pan_axes[0], pan_axes[1])
            receiver_axis = torch.where((picker == 0)[:, None], pan_axes[1], pan_axes[0])

            # The bar: 15-25 cm in front of the picker's pan axis, 2-10 cm to its outer side, any yaw, lying flat
            bar_xyz = torch.zeros((b, 3))
            bar_xyz[:, 0] = picker_axis[:, 0] + torch.rand(b) * 0.10 + 0.15
            bar_xyz[:, 1] = picker_axis[:, 1] + outward * (torch.rand(b) * 0.08 + 0.02)
            bar_xyz[:, 2] = self.bar_half_widths[env_idx]
            bar_yaw = torch.rand(b) * np.pi
            self.bar.set_pose(arm_yaw_pose(bar_xyz[:, 0], bar_xyz[:, 1], bar_yaw, bar_xyz[:, 2]))

            # The bin: 16-24 cm in front of the receiver's pan axis, 6-12 cm to the receiver's outer side, any yaw
            bin_xyz = torch.zeros((b, 3))
            bin_xyz[:, 0] = receiver_axis[:, 0] + torch.rand(b) * 0.08 + 0.16
            bin_xyz[:, 1] = receiver_axis[:, 1] - outward * (torch.rand(b) * 0.06 + 0.06)
            bin_yaw = torch.rand(b) * np.pi
            self.bin.set_pose(arm_yaw_pose(bin_xyz[:, 0], bin_xyz[:, 1], bin_yaw))

            goal_xyz = bin_xyz.clone()
            goal_xyz[:, 2] = self.bin_thickness + self.bar_half_widths[env_idx]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    # ------------------------------------------------------------------ observations
    def _get_obs_agent(self):
        """Per arm, in ARM_NAMES order: noisy joint angles, then the controller's targets (12 numbers per arm)."""
        obs = dict()
        for name, arm in zip(ARM_NAMES, self.arms):
            qpos = arm.robot.get_qpos()
            if self.domain_randomization and self.domain_randomization_config.robot_qpos_noise_std > 0:
                qpos = qpos + torch.randn_like(qpos) * self.domain_randomization_config.robot_qpos_noise_std
            arm_obs = dict(noisy_qpos=qpos)
            controller_state = arm.controller.get_state()
            if len(controller_state) > 0:
                arm_obs.update(controller=controller_state)
            obs[name] = arm_obs
        return obs

    def _get_obs_extra(self, info: dict):
        obs = dict()
        if self.obs_mode_struct.state:
            obs.update(
                picker=self.picker.float(),
                handed_over=self.handed_over.float(),
                bar_pose=self.bar.pose.raw_pose,
                bin_pose=self.bin.pose.raw_pose,
                left_tcp_pose=self.arms[0].tcp_pose.raw_pose,
                right_tcp_pose=self.arms[1].tcp_pose.raw_pose,
            )
        return obs

    # ------------------------------------------------------------------ task logic
    @staticmethod
    def _openness(arm) -> torch.Tensor:
        """How far the gripper is open, 0 (closed) to 1 (open), from its joint limits."""
        lo, hi = arm.robot.get_qlimits()[0, -1, :]
        return (arm.robot.get_qpos()[:, -1] - lo) / (hi - lo)

    def _by_role(self, left: torch.Tensor, right: torch.Tensor, role: str) -> torch.Tensor:
        """Select per env the picker's (role='picker') or the receiver's value from the left/right arms' values."""
        is_left = (self.picker == 0) if role == "picker" else (self.picker == 1)
        if left.dim() > 1:
            is_left = is_left[:, None]
        return torch.where(is_left, left, right)

    def _bar_axis(self) -> torch.Tensor:
        """The bar's long axis in the world frame."""
        return self.bar.pose.to_transformation_matrix()[..., :3, 0]

    def _arm_contacts(self):
        """Contacts between the two arms over the same 16 link pairs as before, split: (jaws, body). jaws: a gripper link or
        moving jaw of one arm against a gripper link or moving jaw of the other (a handover brings them ~3.6 cm apart);
        body: any pair that involves a wrist or forearm link."""
        left, right = self.arms
        jaws = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        body = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for a in CONTACT_LINKS:
            for c in CONTACT_LINKS:
                forces = self.scene.get_pairwise_contact_forces(left.robot.links_map[a], right.robot.links_map[c])
                hit = torch.linalg.norm(forces, axis=-1) >= 1e-2
                if a in JAW_LINKS and c in JAW_LINKS:
                    jaws |= hit
                else:
                    body |= hit
        return jaws, body

    def _arms_touching(self) -> torch.Tensor:
        """True where any of the two arms' gripper, jaw, wrist or forearm links touch each other."""
        jaws, body = self._arm_contacts()
        return jaws | body

    def evaluate(self):
        left, right = self.arms
        bar_p = self.bar.pose.p
        axis = self._bar_axis()

        # Per arm, then per role
        grasping = [arm.is_grasping(self.bar) for arm in self.arms]
        if self.GRASP_NEEDS_OPEN_JAWS:
            grasping = [g & (self._openness(arm) >= self.MIN_GRASP_OPENNESS) for g, arm in zip(grasping, self.arms)]
        touching_bar = [arm.is_touching(self.bar) for arm in self.arms]
        touching_bin = [arm.is_touching(self.bin) for arm in self.arms]
        touching_table = [arm.is_touching(self.table_scene.table) for arm in self.arms]
        static = [arm.is_static() for arm in self.arms]
        picker_grasp = self._by_role(grasping[0], grasping[1], "picker")
        receiver_grasp = self._by_role(grasping[0], grasping[1], "receiver")
        picker_touch = self._by_role(touching_bar[0], touching_bar[1], "picker")
        receiver_touch = self._by_role(touching_bar[0], touching_bar[1], "receiver")

        bar_lifted = bar_p[:, 2] >= self.bar_half_widths + 0.02
        both_hold = picker_grasp & receiver_grasp & bar_lifted
        self.handed_over |= both_hold  # latched until the next reset

        # The bar in the bin: both of its ends inside the bin's walls (in the bin's own frame), its low end on the floor
        bin_mat = self.bin.pose.to_transformation_matrix()
        ends = [bar_p + s * axis * self.bar_half_lengths[:, None] for s in (1.0, -1.0)]
        inside = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for end in ends:
            local = (bin_mat[..., :3, :3].transpose(-1, -2) @ (end - self.bin.pose.p)[..., None])[..., 0]
            inside &= torch.abs(local[:, 0]) < self.bin_half_sizes_x - self.bin_thickness
            inside &= torch.abs(local[:, 1]) < self.bin_half_sizes_y - self.bin_thickness
        bar_low = torch.minimum(ends[0][:, 2], ends[1][:, 2]) - self.bar_half_widths
        bar_in_bin = inside & (bar_low < self.bin_thickness + 0.01)
        bar_static = torch.linalg.norm(self.bar.linear_velocity, axis=-1) <= 2e-2

        arms_jaws, arms_body = self._arm_contacts()
        arms_touching = arms_jaws | arms_body
        self.reached_handover |= picker_grasp & (torch.linalg.norm(bar_p - self.handover_point, axis=1) < 0.05)
        success = (
            self.handed_over & bar_in_bin & bar_static
            & ~touching_bar[0] & ~touching_bar[1] & ~touching_bin[0] & ~touching_bin[1]
            & static[0] & static[1]
        )
        return {
            "success": success,
            "handed_over": self.handed_over.clone(),
            "both_hold": both_hold,
            "picker_grasp": picker_grasp,
            "receiver_grasp": receiver_grasp,
            "picker_touch_bar": picker_touch,
            "receiver_touch_bar": receiver_touch,
            "bar_lifted": bar_lifted,
            "bar_in_bin": bar_in_bin,
            "bar_static": bar_static,
            "arms_touching": arms_touching,
            "arms_touching_jaws": arms_jaws,
            "arms_touching_body": arms_body,
            "reached_handover": self.reached_handover.clone(),
            "robot_touching_table": touching_table[0] | touching_table[1],
            "robot_touching_bin": touching_bin[0] | touching_bin[1],
            "picker_is_left": self.picker == 0,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        left, right = self.arms
        bar_p = self.bar.pose.p
        axis = self._bar_axis()
        grip_offset = (self.bar_half_lengths - 0.012)[:, None]  # a jaw pad's centre, 1.2 cm in from each end
        end_a, end_b = bar_p + axis * grip_offset, bar_p - axis * grip_offset

        tcp_p = self._by_role(left.tcp_pos, right.tcp_pos, "picker")
        tcp_r = self._by_role(left.tcp_pos, right.tcp_pos, "receiver")
        # The picker takes the end nearer to it; the receiver the other one
        a_nearer = torch.linalg.norm(tcp_p - end_a, axis=1) <= torch.linalg.norm(tcp_p - end_b, axis=1)
        picker_end = torch.where(a_nearer[:, None], end_a, end_b)
        receiver_end = torch.where(a_nearer[:, None], end_b, end_a)

        def openness(arm):
            lo, hi = arm.robot.get_qlimits()[0, -1, :]
            return (arm.robot.get_qpos()[:, -1] - lo) / (hi - lo)

        picker_open = self._by_role(openness(left), openness(right), "picker")
        receiver_open = self._by_role(openness(left), openness(right), "receiver")

        # Stage 1, 0-2: the picker reaches its end of the bar
        reward = 2 * (1 - torch.tanh(5 * torch.linalg.norm(tcp_p - picker_end, axis=1)))

        # Stage 2, 2-4: the picker holds the bar and brings it to the handover point
        bar_to_handover = torch.linalg.norm(bar_p - self.handover_point, axis=1)
        carry = 2 * (1 - torch.tanh(5 * bar_to_handover))
        reward = torch.where(info["picker_grasp"], 2 + carry, reward)

        receiver_gap = torch.linalg.norm(tcp_r - receiver_end, axis=1)
        if self.REWARD_VERSION == 3:
            # v3 stage 3, 4-5.5: the bar has reached the handover point (latched) and is still within 10 cm of it, so
            # a nudge from the receiver's approach does not drop the score to stage 2 (v1/v2: a 5 cm cliff)
            at_handover = info["picker_grasp"] & info["reached_handover"] & (bar_to_handover < 0.10)
            reward = torch.where(at_handover, 4 + 1.5 * (1 - torch.tanh(5 * receiver_gap)), reward)
            # v3 stage 4, 6.5-7: both hold; gripping pays a full point over the best hover (v1/v2: +0.1)
            reward = torch.where(info["both_hold"], 6.5 + 0.5 * picker_open, reward)
        else:
            # Stage 3, 4-6: at the handover point, the receiver reaches the free end
            at_handover = info["picker_grasp"] & (bar_to_handover < 0.05)
            receiver_reach = 2 * (1 - torch.tanh(5 * receiver_gap))
            reward = torch.where(at_handover, 4 + receiver_reach, reward)

            # Stage 4, 6-7: both hold the lifted bar; the picker is paid to let go
            reward = torch.where(info["both_hold"], 6 + picker_open, reward)

        if self.REWARD_VERSION in (2, 3, 4):
            picker_gap = torch.linalg.norm(tcp_p - picker_end, axis=1)  # picker's grip point to its end of the bar
            return self._reward_after_handover_v2(reward, info, bar_p, axis, receiver_open, picker_gap)

        # Stage 5, 8-10: after a mid-air handover, the receiver alone carries the bar over the bin. Starts a full
        # point above stage 4's top (7): B002 stalled with both arms holding when release paid only ~0.4 more
        goal = self.bin.pose.p.clone()
        goal[:, 2] = self.bin_thickness + self.bar_half_widths
        bar_to_goal_xy = torch.linalg.norm(goal[:, :2] - bar_p[:, :2], axis=1)
        bin_radius = torch.linalg.norm(self.bin_dimensions[:, :2], axis=1)
        z_far = goal[:, 2] + 2 * self.bin_half_sizes_z + 0.03  # above the walls until it is over the bin
        bar_to_goal_z = torch.where(bar_to_goal_xy <= bin_radius, torch.abs(goal[:, 2] - bar_p[:, 2]), torch.abs(z_far - bar_p[:, 2]))
        bar_to_goal = torch.linalg.norm(goal - bar_p, axis=1)
        if self.CARRY_DISTANCE_SHAPE == "linear":
            near = 1 - torch.clamp(bar_to_goal / self.CARRY_LINEAR_RANGE, 0, 1)
        else:
            near = 1 - torch.tanh(5 * bar_to_goal)
        place = near + (1 - torch.tanh(10 * bar_to_goal_z))
        carrying = self.handed_over & info["receiver_grasp"] & ~info["picker_touch_bar"]
        reward = torch.where(carrying, 8 + place, reward)

        # Stage 6, 10-15: the bar is in the bin; release and keep both arms still
        arm_speed = torch.linalg.norm(left.robot.get_qvel()[:, :-1], axis=1) + torch.linalg.norm(right.robot.get_qvel()[:, :-1], axis=1)
        settled = 1 - torch.tanh(10 * arm_speed)
        released = (~info["receiver_touch_bar"]).float()
        in_bin = self.handed_over & info["bar_in_bin"]
        reward = torch.where(in_bin, 10 + place + released + receiver_open + settled, reward)

        # Success, 16
        reward = torch.where(info["success"], torch.full_like(reward, self.MAX_REWARD), reward)

        # Penalties
        reward -= 6 * info["robot_touching_table"].float()
        reward -= 3 * info["robot_touching_bin"].float()
        reward -= 2 * info["arms_touching"].float()
        reward -= 1 * (~info["bar_lifted"] & ~self.handed_over).float()  # pick it up promptly
        return reward

    def _reward_after_handover_v2(self, reward, info, bar_p, axis, receiver_open, picker_gap):
        """Reward v2 after the latched handover (stages 1-4 as v1, already in `reward`).

        v1 gated each stage on contact flags. Between two stages the flags of neither held, so the score fell back to
        stage 1 (about 0-2): while the picker's jaws open but still touch, while a released bar falls into the bin.
        Its height target also jumped at the bin's rim. Here every state on the path both hold -> release -> carry ->
        over the bin -> lowered or dropped in -> let go scores at least the one before (checked in reward_probe.py).
        """
        left, right = self.arms
        # Progress of the bar toward the bin: horizontal pull, the same at every distance (B006), and height
        bin_mat = self.bin.pose.to_transformation_matrix()
        local = (bin_mat[..., :3, :3].transpose(-1, -2) @ (bar_p - self.bin.pose.p)[..., None])[..., 0]
        over_bin = ((torch.abs(local[:, 0]) < self.bin_half_sizes_x - self.bin_thickness)
                    & (torch.abs(local[:, 1]) < self.bin_half_sizes_y - self.bin_thickness))
        near = 1 - torch.clamp(torch.linalg.norm((bar_p - self.bin.pose.p)[:, :2], axis=1) / self.CARRY_LINEAR_RANGE, 0, 1)
        ends_z = torch.stack([bar_p[:, 2] + s * axis[:, 2] * self.bar_half_lengths for s in (1.0, -1.0)])
        bar_low = ends_z.min(dim=0).values - self.bar_half_widths  # as evaluate()'s bin test
        z_clear = self.bin_thickness + 2 * self.bin_half_sizes_z + 0.01  # 1 cm above the walls
        clearance = 1 - torch.tanh(10 * torch.clamp(z_clear - bar_low, min=0))  # outside: keep above the walls
        lowered = torch.clamp((z_clear - bar_low) / (z_clear - self.bin_thickness), 0, 1)  # inside: lower toward the floor
        height = torch.where(over_bin, 1 + lowered, clearance)  # 0-2; 1 on both sides of the rim at wall-top height
        place = 1.2 * near + 0.4 * height  # 0-2

        # Stage 5, 7-10: handover done, the picker no longer squeezing; +1 once it is off the bar. Includes a bar
        # released over the bin and falling in (v1 scored that about 0)
        holding = self.handed_over & info["receiver_grasp"] & ~info["picker_grasp"]
        falling_in = self.handed_over & ~info["receiver_grasp"] & ~info["picker_grasp"] & over_bin & ~info["bar_in_bin"]
        off_bar = (~info["picker_touch_bar"]).float()
        if self.REWARD_VERSION == 4:
            # v4: the picker's way off the bar is paid as it goes, not only once its jaws are off: 0.5 for the jaws off
            # plus 0.5 for its grip point's distance from its end of the bar, full at PICKER_CLEAR_DISTANCE. B011's
            # failed releases (79 of 99) opened but left a jaw touching: v2's +1 had no slope to follow
            off_bar = 0.5 * off_bar + 0.5 * torch.clamp(picker_gap / self.PICKER_CLEAR_DISTANCE, 0, 1)
        reward = torch.where(holding | falling_in, 7 + off_bar + place, reward)

        # Stage 6, 12-15: the bar in the bin; let go and keep both arms still
        arm_speed = torch.linalg.norm(left.robot.get_qvel()[:, :-1], axis=1) + torch.linalg.norm(right.robot.get_qvel()[:, :-1], axis=1)
        settled = 1 - torch.tanh(10 * arm_speed)
        released = (~info["receiver_touch_bar"]).float()
        in_bin = self.handed_over & info["bar_in_bin"]
        reward = torch.where(in_bin, 10 + place + released + receiver_open + settled, reward)

        # Success, 16
        reward = torch.where(info["success"], torch.full_like(reward, self.MAX_REWARD), reward)

        # Penalties as v1, except touching the bin once the bar is in it: placing it there brushes the walls, and
        # success still requires letting go of everything
        reward -= 6 * info["robot_touching_table"].float()
        reward -= 3 * (info["robot_touching_bin"] & ~in_bin).float()
        if self.REWARD_VERSION == 3:
            # v3: jaw-to-jaw contact is not penalised while the picker is still on the bar at the handover point (the
            # approach, both holding, the release); wrist and forearm contact always is
            handover_phase = info["reached_handover"] & (info["picker_grasp"] | info["picker_touch_bar"])
            reward -= 2 * (info["arms_touching_body"] | (info["arms_touching_jaws"] & ~handover_phase)).float()
        else:
            reward -= 2 * info["arms_touching"].float()
        reward -= 1 * (~info["bar_lifted"] & ~self.handed_over).float()
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / self.MAX_REWARD


@register_env("SO101HandoverBar-v1", max_episode_steps=120)
class HandoverBar(Handover):
    pass


@register_env("SO101HandoverBarBinInState-v1", max_episode_steps=120)
class HandoverBarBinInState(Handover):
    """B004 diagnostic, simulation only: the bin's position and heading are added to the policy's state input
    (4 numbers: x, y, cos yaw, sin yaw), so the bin is known even when no camera sees it. Everything else as v1."""

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        q = self.bin.pose.q
        yaw = 2 * torch.atan2(q[:, 3], q[:, 0])  # the bin only turns about z
        obs["bin_xy_yaw"] = torch.stack([self.bin.pose.p[:, 0], self.bin.pose.p[:, 1], torch.cos(yaw), torch.sin(yaw)], dim=1)
        return obs


@register_env("SO101HandoverBarLinearCarry-v1", max_episode_steps=120)
class HandoverBarLinearCarry(Handover):
    """B006: as v1, but the carry is paid 1 - d / 40 cm for the bar's distance d to the goal (v1: 1 - tanh(5 d),
    almost flat beyond ~20 cm, where the carry starts). Stages, bands and success unchanged."""

    CARRY_DISTANCE_SHAPE = "linear"


@register_env("SO101HandoverBarFirstFrame-v1", max_episode_steps=120)
class HandoverBarFirstFrame(Handover):
    """B007, first-frame memory: each wrist camera's image from the episode's first step is kept and given to the
    policy at every step beside the current one. Sensor order, so channel order after flattening: left now, right now,
    left first, right first (12 RGB channels). At step 0 the receiver's camera sees the bin in 99% of episodes
    (B003 A2); the kept frame holds that view after the bin leaves the camera. On the real rig: store one frame."""

    IMAGE_CHUNK_CAMERAS = ("left", "right", "left", "right")  # colour jitter: a camera's kept frame jitters with it

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        if not hasattr(self, "_first_pending"):
            self._first_pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._first_pending[env_idx] = True  # capture these envs' frames until reset() ends (after the camera sync)

    def reset(self, *args, **kwargs):
        # The base reset observes twice: before and after syncing the wrist cameras. Keep capturing until the
        # second (synced) observation has been stored, then stop.
        obs, info = super().reset(*args, **kwargs)
        self._first_pending[:] = False
        return obs, info

    def _get_obs_sensor_data(self, apply_texture_transforms: bool = True):
        obs = super()._get_obs_sensor_data(apply_texture_transforms)
        cams = [name for name in obs if name.endswith("_wrist")]
        if not hasattr(self, "_first_frames"):
            self._first_frames = {name: {k: v.clone() for k, v in obs[name].items()} for name in cams}
        for name in cams:
            for k, v in obs[name].items():
                pending = self._first_pending.to(v.device)
                self._first_frames[name][k][pending] = v[pending]
        for name in cams:
            obs[f"{name}_first"] = {k: v.clone() for k, v in self._first_frames[name].items()}
        return obs


@register_env("SO101HandoverBarV2-v1", max_episode_steps=120)
class HandoverBarV2(Handover):
    """Task 1 with reward v2 (after the handover, scores follow the bar's progress; no dips)."""

    REWARD_VERSION = 2


@register_env("SO101HandoverBarV2Grip-v1", max_episode_steps=120)
class HandoverBarV2Grip(Handover):
    """Task 1, reward v2, and a grasp only when the jaws are stopped by the bar (GRASP_NEEDS_OPEN_JAWS)."""

    REWARD_VERSION = 2
    GRASP_NEEDS_OPEN_JAWS = True


@register_env("SO101HandoverBarV4-v1", max_episode_steps=120)
class HandoverBarV4(Handover):
    """B016: task 1 with reward v4 = v2 plus a slope for the picker moving off the bar after it lets go."""

    REWARD_VERSION = 4


@register_env("SO101HandoverBarV2TopCamLeHome-v1", max_episode_steps=120)
class HandoverBarV2TopCamLeHome(LeHomeTopCamera, Handover):
    """B010, as run: task 1 with reward v2 and a top camera placed as LeHome's, rolled the sim's way (image upside down
    against the real camera's). Kept for reproducing B010; new runs use v2."""

    REWARD_VERSION = 2
    TOP_CAMERA_UP = (0, 0, 1)


@register_env("SO101HandoverBarV2TopCamLeHome-v2", max_episode_steps=120)
class HandoverBarV2TopCamLeHomeV2(LeHomeTopCamera, Handover):
    """Task 1 with reward v2 and the top camera as the organisers' real one: placed and rolled so its image matches theirs."""

    REWARD_VERSION = 2
