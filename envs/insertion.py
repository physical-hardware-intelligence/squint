"""Phi task 2: two SO-101 arms insert a peg into a socket in mid-air.

Layout as task 1 (`handover.py`): world frame midway between the arms' base origins, x forward, y to the robots' left;
the left arm's base at y = +s/2, the right arm's at y = -s/2, both facing +x, s = 46 cm pan axis to pan axis.
The left arm picks the socket, the right arm the peg (ALOHA's Bimanual Insertion assignment). Each object starts lying
in front of its own arm, its long axis pointing roughly at the other arm.

Parts (`insertion_parts.py`, printed from `experiments/bimanual/make_insert_stl.py`): a 22 x 22 x 90 mm peg and a
40 x 40 x 80 mm socket with a 32 x 32 mm hole 50 mm deep (5 mm clearance a side); the other 30 mm of the socket is solid.

Two tasks share the scene, the observations and the geometry:
- SO101InsertPeg-v1: ALOHA's success test. The peg 25 mm into the hole while both arms still hold their part.
- SO101InsertPegRetain-v1: the peg fully inserted, then the peg arm lets go and moves away and the peg stays in the
  socket, which the left arm still holds up. Checked every step, so a peg that slides out stops counting.

Where a released peg stays (`experiments/bimanual/insert_probe.py`: socket held rigidly, peg released on the axis at
25, 42, 46 or 48 mm, 2 s of physics, friction 0.3 and 0.7): released at 25 mm it falls out at every pitch from 10 deg
mouth-down to 20 deg mouth-up. Mouth up 10-20 deg, it slides to the floor from 42 mm at both frictions; level, from
46 mm (friction 0.3) or 42 mm (0.7); 10 deg mouth-down, it falls out at friction 0.3 from any depth. A seated peg rests
on the floor, 50.5 mm in. So the retain task pays depth to the floor and pays the release only with the peg seated
(45 mm) and the mouth >= 10 deg up, the region where every case stayed in.
"""

from dataclasses import dataclass
from typing import Any, Sequence, Union

import dacite
import numpy as np
import sapien
import torch

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat

from .base_random_env import DefaultRandomizationConfig
from .bimanual_base import ARM_NAMES, BimanualWristCameraEnv, arm_yaw_pose, nominal_yaw_quat
from .handover import CONTACT_LINKS, PAN_AXIS_X
from .insertion_parts import (PEG_HALF_LENGTH, PEG_HALF_WIDTH, SOCKET_HALF_HOLE, SOCKET_HALF_LENGTH, SOCKET_HALF_OUTER,
                              SOCKET_HOLE_DEPTH)
from .robot.so101 import SO101

GRIP_INSET = 0.025  # the peg arm's reach target: 25 mm in from the peg's outer end, so 65 mm can go into the hole
LIFT_OK = 0.03  # an object counts as lifted 3 cm above its resting height
HOLE_TOL = 0.001  # 'in the hole' allows 1 mm past the walls (contact penetration, a few tenths of a mm, plus margin)
FLOOR_TOL = 0.005  # a tip more than 5 mm past the hole floor is behind the socket's closed end, not in the hole
INSERT_DEPTH = 0.025  # v1 success depth (ALOHA's test is contact with the pins; 25 mm is half the hole)
SEAT_MARGIN = 0.005  # seated = within 5 mm of the hole floor (45 mm nominal)
FULL_MARGIN = 0.002  # depth is paid up to 2 mm short of the floor (48 mm nominal), reachable under every size draw
ALIGN_MIN = float(np.cos(np.deg2rad(20)))  # v1 success: the peg within 20 deg of the socket axis
TILT_FULL = 20.0  # retain: mouth-up tilt paid in full at 20 deg (a peg released 42 mm deep slides to the floor, both frictions)
TILT_OK = 10.0  # retain: opening and letting go pay in full only with the mouth >= 10 deg up (seated pegs stayed at both frictions)
RETREAT_OK = 0.04  # retain success: the peg arm's TCP >= 4 cm from the peg's centre line
RETREAT_FULL = 0.05  # retain: moving away is paid up to 5 cm
# Fingertip distance vs jaw angle in this simulator: 14.9 mm + 65 mm per rad from 0 to 0.4 rad (insert_probe.py checks it
# within 0.5 mm; measured 0.04-0.17 mm). A larger angle opens the jaw. Below 0 rad the tips meet and the line no longer
# holds; the reward reads the jaw only between touching the peg (0.11 rad) and open (0.26 rad)
JAW_TIP_DIST_0 = 0.0149
JAW_TIP_DIST_PER_RAD = 0.065
JAW_OPEN_PAST_CONTACT = 0.15  # rad past touching the peg = tips ~5 mm clear on each side


@dataclass
class InsertRandomizationConfig(DefaultRandomizationConfig):
    """Domain randomization for the insertion, extending the wrist camera randomization."""
    robot_qpos_noise_std: float = np.deg2rad(5)
    size_scale_range: Sequence[float] = (0.97, 1.03)  # print tolerance [estimate]; one scale per part, all its dimensions
    friction_range: Sequence[float] = (0.3, 0.7)
    density_range: Sequence[float] = (250, 450)  # kg/m^3: a 15%-infill PLA print [estimate until weighed], as task 1
    peg_color: Sequence[float] = (1.0, 0.0, 0.0)
    socket_color: Sequence[float] = (0.0, 0.3, 1.0)
    arm_separation_range: Sequence[float] = (0.455, 0.465)
    arm_yaw_noise: float = np.deg2rad(1)


def dense_reward(s: dict) -> torch.Tensor:
    """The insertion reward from per-env quantities, so `insert_probe.py` can check it on constructed states.
    Rule from task 1: along the intended path every state scores at least the one before, and a regression (dropping a
    part, the peg falling out) scores less.

    Stage 1, 0-6: per arm, reach its grip point (0-1), then hold it (2) and lift it (0-1); the two arms add up.
    Stage 2, 6-10, both parts held and lifted: the peg tip approaches the mouth along the socket axis, sideways distance
    counted double (`near`, 0-1), and points into it (`align`, counted only as the tip nears). v1: 2.5 near + 1.5 align.
    Retain: 1.75 near + 1.25 align + 1 for the mouth tilted up (0-20 deg), so the socket is presented the way that keeps
    a released peg in. No fixed meeting point: the arms choose where to meet.
    Stage 3, from 10, both held and the tip in the hole: depth 0-25 mm pays 0-4 (scaled by alignment).
        v1: success (25 mm, aligned, both held) pays 16.
        Retain: 25 mm to full depth (48 mm) pays 14-16, so the peg is pushed to the floor, where it stays when released.
    Retain stage 4, seated (within 5 mm of the floor, 15.7-16) with the socket held up: opening the peg jaw adds up to
        0.5, in full only with the mouth >= 10 deg up. Opening shallower or mouth-down adds nothing.
    Retain stage 5, the peg arm off the peg and the peg still seated: moving away 0-5 cm, 16.5-17.5, also only with the
        mouth >= 10 deg up (with it level or down, letting go pays the same 16 as holding). Success, 18, whatever the tilt.
        A peg that falls out after the release drops the reward to stage 1 (at most 4), so letting go early is a loss.
    Penalties: a robot touching the table -2, the arms touching each other -1."""
    reward = s["socket_progress"] + s["peg_progress"]
    up, in_hole = s["both_lifted"], s["in_hole"]
    align = s["align"].clamp(0, 1)
    if s["retain"]:
        stage2 = 6 + 1.75 * s["near"] + 1.25 * align * s["near"] + 1.0 * s["tilt"]
    else:
        stage2 = 6 + 2.5 * s["near"] + 1.5 * align * s["near"]
    reward = torch.where(up, torch.maximum(stage2, reward), reward)
    stage3 = 10 + 4 * (s["axial"] / INSERT_DEPTH).clamp(0, 1) * align
    if s["retain"]:
        deep = ((s["axial"] - INSERT_DEPTH) / (s["full_depth"] - INSERT_DEPTH)).clamp(0, 1)
        stage3 = stage3 + 2 * deep * align
        stage3 = stage3 + 0.5 * s["opened"] * s["tilt_ok"] * (s["seated"] & s["socket_up"]).float()
    reward = torch.where(up & in_hole, torch.maximum(stage3, reward), reward)
    if s["retain"]:
        # scaled by the same mouth-up factor as opening, so letting go mouth-level or mouth-down pays no more than holding
        stage5 = 16 + s["tilt_ok"] * (0.5 + (s["retreat"] / RETREAT_FULL).clamp(0, 1))
        reward = torch.where(s["released"] & s["seated"] & s["socket_up"], torch.maximum(stage5, reward), reward)
    reward = torch.where(s["success"], torch.full_like(reward, s["max_reward"]), reward)
    return reward - 2 * s["robot_touching_table"].float() - 1 * s["arms_touching"].float()


class Insertion(BimanualWristCameraEnv):
    """
    **Task Description:**
    The left arm picks up a socket, the right arm a peg; together they insert the peg into the socket in mid-air.

    **Randomizations:**
    - each part's position (in front of its arm) and yaw (+-22.5 deg about pointing at the other arm)
    - with domain_randomization: part sizes (+-3%), mass, friction; arm spacing and yaw; cameras, lights, grippers

    **Success Conditions (SO101InsertPeg-v1):**
    - the peg's leading face at least 25 mm inside the hole, within the hole's cross-section, within 20 deg of the axis
    - both parts off the table, each held by its own arm
    """

    SUPPORTED_ROBOTS = [("so101", "so101")]
    SUPPORTED_OBS_MODES = ["none", "state", "state_dict", "rgb", "rgb+segmentation", "rgb+state", "rgb+segmentation+state",
                           "rgb+depth+segmentation", "rgb+depth+segmentation+state"]

    MAX_REWARD = 16.0
    RETAIN = False  # the retain task's success test and reward stages (InsertPegRetain)
    LIFT_HEIGHT = 0.08  # lift paid up to 8 cm above the part's resting height

    def __init__(
        self,
        *args,
        robot_uids=("so101", "so101"),
        control_mode="pd_joint_target_delta_pos",
        domain_randomization_config: Union[InsertRandomizationConfig, dict] = InsertRandomizationConfig(),
        domain_randomization=False,
        **kwargs,
    ):
        self.rest_qpos = SO101.keyframes["start"].qpos.tolist()
        self.domain_randomization_config = InsertRandomizationConfig()
        merged = self.domain_randomization_config.dict()
        if isinstance(domain_randomization_config, dict):
            common.dict_merge(merged, domain_randomization_config)
            self.domain_randomization_config = dacite.from_dict(
                data_class=InsertRandomizationConfig, data=merged, config=dacite.Config(strict=True)
            )
        elif isinstance(domain_randomization_config, InsertRandomizationConfig):
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

        peg_scale, socket_scale = per_env(cfg.size_scale_range), per_env(cfg.size_scale_range)
        peg_friction, socket_friction = per_env(cfg.friction_range), per_env(cfg.friction_range)
        peg_density, socket_density = per_env(cfg.density_range), per_env(cfg.density_range)
        t = lambda x: common.to_tensor(np.asarray(x, dtype=np.float32), device=self.device)  # noqa: E731
        self.peg_half_width = t(PEG_HALF_WIDTH * peg_scale)
        self.peg_half_length = t(PEG_HALF_LENGTH * peg_scale)
        self.socket_half_outer = t(SOCKET_HALF_OUTER * socket_scale)
        self.socket_half_hole = t(SOCKET_HALF_HOLE * socket_scale)
        self.socket_half_length = t(SOCKET_HALF_LENGTH * socket_scale)
        self.socket_hole_depth = t(SOCKET_HOLE_DEPTH * socket_scale)

        pegs, sockets = [], []
        peg_rgba = list(cfg.peg_color) + [1.0]
        socket_rgba = list(cfg.socket_color) + [1.0]
        for i in range(self.num_envs):
            # Peg: one box, long axis = local x
            hw, hl = PEG_HALF_WIDTH * peg_scale[i], PEG_HALF_LENGTH * peg_scale[i]
            mat = sapien.pysapien.physx.PhysxMaterial(static_friction=peg_friction[i], dynamic_friction=peg_friction[i], restitution=0)
            b = self.scene.create_actor_builder()
            b.add_box_collision(half_size=[hl, hw, hw], material=mat, density=peg_density[i])
            b.add_box_visual(half_size=[hl, hw, hw], material=sapien.render.RenderMaterial(base_color=peg_rgba))
            b.initial_pose = sapien.Pose(p=[0.25, -0.2, hw])
            b.set_scene_idxs([i])
            peg = b.build(name=f"peg-{i}")
            pegs.append(peg)
            self.remove_from_state_dict_registry(peg)

            # Socket: long axis = local x, the hole's mouth at +x. A solid block at the -x end, four walls around the hole
            so, sh, sl, sd = (v * socket_scale[i] for v in (SOCKET_HALF_OUTER, SOCKET_HALF_HOLE, SOCKET_HALF_LENGTH, SOCKET_HOLE_DEPTH))
            wall = so - sh
            mat = sapien.pysapien.physx.PhysxMaterial(static_friction=socket_friction[i], dynamic_friction=socket_friction[i], restitution=0)
            vis = sapien.render.RenderMaterial(base_color=socket_rgba)
            b = self.scene.create_actor_builder()
            block_half = (2 * sl - sd) / 2
            parts = [
                (sapien.Pose([-sl + block_half, 0, 0]), [block_half, so, so]),  # solid block
                (sapien.Pose([sl - sd / 2, 0, sh + wall / 2]), [sd / 2, so, wall / 2]),  # top wall
                (sapien.Pose([sl - sd / 2, 0, -sh - wall / 2]), [sd / 2, so, wall / 2]),  # bottom wall
                (sapien.Pose([sl - sd / 2, sh + wall / 2, 0]), [sd / 2, wall / 2, sh]),  # side walls
                (sapien.Pose([sl - sd / 2, -sh - wall / 2, 0]), [sd / 2, wall / 2, sh]),
            ]
            for pose, half in parts:
                b.add_box_collision(pose=pose, half_size=half, material=mat, density=socket_density[i])
                b.add_box_visual(pose=pose, half_size=half, material=vis)
            b.initial_pose = sapien.Pose(p=[0.25, 0.2, so])
            b.set_scene_idxs([i])
            socket = b.build(name=f"socket-{i}")
            sockets.append(socket)
            self.remove_from_state_dict_registry(socket)
        self.peg = Actor.merge(pegs, name="peg")
        self.socket = Actor.merge(sockets, name="socket")
        self.add_to_state_dict_registry(self.peg)
        self.add_to_state_dict_registry(self.socket)

        if self.apply_greenscreen:
            for arm in self.arms:
                self.remove_object_from_greenscreen(arm.robot)
            self.remove_object_from_greenscreen(self.peg)
            self.remove_object_from_greenscreen(self.socket)

        self.rest_qpos = common.to_tensor(self.rest_qpos, device=self.device)
        self.table_pose = Pose.create_from_pq(p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2))
        self._load_camera_mount()
        self._randomize_robot_color()

    @property
    def _default_human_render_camera_configs(self):
        """A view from in front of the rig for watching episodes (videos only; the policy never sees it)."""
        pose = sapien_utils.look_at(eye=[0.62, 0.0, 0.42], target=[0.20, 0.0, 0.08])
        return CameraConfig("render_camera", pose=pose, width=480, height=360, fov=np.deg2rad(60), near=0.01, far=100)

    # ------------------------------------------------------------------ episodes
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        cfg = self.domain_randomization_config
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.table_scene.table.set_pose(self.table_pose)
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
                arm.robot.set_qpos(self.rest_qpos + torch.randn(size=(b, self.rest_qpos.shape[-1])) * cfg.initial_qpos_noise_scale)

            # Socket (left arm): 15-22 cm in front of the left pan axis, 0-8 cm inward of it; mouth toward -y (the peg) +-22.5 deg
            jitter = (2 * torch.rand(2, b) - 1) * (np.pi / 8)
            sx = pan_axes[0][:, 0] + torch.rand(b) * 0.07 + 0.15
            sy = pan_axes[0][:, 1] - torch.rand(b) * 0.08
            self.socket.set_pose(arm_yaw_pose(sx, sy, -np.pi / 2 + jitter[0], self.socket_half_outer[env_idx]))
            # Peg (right arm): mirror image; the peg is symmetric, so its yaw is about +y
            px = pan_axes[1][:, 0] + torch.rand(b) * 0.07 + 0.15
            py = pan_axes[1][:, 1] + torch.rand(b) * 0.08
            self.peg.set_pose(arm_yaw_pose(px, py, np.pi / 2 + jitter[1], self.peg_half_width[env_idx]))

    # ------------------------------------------------------------------ observations
    def _get_obs_agent(self):
        """Per arm, in ARM_NAMES order: noisy joint angles, then the controller's targets (12 numbers per arm), as task 1."""
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
            obs.update(peg_pose=self.peg.pose.raw_pose, socket_pose=self.socket.pose.raw_pose,
                       left_tcp_pose=self.arms[0].tcp_pose.raw_pose, right_tcp_pose=self.arms[1].tcp_pose.raw_pose)
        return obs

    # ------------------------------------------------------------------ geometry
    def geometry(self) -> dict:
        """Everything the reward and success tests read about the two parts, from their poses only.
        socket_axis: unit vector out of the hole's mouth. mouth: centre of the hole's opening. tip: centre of the peg's
        leading face, the end further along the inward direction (-socket_axis). The nearer end to the mouth is not
        used: past 45 mm (half the peg) the outer end is nearer, and a deep peg would read as outside, pointing away.
        axial: how far the tip is inside the mouth plane (negative = outside). lateral: its distance from the axis.
        fit: the leading face's largest overshoot past the hole's walls, per socket axis (y, z), allowing for the peg's
        roll and tilt; <= 0 means the face lies inside the hole's cross-section. align: cosine between the peg's
        leading direction and the inward direction (1 = pointing straight in). pitch: the mouth's elevation, deg (> 0 up)."""
        s_mat = self.socket.pose.to_transformation_matrix()
        socket_axis, s_y, s_z = s_mat[..., :3, 0], s_mat[..., :3, 1], s_mat[..., :3, 2]
        mouth = self.socket.pose.p + socket_axis * self.socket_half_length[:, None]
        p_mat = self.peg.pose.to_transformation_matrix()
        peg_axis, p_y, p_z = p_mat[..., :3, 0], p_mat[..., :3, 1], p_mat[..., :3, 2]
        lead = ((peg_axis * socket_axis).sum(-1) <= 0)[:, None]  # +x end leads when the peg's x points into the hole
        tip_dir = torch.where(lead, peg_axis, -peg_axis)
        tip = self.peg.pose.p + tip_dir * self.peg_half_length[:, None]
        far = self.peg.pose.p - tip_dir * self.peg_half_length[:, None]
        rel = tip - mouth
        axial = -(rel * socket_axis).sum(-1)
        lateral = torch.linalg.norm(rel + axial[:, None] * socket_axis, axis=1)
        fit = []
        for u in (s_y, s_z):
            # the square face's half extent along u: its two edge directions (peg y, z) projected on u
            extent = self.peg_half_width * ((p_y * u).sum(-1).abs() + (p_z * u).sum(-1).abs())
            fit.append((rel * u).sum(-1).abs() + extent - self.socket_half_hole)
        return dict(
            socket_axis=socket_axis, peg_axis=peg_axis, mouth=mouth, tip=tip, axial=axial, lateral=lateral,
            fit=torch.maximum(*fit), align=-(tip_dir * socket_axis).sum(-1),
            pitch=torch.rad2deg(torch.asin(socket_axis[:, 2].clamp(-1, 1))),
            peg_grip=far + tip_dir * GRIP_INSET,
            socket_grip=self.socket.pose.p - socket_axis * (self.socket_hole_depth / 2)[:, None],  # solid block centre
        )

    def _arm_touching(self, arm, actor) -> torch.Tensor:
        touching = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for link in CONTACT_LINKS:
            forces = self.scene.get_pairwise_contact_forces(arm.robot.links_map[link], actor)
            touching |= torch.linalg.norm(forces, axis=-1) >= 1e-2
        return touching

    def _arms_touching(self) -> torch.Tensor:
        left, right = self.arms
        touching = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for a in CONTACT_LINKS:
            for c in CONTACT_LINKS:
                forces = self.scene.get_pairwise_contact_forces(left.robot.links_map[a], right.robot.links_map[c])
                touching |= torch.linalg.norm(forces, axis=-1) >= 1e-2
        return touching

    def evaluate(self):
        left, right = self.arms
        g = self.geometry()
        socket_grasp, peg_grasp = left.is_grasping(self.socket), right.is_grasping(self.peg)
        socket_touch, peg_touch = left.is_touching(self.socket), right.is_touching(self.peg)
        socket_lift = self.socket.pose.p[:, 2] - self.socket_half_outer  # centre above its resting height
        peg_lift = self.peg.pose.p[:, 2] - self.peg_half_width
        socket_lifted, peg_lifted = socket_lift >= LIFT_OK, peg_lift >= LIFT_OK
        # Held: grasped, or lifted and touched by its arm. A lifted part only stays up if held, so this keeps "held"
        # true through the grasp-force flicker that contact between the parts causes
        socket_held = socket_grasp | (socket_lifted & socket_touch)
        peg_held = peg_grasp | (peg_lifted & peg_touch)
        both_lifted = socket_held & peg_held & socket_lifted & peg_lifted
        in_hole = (g["axial"] > 0) & (g["axial"] < self.socket_hole_depth + FLOOR_TOL) & (g["fit"] <= HOLE_TOL)
        socket_up = socket_held & socket_lifted
        seated = in_hole & (g["axial"] >= self.socket_hole_depth - SEAT_MARGIN)
        # released: no part of the peg arm that can reach it (jaws, wrist, forearm) touches the peg, so the arm cannot
        # prop it in. retreat: the TCP's distance from the peg's centre line (0 while gripping it anywhere along it)
        released = ~self._arm_touching(right, self.peg)
        along = ((right.tcp_pos - self.peg.pose.p) * g["peg_axis"]).sum(-1).clamp(-self.peg_half_length, self.peg_half_length)
        retreat = torch.linalg.norm(right.tcp_pos - self.peg.pose.p - along[:, None] * g["peg_axis"], axis=1)
        if self.RETAIN:
            success = socket_up & seated & released & (retreat >= RETREAT_OK)
        else:
            success = in_hole & (g["axial"] >= INSERT_DEPTH) & (g["align"] >= ALIGN_MIN) & both_lifted
        return {
            "success": success,
            "socket_grasp": socket_grasp, "peg_grasp": peg_grasp,
            "socket_held": socket_held, "peg_held": peg_held,
            "socket_lift": socket_lift, "peg_lift": peg_lift,
            "both_lifted": both_lifted, "in_hole": in_hole, "socket_up": socket_up, "seated": seated,
            "released": released, "retreat": retreat,
            "axial": g["axial"], "lateral": g["lateral"], "fit": g["fit"], "align": g["align"], "socket_pitch": g["pitch"],
            "arms_touching": self._arms_touching(),
            "robot_touching_table": left.is_touching(self.table_scene.table) | right.is_touching(self.table_scene.table),
        }

    # ------------------------------------------------------------------ reward
    def arm_progress(self, tcp, grip_point, held, lift):
        """Stage 1 for one arm: reach its grip point 0-1 while not holding; 2 + lift 0-1 once held."""
        reach = 1 - torch.tanh(5 * torch.linalg.norm(tcp - grip_point, axis=1))
        return torch.where(held, 2 + torch.clamp(lift / self.LIFT_HEIGHT, 0, 1), reach)

    def reward_inputs(self, info: dict) -> dict:
        """The per-env quantities dense_reward reads, from the simulator."""
        left, right = self.arms
        g = self.geometry()
        # Approach along the socket axis: off-axis distance counts double, which pulls the tip onto the axis; with the
        # plain 3-D distance a tip parked 4 cm beside the socket outscored an aligned tip 5 cm straight out.
        # WHY |axial|, not the outside part only: a tip on the axis line behind the socket's closed end has axial > 0,
        # and with the outside part only it scored as if at the mouth (I023 learned it: tip 34-43 cm past the mouth,
        # near up to 0.97). Inside the hole stage 3 applies, so this term only scores tips that are not in
        approach = torch.sqrt(g["axial"] ** 2 + (2 * g["lateral"]) ** 2)
        contact = (2 * self.peg_half_width - JAW_TIP_DIST_0) / JAW_TIP_DIST_PER_RAD  # jaw angle at which the tips touch the peg
        return dict(
            retain=self.RETAIN, max_reward=self.MAX_REWARD,
            socket_progress=self.arm_progress(left.tcp_pos, g["socket_grip"], info["socket_held"], info["socket_lift"]),
            peg_progress=self.arm_progress(right.tcp_pos, g["peg_grip"], info["peg_held"], info["peg_lift"]),
            both_lifted=info["both_lifted"], in_hole=info["in_hole"], near=1 - torch.tanh(4 * approach),
            align=g["align"], axial=g["axial"], full_depth=self.socket_hole_depth - FULL_MARGIN,
            tilt=torch.clamp(g["pitch"] / TILT_FULL, 0, 1), tilt_ok=torch.clamp(g["pitch"] / TILT_OK, 0, 1),
            opened=torch.clamp((right.robot.get_qpos()[:, -1] - contact) / JAW_OPEN_PAST_CONTACT, 0, 1),
            seated=info["seated"], socket_up=info["socket_up"], released=info["released"], retreat=info["retreat"],
            success=info["success"], robot_touching_table=info["robot_touching_table"], arms_touching=info["arms_touching"],
        )

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return dense_reward(self.reward_inputs(info))

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / self.MAX_REWARD


@register_env("SO101InsertPeg-v1", max_episode_steps=120)
class InsertPeg(Insertion):
    """ALOHA's test: the peg 25 mm in while both arms hold their parts."""


@register_env("SO101InsertPegRetain-v1", max_episode_steps=150)
class InsertPegRetain(Insertion):
    """The peg fully inserted, released, and still in the held socket after the peg arm moves 4 cm away (ALOHA's Put On
    Shoe test: success only if the shoe stays on after the arms let go). 150 steps leave time to release, move away,
    and retry once."""
    RETAIN = True
    MAX_REWARD = 18.0
