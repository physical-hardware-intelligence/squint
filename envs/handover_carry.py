"""B008: the handover's carry on its own. Each episode starts where a trained policy's receiver had just taken the bar
(the picker's jaws just off it), restored from a bank of saved simulator states, so the carry and the placing can be
learned without the handover in the way. Reward, success and layout are task 1's (`handover.py`).

The bank comes from `experiments/bimanual/harvest_carry_starts.py`. A saved state only fits the env index it came from:
object sizes are drawn once per env index when the scene is built, and are checked against the bank at the first reset.
An env index with no saved state starts a normal episode (counted in `carry_start_misses`).
"""

import os

import torch

from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from .handover import Handover, HandoverBarBinInState, HandoverBarFirstFrame

def controllers_of(arm):
    """(name, controller) pairs: a combined controller's parts, or the arm's single controller (the SO-101 drives its 6
    joints, gripper included, with one `pd_joint_target_delta_pos` controller)."""
    ctrl = arm.controller
    return ctrl.controllers.items() if hasattr(ctrl, "controllers") else [("all", ctrl)]


CARRY_EPISODE_STEPS = 60  # 6 s: B003's carries started about 30 cm from the bin


class CarryStartMixin:
    REWARD_VERSION = 2  # B008 trains the carry with reward v2; one condition keeps v1 for comparison
    CARRY_BANK = os.environ.get("SQUINT_CARRY_BANK", "runs/carry_starts.pt")
    GEOMETRY_KEYS = ("bar_half_widths", "bar_half_lengths", "bin_half_sizes_x", "bin_half_sizes_y", "bin_half_sizes_z")

    def _load_carry_bank(self):
        bank = torch.load(self.CARRY_BANK, map_location=self.device, weights_only=False)
        n = self.num_envs
        if len(bank["geometry"]["bar_half_widths"]) < n:
            raise ValueError(f"carry bank {self.CARRY_BANK} covers {len(bank['geometry']['bar_half_widths'])} env indices, not {n}")
        self._bank = bank
        self._bank_rows = [torch.nonzero(bank["env_index"] == i)[:, 0] for i in range(n)]
        self.carry_start_misses = 0
        self._restored = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._restored_rows = torch.zeros(n, dtype=torch.long, device=self.device)

    def _check_geometry(self):
        """Object sizes are drawn per env index when the scene is built; a saved state only fits the sizes it was saved
        with. Checked at the first episode and again after every rebuild (eval envs rebuild by default)."""
        n = self.num_envs
        for k in self.GEOMETRY_KEYS:
            mine, theirs = getattr(self, k)[:n], self._bank["geometry"][k][:n].to(self.device)
            if not torch.allclose(mine.float(), theirs.float()):
                raise ValueError(f"carry bank {self.CARRY_BANK}: {k} differs from this env's (scene rebuilt with other "
                                 "sizes?); the saved states do not fit. Use reconfiguration_freq 0 for carry envs.")
        self._geometry_ok = True

    def _after_reconfigure(self, options: dict):
        super()._after_reconfigure(options)
        self._geometry_ok = False

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        if not hasattr(self, "_bank"):
            self._load_carry_bank()
        if not getattr(self, "_geometry_ok", False):
            self._check_geometry()
        self._restored[env_idx] = False
        rows, envs = [], []
        for e in env_idx.tolist():
            candidates = self._bank_rows[e]
            if len(candidates) == 0:
                self.carry_start_misses += 1
                continue
            rows.append(int(candidates[int(torch.randint(len(candidates), (1,)))]))
            envs.append(e)
        if not envs:
            return
        # set_sim_state writes rows in ascending env order (the reset mask's order): sort envs and rows together
        order = sorted(range(len(envs)), key=lambda i: envs[i])
        rows = torch.tensor([rows[i] for i in order], device=self.device)
        envs = torch.tensor([envs[i] for i in order], device=self.device)
        b = self._bank
        state = {group: {name: v[rows] for name, v in b[group].items()} for group in ("actors", "articulations")}
        self.scene.set_sim_state(state, envs)
        self.picker[envs] = b["picker"][rows]
        self.handed_over[envs] = True
        # Per-episode draws made before the restore: give back the saved episode's background and gripper drive, so
        # the grip squeezes as it did and the kept first frame matches the current frames' background
        if self._background_colors is not None and "background" in b:
            self._background_colors[envs] = b["background"][rows].to(self._background_colors.device)
        if "gripper_stiffness" in b:
            for a, arm in enumerate(self.arms):
                joint = arm.robot.joints_map["gripper"]
                for e, r in zip(envs.tolist(), rows.tolist()):
                    stiff, damp = float(b["gripper_stiffness"][r, a]), float(b["gripper_damping"][r, a])
                    joint._objs[e].set_drive_properties(stiff, damp, force_limit=100)
                    self._gripper_stiffness[a, e] = stiff
                    self._gripper_damping[a, e] = damp
        goal = self.bin.pose.p[env_idx].clone()  # the goal marker follows the (restored) bin for every env being reset
        goal[:, 2] = self.bin_thickness + self.bar_half_widths[env_idx]
        self.goal_site.set_pose(Pose.create_from_pq(goal))
        self._restored[envs] = True
        self._restored_rows[envs] = rows

    def _after_reset_before_observe(self):
        """Controllers were reset to the restored joint angles; give them back their saved targets (the gripper's
        target sits inside the bar: that is what makes it squeeze). Restored envs also get their first frame back."""
        super()._after_reset_before_observe()
        envs = torch.nonzero(self._restored)[:, 0]
        if len(envs) == 0:
            return
        rows = self._restored_rows[envs]
        for a, arm in enumerate(self.arms):
            for uid, ctrl in controllers_of(arm):
                if hasattr(ctrl, "_target_qpos"):
                    ctrl._target_qpos[envs] = self._bank["targets"][a][uid][rows]
        if hasattr(self, "_first_frames"):
            for cam, frames in self._first_frames.items():
                for k in frames:
                    frames[k][envs] = self._bank["first"][cam][k][rows].to(frames[k].device)
            self._first_pending[envs] = False
        self._restored[envs] = False


@register_env("SO101HandoverCarry-v1", max_episode_steps=CARRY_EPISODE_STEPS)
class HandoverCarry(CarryStartMixin, Handover):
    """B008 condition: wrist cameras only (as task 1)."""


@register_env("SO101HandoverCarryFirstFrame-v1", max_episode_steps=CARRY_EPISODE_STEPS)
class HandoverCarryFirstFrame(CarryStartMixin, HandoverBarFirstFrame):
    """B008 condition: plus both wrist images from the start of the original episode (bin in the receiver's view)."""


@register_env("SO101HandoverCarryTopCam-v1", max_episode_steps=CARRY_EPISODE_STEPS)
class HandoverCarryTopCam(CarryStartMixin, Handover):
    """B008 condition: plus a fixed top camera that sees both arms, the bar and the bin."""

    TOP_CAMERA = True


@register_env("SO101HandoverCarryBinInState-v1", max_episode_steps=CARRY_EPISODE_STEPS)
class HandoverCarryBinInState(CarryStartMixin, HandoverBarBinInState):
    """B008 condition (ceiling, simulation only): plus the bin's position and heading as 4 state numbers."""


@register_env("SO101HandoverCarryBinInStateRewardV1-v1", max_episode_steps=CARRY_EPISODE_STEPS)
class HandoverCarryBinInStateRewardV1(CarryStartMixin, HandoverBarBinInState):
    """B008 comparison: the bin-state ceiling with reward v1, to measure what v2's fixes change."""

    REWARD_VERSION = 1
