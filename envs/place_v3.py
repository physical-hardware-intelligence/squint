"""Phi experiment 2026-10-15: PlaceCube v3 = v2 (envs/place_v2.py, unchanged) plus fixes for two real-arm failures of
v2_s1 (runs/v2_s1/real/flight, phi/group/squint SYSTEM.md E40). Every v2 term is kept; the new terms only act in states
v2 left unconstrained.

1. HOME, once the cube is placed. In v2 a placed cube pays a flat 9 whatever the arm does, as long as each arm joint
   moves under 0.15 rad/s (agent.is_static), so post-place motion is free: in sim v2_s1 drifts 19 deg per 100 deploy
   steps and ends 48 deg from the start pose; on the real arm it hovered, and in one run grasped the placed cube again.
   Now, while the cube is placed (in the inset area, on the floor, still, robot touching neither cube nor bin):
       reward = 9 - HOME_WEIGHT * tanh(HOME_K * |q_arm - q_start|) - HOLD_PENALTY * |a_grip| - 6 * touching_table
   It needs no "robot static", so leaving for the start pose does not drop the agent out of the placed branch.
   HOME_WEIGHT < 1 keeps every placed state above every state that touches a cube lying in the bin (<= 8, see below),
   so re-picking a placed cube is worse at any arm pose. At the start pose this equals v2's success reward.

2. APERTURE, optional (SO101PlaceCubeV3Ap-v1). Sim policies, v1 and v2 alike, close while approaching and reach the
   cube at a median 21-23 deg (tip gap ~4.2 cm for a 2.5 cm cube, ~0.85 cm spare per side); after a forced drop they
   re-grasp with the jaws at 8-11 deg. Sim tracks joint targets to 0.05-0.07 deg; the real arm lags 2.5-3.2 deg on
   lift and elbow, and its first closes stopped on the cube edge at 9-14 deg. While the cube is neither grasped nor
   over the bin and the TCP is more than APERTURE_NEAR from it, a penalty grows as the gripper closes below
   APERTURE_DEG. It is zero within APERTURE_NEAR, so the close itself costs nothing.

Upper bound used in 1: a cube lying in the bin and touched by the robot is in upstream's "above bin" branch,
4 + place (<= 2) + dropped (0 while touching) + openness (<= 1) + static (<= 1) = 8; grasped and lifted out of the
bin pays 3 + place + squeeze (<= 0.3) <= 5.3. Placed far from the start pose pays >= 9 - HOME_WEIGHT = 8.25.
"""
from typing import Any

import numpy as np
import torch
from mani_skill.utils.registration import register_env

from envs.place_v2 import HOLD_PENALTY, PlaceCubeV2

HOME_WEIGHT = 0.75     # WHY: < 1 so placed (>= 8.25) always beats touching a cube in the bin (<= 8)
HOME_K = 2.0           # 1/rad: tanh = 0.93 at the 48 deg v2_s1 ends at, 0.2 at 6 deg, slope still 2/rad near zero
APERTURE_DEG = 40.0    # tip gap 6.4 cm: ~1.9 cm spare per side for a 2.5 cm cube, ~1.4 cm at 45 deg yaw
APERTURE_NEAR = 0.015  # m TCP to cube centre: penalty 0 inside, full beyond NEAR + APERTURE_RAMP
APERTURE_RAMP = 0.02
APERTURE_WEIGHT = 0.5  # vs reaching reward <= 2 and the >= 2 jump from ungrasped to grasped


class PlaceCubeV3(PlaceCubeV2):
    use_aperture = False

    def evaluate(self):
        info = super().evaluate()
        info["placed"] = (info["is_item_above_bin"] & info["item_on_bin_floor"] & (info["item_vel"] <= 2e-2)
                          & ~info["robot_touching_item"] & ~info["robot_touching_bin"])
        info["start_pose_dist"] = torch.linalg.norm(self.agent.robot.get_qpos()[:, :5] - self.rest_qpos[:5], dim=-1)
        return info

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        reward = super().compute_dense_reward(obs, action, info)   # v2, unchanged
        a_grip = action[..., -1].to(reward.device)
        placed_r = (9 - HOME_WEIGHT * torch.tanh(HOME_K * info["start_pose_dist"]) - HOLD_PENALTY * a_grip.abs()
                    - 6 * info["robot_touching_table"].float())
        reward = torch.where(info["placed"], placed_r, reward)
        if self.use_aperture:
            q_grip = self.agent.robot.get_qpos()[:, -1]
            short = torch.clamp((np.deg2rad(APERTURE_DEG) - q_grip) / np.deg2rad(APERTURE_DEG), 0, 1)
            d = torch.linalg.norm(self.agent.tcp_pose.p - self.item.pose.p, dim=-1)
            far = torch.clamp((d - APERTURE_NEAR) / APERTURE_RAMP, 0, 1)
            approaching = (~info["is_item_grasped"] & ~info["is_item_above_bin"]).float()
            reward = reward - APERTURE_WEIGHT * approaching * far * short
        return reward


class PlaceCubeV3Ap(PlaceCubeV3):
    use_aperture = True


register_env("SO101PlaceCubeV3-v1", max_episode_steps=50)(PlaceCubeV3)
register_env("SO101PlaceCubeV3Ap-v1", max_episode_steps=50)(PlaceCubeV3Ap)
