"""Phi experiment 2026-10-15: PlaceCube, but the gripper is asked to hold still once the task is done.

Why: in SO101PlaceCube-v1 the reward after success is a flat 9 (place.py compute_dense_reward), so nothing tells the
policy to stop opening, and the "open over the bin" habit learned before success carries on to the joint limit.
Change (one line of reward, nothing else): after success, subtract HOLD_PENALTY * |gripper action|. Same scene,
success test, colours and observation as SO101PlaceCube-v1, so the two checkpoints compare like for like.
"""
from typing import Any

import torch
from mani_skill.utils.registration import register_env

from envs.place import PlaceCube

# WHY 0.5: well under the gap between success (9) and every non-success state, so success stays the best outcome.
HOLD_PENALTY = 0.5


@register_env("SO101PlaceCubeHold-v1", max_episode_steps=50)
class PlaceCubeHold(PlaceCube):
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        reward = super().compute_dense_reward(obs, action, info)
        grip = action[..., -1].abs().to(reward.device)
        return reward - HOLD_PENALTY * grip * info["success"].float()
