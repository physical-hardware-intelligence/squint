"""B002 pre-flight: the handover's reward stages and success flag on constructed states (no learning, no grasping).

Places the bar and the arms directly, then checks: each stage's reward band, that the bands rise in order, the
mid-air-handover latch, the bar-in-bin test (lying, standing, outside, and on a rotated bin), and the maximum.
Run from the repository root after `source env.sh`: .venv/bin/python experiments/bimanual/check_reward.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import envs  # noqa: E402,F401
from envs.bimanual_base import arm_yaw_pose  # noqa: E402

env = gym.make("SO101HandoverBar-v1", num_envs=1, obs_mode="state_dict", sim_backend="cpu")
env.reset(seed=3)
u = env.unwrapped
T, F = torch.tensor([True]), torch.tensor([False])
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def place_bar(x, y, z, yaw=0.0, pitch_up=False):
    pose = arm_yaw_pose(torch.tensor([x]), torch.tensor([y]), torch.tensor([yaw]), torch.tensor([z]))
    if pitch_up:  # long axis vertical: rotate 90 degrees about the bar's y axis
        from mani_skill.utils.structs.pose import Pose
        q = torch.tensor([[np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0]])
        pose = pose * Pose.create_from_pq(q=q)
    u.bar.set_pose(pose)


def info_with(**flags):
    info = {k: v.clone() for k, v in u.evaluate().items()}
    for k, v in flags.items():
        info[k] = T if v else F
    return info


def reward(info):
    return float(u.compute_dense_reward(None, None, info)[0])


# The bar in the bin: lying along the bin's long side, standing, outside; then on a bin rotated 37 degrees
bp, hw = u.bin.pose.p[0].numpy(), float(u.bar_half_widths[0])
bin_yaw = float(2 * np.arctan2(u.bin.pose.q[0, 3], u.bin.pose.q[0, 0]))
place_bar(bp[0], bp[1], float(u.bin_thickness) + hw, yaw=bin_yaw + np.pi / 2)
check("bar lying in the bin counts as in", bool(u.evaluate()["bar_in_bin"][0]))
place_bar(bp[0], bp[1], float(u.bin_thickness) + float(u.bar_half_lengths[0]), pitch_up=True)
check("bar standing in the bin counts as in", bool(u.evaluate()["bar_in_bin"][0]))
place_bar(bp[0] + 0.15, bp[1], hw)
check("bar beside the bin counts as out", not bool(u.evaluate()["bar_in_bin"][0]))
u.bin.set_pose(arm_yaw_pose(torch.tensor([bp[0]]), torch.tensor([bp[1]]), torch.tensor([bin_yaw + np.deg2rad(37)])))
place_bar(bp[0], bp[1], float(u.bin_thickness) + hw, yaw=bin_yaw + np.deg2rad(37) + np.pi / 2)
check("bar along a rotated bin's long side counts as in (PlaceCube's world-axis bug avoided)", bool(u.evaluate()["bar_in_bin"][0]))

# The latch: only both grippers on the lifted bar count as a handover, and it persists until reset
u.handed_over[:] = False
place_bar(*u.handover_point.numpy().tolist())
info = info_with(picker_grasp=True, receiver_grasp=True)
both = info["picker_grasp"] & info["receiver_grasp"] & info["bar_lifted"]
check("handover needs the bar lifted", bool(info["bar_lifted"][0]), f"bar z {float(u.bar.pose.p[0, 2]) * 100:.1f} cm")
u.handed_over |= both
check("latch set by a mid-air double grasp", bool(u.handed_over[0]))
place_bar(bp[0] + 0.15, bp[1], hw)
check("latch persists after the bar is put down", bool(u.evaluate()["handed_over"][0]))

# Stage bands, in order (the arms' positions are wherever reset left them; only the flags change)
u.handed_over[:] = False
place_bar(*u.handover_point.numpy().tolist())
r1 = reward(info_with(picker_grasp=False, both_hold=False, robot_touching_table=False, arms_touching=False))
r2 = reward(info_with(picker_grasp=True, both_hold=False, robot_touching_table=False, arms_touching=False))
r4 = reward(info_with(picker_grasp=True, both_hold=True, robot_touching_table=False, arms_touching=False))
u.handed_over[:] = True
r5 = reward(info_with(picker_grasp=False, both_hold=False, receiver_grasp=True, picker_touch_bar=False,
                      bar_in_bin=False, robot_touching_table=False, arms_touching=False))
r6 = reward(info_with(picker_grasp=False, both_hold=False, receiver_grasp=False, receiver_touch_bar=False,
                      bar_in_bin=True, success=False, robot_touching_table=False, arms_touching=False, robot_touching_bin=False))
rs = reward(info_with(success=True, bar_in_bin=True, robot_touching_table=False, arms_touching=False, robot_touching_bin=False))
check("stage 1 (reach) within 0-2", 0 <= r1 <= 2, f"{r1:.2f}")
check("stage 2 (picker holds, bar at the handover point) near 4", 3.5 <= r2 <= 6.0, f"{r2:.2f}")
check("stage 4 (both hold) within 6-7", 6 <= r4 <= 7, f"{r4:.2f}")
check("stage 5 (receiver carries) within 7-9", 7 <= r5 <= 9, f"{r5:.2f}")
check("stage 6 (in the bin) within 9-14", 9 <= r6 <= 14, f"{r6:.2f}")
check("success = 15", abs(rs - 15) < 1e-6, f"{rs:.2f}")
check("bands rise in order", r1 < r2 < r4 < r5 < r6 < rs)
info = info_with(success=True, bar_in_bin=True, robot_touching_table=False, arms_touching=False, robot_touching_bin=False)
check("normalized maximum = 1", abs(float(u.compute_normalized_dense_reward(None, None, info)[0]) - 1) < 1e-6)
rp = reward(info_with(success=True, bar_in_bin=True, robot_touching_table=True, arms_touching=True, robot_touching_bin=False))
check("penalties apply: table -6, arms -2", abs(rp - 7) < 1e-6, f"{rp:.2f}")

env.close()
print(f"{sum(results)} of {len(results)} checks passed")
sys.exit(0 if all(results) else 1)
