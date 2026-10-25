"""B002 pre-flight: the handover's reward stages and success flag on constructed states (no learning, no grasping).

Places the bar and the arms directly, then checks: each stage's reward band, that the bands rise in order, the
mid-air-handover latch, the bar-in-bin test (lying, standing, outside, and on a rotated bin), and the maximum.
Run from the repository root after `source env.sh`: .venv/bin/python experiments/bimanual/check_reward.py [env_id]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import envs  # noqa: E402,F401
from envs.bimanual_base import arm_yaw_pose  # noqa: E402

ENV_ID = sys.argv[1] if len(sys.argv) > 1 else "SO101HandoverBar-v1"
env = gym.make(ENV_ID, num_envs=1, obs_mode="state_dict", sim_backend="cpu")
print(ENV_ID)
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
check("stage 5 (receiver carries) within 8-10", 8 <= r5 <= 10, f"{r5:.2f}")
check("stage 6 (in the bin) within 10-15", 10 <= r6 <= 15, f"{r6:.2f}")
check("success = 16", abs(rs - 16) < 1e-6, f"{rs:.2f}")
check("bands rise in order", r1 < r2 < r4 < r5 < r6 < rs)
check("a clean release pays at least a full point over both holding", r5 - 7 >= 1 - 1e-6, f"stage 5 {r5:.2f} vs stage 4 top 7")
info = info_with(success=True, bar_in_bin=True, robot_touching_table=False, arms_touching=False, robot_touching_bin=False)
check("normalized maximum = 1", abs(float(u.compute_normalized_dense_reward(None, None, info)[0]) - 1) < 1e-6)
rp = reward(info_with(success=True, bar_in_bin=True, robot_touching_table=True, arms_touching=True, robot_touching_bin=False))
check("penalties apply: table -6, arms -2", abs(rp - 8) < 1e-6, f"{rp:.2f}")

# The carry's pull: reward gained for 10 cm closer to the goal, bar carried at the goal's height (informational)
u.handed_over[:] = True
gp = u.bin.pose.p[0].numpy()
goal_z = float(u.bin_thickness) + hw
carry = dict(picker_grasp=False, both_hold=False, receiver_grasp=True, picker_touch_bar=False, bar_in_bin=False,
             robot_touching_table=False, arms_touching=False)
pull = {}
for d in (0.30, 0.20, 0.10):
    place_bar(gp[0], gp[1] + d, goal_z)
    pull[d] = reward(info_with(**carry))
print(f"INFO  carry reward at 30 / 20 / 10 cm from the goal: {pull[0.30]:.2f} / {pull[0.20]:.2f} / {pull[0.10]:.2f}"
      f"  (gain 30->20 cm: {pull[0.20] - pull[0.30]:.2f}, 20->10 cm: {pull[0.10] - pull[0.20]:.2f})")
check("carry reward rises as the bar nears the goal", pull[0.30] < pull[0.20] < pull[0.10])

env.close()
print(f"{sum(results)} of {len(results)} checks passed")
sys.exit(0 if all(results) else 1)
