"""Score task 1's reward on hand-set states along the intended path, to find dips (states the intended behaviour must
pass through that score lower than the state before), and measure where the bin sits relative to the receiver.

    python experiments/bimanual/reward_probe.py [env_id]      (CPU, one env; run after `source env.sh`)
Flags are set by hand where the physics (grasp contacts) cannot be posed: this scores the reward function, not a
policy. Prints a table; no pass/fail.
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
env.reset(seed=3)
u = env.unwrapped
T, F = torch.tensor([True]), torch.tensor([False])
hw = float(u.bar_half_widths[0])


def place_bar(x, y, z):
    u.bar.set_pose(arm_yaw_pose(torch.tensor([x]), torch.tensor([y]), torch.tensor([0.0]), torch.tensor([z])))


def score(**flags):
    info = {k: v.clone() for k, v in u.evaluate().items()}
    for k, v in flags.items():
        info[k] = T if v else F
    return float(u.compute_dense_reward(None, None, info)[0])


clean = dict(robot_touching_table=False, arms_touching=False, robot_touching_bin=False)
hp = u.handover_point.numpy().reshape(-1)
bp = u.bin.pose.p[0].numpy()
rows = []

u.handed_over[:] = False
place_bar(*hp)
rows.append(("both hold the lifted bar, picker jaws closed", score(picker_grasp=True, receiver_grasp=True, both_hold=True, picker_touch_bar=True, **clean)))
u.handed_over[:] = True
rows.append(("after the latch: picker opening, a jaw still touching", score(picker_grasp=False, receiver_grasp=True, both_hold=False, picker_touch_bar=True, **clean)))
rows.append(("after the latch: picker clear, receiver carries", score(picker_grasp=False, receiver_grasp=True, both_hold=False, picker_touch_bar=False, bar_in_bin=False, **clean)))
place_bar(bp[0], bp[1], 0.10)
rows.append(("carrying, bar 10 cm above the bin centre", score(picker_grasp=False, receiver_grasp=True, picker_touch_bar=False, bar_in_bin=False, **clean)))
rows.append(("released over the bin, bar falling (not yet in)", score(picker_grasp=False, receiver_grasp=False, picker_touch_bar=False, receiver_touch_bar=False, bar_in_bin=False, **clean)))
place_bar(bp[0], bp[1], float(u.bin_thickness) + hw)
rows.append(("bar lying in the bin, receiver still holding", score(receiver_grasp=True, receiver_touch_bar=True, bar_in_bin=True, success=False, **clean)))
rows.append(("bar in the bin, released, gripper touching the bin wall", score(receiver_grasp=False, receiver_touch_bar=False, bar_in_bin=True, success=False, robot_touching_table=False, arms_touching=False, robot_touching_bin=True)))
rows.append(("bar in the bin, released, clear (success)", score(success=True, bar_in_bin=True, **clean)))
# The rim: carried at a safe height (bottom 1.5 cm above the walls) just outside the bin, then just inside
wall_top = float(u.bin_thickness) + 2 * float(u.bin_half_sizes_z[0])
inner_x = float(u.bin_half_sizes_x[0]) - float(u.bin_thickness)
byaw = float(2 * np.arctan2(u.bin.pose.q[0, 3], u.bin.pose.q[0, 0]))
for label, r in (("just outside the rim", inner_x + 0.03), ("just inside the rim", inner_x - 0.015)):
    place_bar(bp[0] + r * np.cos(byaw), bp[1] + r * np.sin(byaw), wall_top + 0.015 + hw)
    rows.append((f"carrying, bar above wall height, {label}", score(picker_grasp=False, receiver_grasp=True, picker_touch_bar=False, bar_in_bin=False, **clean)))
place_bar(bp[0], bp[1], wall_top + 0.015 + hw)
rows.append(("carrying, above wall height over the bin centre", score(picker_grasp=False, receiver_grasp=True, picker_touch_bar=False, bar_in_bin=False, **clean)))
place_bar(bp[0], bp[1], float(u.bin_thickness) + hw + 0.02)
rows.append(("carrying, lowered to 2 cm above the floor (not yet counted in)", score(picker_grasp=False, receiver_grasp=True, picker_touch_bar=False, bar_in_bin=False, success=False, **clean)))
place_bar(bp[0] + 0.2, bp[1], hw)
rows.append(("after the latch: bar dropped on the table", score(picker_grasp=False, receiver_grasp=False, picker_touch_bar=False, receiver_touch_bar=False, bar_in_bin=False, **clean)))

print(f"{ENV_ID}: reward (of 16) on hand-set states")
for name, r in rows:
    print(f"  {r:6.2f}  {name}")

# Where is the bin relative to the receiver? (300 layouts, the receiver's pan axis and heading as origin)
rel = []
for seed in range(300):
    env.reset(seed=seed)
    rcv = 1 - int(u.picker[0])  # 0 = left, 1 = right
    base = u.arms[rcv].robot.pose
    mat = base.to_transformation_matrix()[0]
    local = (mat[:3, :3].T @ (u.bin.pose.p[0] - mat[:3, 3])).numpy()
    rel.append((local[0], local[1] * (1 if rcv == 0 else -1)))  # mirror the right arm so "outward" is +y for both
rel = np.array(rel)
print(f"bin centre in the receiver's frame (outward = +y), 300 layouts: x {rel[:, 0].min()*100:.1f}-{rel[:, 0].max()*100:.1f} cm,"
      f" y {rel[:, 1].min()*100:.1f}-{rel[:, 1].max()*100:.1f} cm; mean ({rel[:, 0].mean()*100:.1f}, {rel[:, 1].mean()*100:.1f})")
# A bar stood upright at the mean position: inside the bin's inner walls in how many layouts? (bin yaw varies)
inside = 0
for seed in range(300):
    env.reset(seed=seed)
    rcv = 1 - int(u.picker[0])
    mat = u.arms[rcv].robot.pose.to_transformation_matrix()[0]
    m = rel.mean(0)
    p_local = torch.tensor([m[0], m[1] * (1 if rcv == 0 else -1), 0.0], dtype=torch.float32)
    p_world = mat[:3, :3] @ p_local + mat[:3, 3]
    bmat = u.bin.pose.to_transformation_matrix()[0]
    q = (bmat[:3, :3].T @ (p_world - bmat[:3, 3])).numpy()
    margin_x = float(u.bin_half_sizes_x[0] - u.bin_thickness - hw)
    margin_y = float(u.bin_half_sizes_y[0] - u.bin_thickness - hw)
    inside += (abs(q[0]) < margin_x) and (abs(q[1]) < margin_y)
print(f"an upright bar placed blind at that mean position lands inside the bin in {inside} of 300 layouts")
env.close()
