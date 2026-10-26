"""Score the reward along the handover itself (receiver approaching, gripping, the picker letting go), reward v2 vs v3.

    python experiments/bimanual/reward_probe_handover.py      (CPU, one env; run after `source env.sh`)
The grippers' tip positions are set directly (the picker's at its end of the bar, the receiver's at a chosen distance
from the other end), and the picker's gripper is set to 11.5% open: the median measured while both arms held the bar
in B011 seed 1's handovers (10-90%: 9.8-14.7%). Contact flags are set by hand. Prints a table; no pass/fail.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import envs  # noqa: E402,F401
from envs.bimanual_base import arm_yaw_pose  # noqa: E402

PICKER_OPEN = 0.115

env = gym.make("SO101HandoverBarV3-v1", num_envs=1, obs_mode="state_dict", sim_backend="cpu")
env.reset(seed=3)
u = env.unwrapped
p_arm, r_arm = u.arms[int(u.picker[0])], u.arms[1 - int(u.picker[0])]

# The picker's gripper at the measured opening
lo, hi = p_arm.robot.get_qlimits()[0, -1, :]
q = p_arm.robot.get_qpos().clone()
q[:, -1] = lo + PICKER_OPEN * (hi - lo)
p_arm.robot.set_qpos(q)

# Gripper tips placed directly
cls = type(p_arm)
real_tcp = cls.tcp_pos
placed = {}
cls.tcp_pos = property(lambda self: placed.get(id(self), real_tcp.fget(self)))

hp = u.handover_point.reshape(-1)
T, F = torch.tensor([True]), torch.tensor([False])


def setup(bar_offset_x, receiver_gap):
    """Bar centred `bar_offset_x` from the handover point, long axis along x; picker tip on one end, receiver tip
    `receiver_gap` beyond the other end."""
    u.bar.set_pose(arm_yaw_pose(torch.tensor([float(hp[0]) + bar_offset_x]), torch.tensor([float(hp[1])]),
                                torch.tensor([0.0]), torch.tensor([float(hp[2])])))
    axis = u._bar_axis()[0]
    off = float(u.bar_half_lengths[0]) - 0.012
    end_a = u.bar.pose.p[0] + axis * off
    end_b = u.bar.pose.p[0] - axis * off
    placed[id(p_arm)] = end_a[None]
    placed[id(r_arm)] = (end_b - axis * receiver_gap)[None]


def score(version, latched=False, **flags):
    u.handed_over[:] = latched
    info = {k: v.clone() for k, v in u.evaluate().items()}
    base = dict(picker_grasp=True, receiver_grasp=False, both_hold=False, picker_touch_bar=True, receiver_touch_bar=False,
                reached_handover=True, bar_lifted=True, bar_in_bin=False, success=False, robot_touching_table=False,
                robot_touching_bin=False, arms_touching_jaws=False, arms_touching_body=False)
    base.update(flags)
    base["arms_touching"] = base["arms_touching_jaws"] or base["arms_touching_body"]
    for k, v in base.items():
        info[k] = T if v else F
    u.handed_over[:] = latched
    u.REWARD_VERSION = version
    return float(u.compute_dense_reward(None, None, info)[0])


results = []
setup(0.0, 0.20)
results.append(("receiver 20 cm from its end of the bar", score(2), score(3)))
setup(0.0, 0.005)
for name, flags in (
    ("receiver at its end, not gripping (hover)", dict()),
    ("receiver grips: both hold", dict(receiver_grasp=True, both_hold=True, receiver_touch_bar=True)),
    ("both hold, the two grippers' jaws touch", dict(receiver_grasp=True, both_hold=True, receiver_touch_bar=True, arms_touching_jaws=True)),
    ("both hold, wrists or forearms touch", dict(receiver_grasp=True, both_hold=True, receiver_touch_bar=True, arms_touching_body=True)),
):
    results.append((name, score(2, **flags), score(3, **flags)))
setup(0.07, 0.005)
results.append(("hover, bar nudged 7 cm off the handover point", score(2), score(3)))
setup(0.12, 0.005)
results.append(("hover, bar nudged 12 cm off the handover point", score(2), score(3)))
setup(0.0, 0.005)
release = dict(picker_grasp=False, receiver_grasp=True, receiver_touch_bar=True)
results.append(("handover done: picker opening, a jaw on the bar, jaws touch",
                score(2, latched=True, **release, arms_touching_jaws=True), score(3, latched=True, **release, arms_touching_jaws=True)))
results.append(("handover done: picker clear of the bar",
                score(2, latched=True, **release, picker_touch_bar=False), score(3, latched=True, **release, picker_touch_bar=False)))
cls.tcp_pos = real_tcp

print(f"picker gripper {PICKER_OPEN:.1%} open; reward (of 16)")
print(f"{'state':58s} {'v2':>6s} {'v3':>6s}")
for name, r2, r3 in results:
    print(f"{name:58s} {r2:6.2f} {r3:6.2f}")
env.close()
