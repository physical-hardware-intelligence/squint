"""B021's checks on reward v6 on constructed states (B012's lesson: probe the shortcuts, not only the intended path).

    python experiments/bimanual/reward_probe_v6.py      (CPU, one env)
Grippers' tip positions are set directly, as reward_probe_handover.py; the picker's gripper is 11.5% open (B011's
median while both held), the receiver's 25% when open (B020's median approach) or 0% when shut, 10.5% when gripping.
Contact flags are set by hand. Prints a v2 / v5 / v6 table and exits 1 unless every check passes:
  S1 gripping at the handover point beats the best hover by at least 0.5
  S2 the grip pays less the farther the bar is from the handover point, and 25 cm off v6 pays at most 0.05 over v2
  S3 the release never pays less as it goes: grip, picker half open, fully open, a jaw still on, clear of the bar
  S4 an open gripper beats a shut one at the same spot (20 cm out and at the bar)
  S5 every reach state stays at or above 4, stage 2's top
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import envs  # noqa: E402,F401
from envs.bimanual_base import arm_yaw_pose  # noqa: E402

PICKER_OPEN, RECEIVER_OPEN, RECEIVER_GRIP = 0.115, 0.25, 0.105

env = gym.make("SO101HandoverBarV6Grip-v1", num_envs=1, obs_mode="state_dict", sim_backend="cpu")
env.reset(seed=3)
u = env.unwrapped
p_arm, r_arm = u.arms[int(u.picker[0])], u.arms[1 - int(u.picker[0])]


def set_open(arm, frac):
    lo, hi = arm.robot.get_qlimits()[0, -1, :]
    q = arm.robot.get_qpos().clone()
    q[:, -1] = lo + frac * (hi - lo)
    arm.robot.set_qpos(q)


# Gripper tips placed directly
cls = type(p_arm)
real_tcp = cls.tcp_pos
placed = {}
cls.tcp_pos = property(lambda self: placed.get(id(self), real_tcp.fget(self)))
hp = u.handover_point.reshape(-1)
T, F = torch.tensor([True]), torch.tensor([False])


def setup(bar_offset_x, receiver_gap, picker_open=PICKER_OPEN, receiver_open=RECEIVER_OPEN):
    """Bar centred `bar_offset_x` from the handover point, long axis along x; picker tip on one end, receiver tip
    `receiver_gap` beyond the other end; grippers at the given openings."""
    u.bar.set_pose(arm_yaw_pose(torch.tensor([float(hp[0]) + bar_offset_x]), torch.tensor([float(hp[1])]),
                                torch.tensor([0.0]), torch.tensor([float(hp[2])])))
    axis = u._bar_axis()[0]
    off = float(u.bar_half_lengths[0]) - 0.012
    placed[id(p_arm)] = (u.bar.pose.p[0] + axis * off)[None]
    placed[id(r_arm)] = (u.bar.pose.p[0] - axis * off - axis * receiver_gap)[None]
    set_open(p_arm, picker_open)
    set_open(r_arm, receiver_open)


def score(latched=False, **flags):
    """(v2, v5, v6) for the current setup with these contact flags."""
    out = []
    for version in (2, 5, 6):
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
        out.append(float(u.compute_dense_reward(None, None, info)[0]))
    return out


grip = dict(receiver_grasp=True, both_hold=True, receiver_touch_bar=True)
release = dict(picker_grasp=False, receiver_grasp=True, receiver_touch_bar=True)
rows = {}
setup(0.0, 0.20); rows["receiver 20 cm out, open"] = score()
setup(0.0, 0.20, receiver_open=0.0); rows["receiver 20 cm out, shut"] = score()
setup(0.0, 0.005); rows["receiver at the bar, open (hover)"] = score()
setup(0.0, 0.005, receiver_open=0.0); rows["receiver at the bar, shut"] = score()
for cm in (0, 7, 12, 25):
    setup(cm / 100, 0.005, receiver_open=RECEIVER_GRIP); rows[f"both hold, bar {cm} cm from the handover point"] = score(**grip)
setup(0.0, 0.005, picker_open=0.5, receiver_open=RECEIVER_GRIP); rows["both hold, picker 50% open"] = score(**grip)
setup(0.0, 0.005, picker_open=1.0, receiver_open=RECEIVER_GRIP); rows["both hold, picker 100% open"] = score(**grip)
setup(0.0, 0.005, picker_open=1.0, receiver_open=RECEIVER_GRIP); rows["handed over, a picker jaw still on the bar"] = score(latched=True, **release)
setup(0.0, 0.005, picker_open=1.0, receiver_open=RECEIVER_GRIP)
rows["handed over, picker clear of the bar"] = score(latched=True, **release, picker_touch_bar=False)
cls.tcp_pos = real_tcp

print(f"picker {PICKER_OPEN:.1%} open while gripping; receiver {RECEIVER_OPEN:.0%} when open, {RECEIVER_GRIP:.1%} when gripping")
print(f"{'state':48s} {'v2':>6s} {'v5':>6s} {'v6':>6s}")
for name, (r2, r5, r6) in rows.items():
    print(f"{name:48s} {r2:6.2f} {r5:6.2f} {r6:6.2f}")
v6 = {k: v[2] for k, v in rows.items()}
v2 = {k: v[0] for k, v in rows.items()}
g = [v6[f"both hold, bar {cm} cm from the handover point"] for cm in (0, 7, 12, 25)]
path = [g[0], v6["both hold, picker 50% open"], v6["both hold, picker 100% open"],
        v6["handed over, a picker jaw still on the bar"], v6["handed over, picker clear of the bar"]]
reach = [v6[k] for k in ("receiver 20 cm out, open", "receiver 20 cm out, shut", "receiver at the bar, open (hover)", "receiver at the bar, shut")]
checks = {
    "S1 grip at the point beats the best hover by >= 0.5": g[0] - v6["receiver at the bar, open (hover)"] >= 0.5,
    "S2 grip pays less farther from the point": g[0] > g[1] > g[2] > g[3],
    "S2 25 cm off: v6 at most 0.05 over v2": g[3] - v2["both hold, bar 25 cm from the handover point"] <= 0.05,
    "S3 the release never pays less as it goes": all(b >= a - 1e-6 for a, b in zip(path, path[1:])),
    "S4 open beats shut at the same spot": v6["receiver 20 cm out, open"] > v6["receiver 20 cm out, shut"]
                                          and v6["receiver at the bar, open (hover)"] > v6["receiver at the bar, shut"],
    "S5 reach states at or above 4": min(reach) >= 4 - 1e-6,
}
for k, ok in checks.items():
    print(f"{'PASS' if ok else 'FAIL'}  {k}")
env.close()
sys.exit(0 if all(checks.values()) else 1)
