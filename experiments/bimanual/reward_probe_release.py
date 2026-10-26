"""Score the reward along the picker's release (after the handover: the receiver grips, the picker has stopped squeezing
and moves off the bar), reward v2 vs v4. Fails if v4 ever drops along the path, or if v2 changed.

    python experiments/bimanual/reward_probe_release.py      (CPU, one env; run after `source env.sh`)
The grippers' grip points are set directly: the receiver's at its end of the bar, the picker's at a chosen distance out
from its end along the bar's axis. Contact flags are set by hand (the picker's jaw touches up to 2 cm out).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import envs  # noqa: E402,F401
from envs.bimanual_base import arm_yaw_pose  # noqa: E402

env = gym.make("SO101HandoverBarV4-v1", num_envs=1, obs_mode="state_dict", sim_backend="cpu")
env.reset(seed=3)
u = env.unwrapped
p_arm, r_arm = u.arms[int(u.picker[0])], u.arms[1 - int(u.picker[0])]
cls = type(p_arm)
real_tcp = cls.tcp_pos
placed = {}
cls.tcp_pos = property(lambda self: placed.get(id(self), real_tcp.fget(self)))
hp = u.handover_point.reshape(-1)
T, F = torch.tensor([True]), torch.tensor([False])
u.bar.set_pose(arm_yaw_pose(torch.tensor([float(hp[0])]), torch.tensor([float(hp[1])]), torch.tensor([0.0]), torch.tensor([float(hp[2])])))
axis = u._bar_axis()[0]
off = float(u.bar_half_lengths[0]) - 0.012
end_a, end_b = u.bar.pose.p[0] + axis * off, u.bar.pose.p[0] - axis * off
placed[id(r_arm)] = end_b[None]


def score(version, gap, touching, both=False):
    placed[id(p_arm)] = (end_a + axis * gap)[None]
    u.handed_over[:] = True
    info = {k: v.clone() for k, v in u.evaluate().items()}
    flags = dict(picker_grasp=both, receiver_grasp=True, both_hold=both, picker_touch_bar=touching or both,
                 receiver_touch_bar=True, reached_handover=True, bar_lifted=True, bar_in_bin=False, success=False,
                 robot_touching_table=False, robot_touching_bin=False, arms_touching_jaws=False, arms_touching_body=False,
                 arms_touching=False)
    for k, v in flags.items():
        info[k] = T if v else F
    u.handed_over[:] = True
    u.REWARD_VERSION = version
    return float(u.compute_dense_reward(None, None, info)[0])


rows = [("both hold (picker still squeezing)", 0.0, True, True)]
rows += [(f"picker {g * 100:.0f} cm out, jaw {'touching' if t else 'off'}", g, t, False)
         for g, t in ((0.0, True), (0.01, True), (0.02, True), (0.02, False), (0.04, False), (0.06, False), (0.10, False))]
res = [(name, score(2, g, t, b), score(4, g, t, b)) for name, g, t, b in rows]
cls.tcp_pos = real_tcp
print(f"{'state':42s} {'v2':>6s} {'v4':>6s}")
for name, r2, r4 in res:
    print(f"{name:42s} {r2:6.2f} {r4:6.2f}")
v4 = [r4 for _, _, r4 in res]
ok_mono = all(b >= a - 1e-6 for a, b in zip(v4, v4[1:]))
v2_touch, v2_off = res[1][1], res[-1][1]
ok_v2 = abs(v2_touch - 7.71) < 0.02 and abs(v2_off - 8.71) < 0.02  # reward-design §7: 7.71 touching, 8.71 clear
print(f"{'PASS' if ok_mono else 'FAIL'}  v4 never drops along the release")
print(f"{'PASS' if ok_v2 else 'FAIL'}  v2 unchanged (7.71 touching, 8.71 clear; got {v2_touch:.2f}, {v2_off:.2f})")
print(f"{'PASS' if v4[-1] - v4[1] > 0.9 else 'FAIL'}  v4 pays the whole way off ({v4[1]:.2f} -> {v4[-1]:.2f})")
env.close()
sys.exit(0 if ok_mono and ok_v2 else 1)
