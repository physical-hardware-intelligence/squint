"""Checks for task 2 (`envs/insertion.py`). Prints one line per check and exits 1 if any fails.

    python experiments/bimanual/insert_probe.py       (CPU simulator, one env; run after `source env.sh`)
1. Reward, on constructed states fed to the env's own `dense_reward`: along the intended path every state scores at
   least the one before; wrong moves (opening too shallow, mouth down or level, the peg falling out) gain nothing.
2. Geometry, on posed parts read through the env's own `evaluate`: the leading face counts as in the hole only when it
   fits the hole's cross-section (offset, roll), the tip does not flip past half the peg, the hole has a floor.
3. Physics: the socket held rigidly at a pitch, the peg released on the axis at a depth, 2 s of simulation, then the
   env's own `seated` flag. This is where the retain reward's depth and tilt thresholds come from.
4. The jaw constants: fingertip distance against jaw angle, measured on the right arm (0-0.4 rad, the range the
   reward reads).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mani_skill.utils.structs.pose import Pose  # noqa: E402
from transforms3d.quaternions import axangle2quat, mat2quat, qmult  # noqa: E402

import envs  # noqa: E402,F401
from envs.insertion import (FULL_MARGIN, JAW_TIP_DIST_0, JAW_TIP_DIST_PER_RAD, TILT_FULL, TILT_OK,  # noqa: E402
                            dense_reward)
from envs.insertion_parts import SOCKET_HOLE_DEPTH  # noqa: E402

checks = []


def check(name, ok, detail=""):
    checks.append(ok)
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


# ---------------------------------------------------------------- 1. reward
T1 = lambda x: torch.tensor([float(x)])  # noqa: E731
B1 = lambda x: torch.tensor([bool(x)])  # noqa: E731
FULL = SOCKET_HOLE_DEPTH - FULL_MARGIN


def r(retain=True, prog=(3, 3), up=True, near=0.0, align=1.0, pitch=0.0, depth_mm=None, opened=0.0, released=False,
      retreat_cm=0.0, socket_up=True, success=False, table=False, touching=False):
    """Reward of one constructed state. depth_mm None = the tip outside the hole."""
    a = (depth_mm or 0) / 1000
    in_hole = depth_mm is not None and 0 < a < SOCKET_HOLE_DEPTH + 0.005
    s = dict(retain=retain, max_reward=18.0 if retain else 16.0, socket_progress=T1(prog[0]), peg_progress=T1(prog[1]),
             both_lifted=B1(up), in_hole=B1(in_hole), near=T1(near), align=T1(align), axial=T1(a), full_depth=T1(FULL),
             tilt=T1(np.clip(pitch / TILT_FULL, 0, 1)), tilt_ok=T1(np.clip(pitch / TILT_OK, 0, 1)), opened=T1(opened),
             seated=B1(in_hole and a >= SOCKET_HOLE_DEPTH - 0.005), socket_up=B1(socket_up), released=B1(released),
             retreat=T1(retreat_cm / 100), success=B1(success), robot_touching_table=B1(table), arms_touching=B1(touching))
    return float(dense_reward(s))


reach = lambda cm: 1 - np.tanh(5 * cm / 100)  # noqa: E731  stage 1 reach for the peg arm once it lets go
path = [
    ("both reaching", r(prog=(0.6, 0.6), up=False)),
    ("both held, lifted", r(prog=(3, 3), up=False)),
    ("up, tip 15 cm out, crooked", r(near=1 - np.tanh(0.6), align=0.5)),
    ("up, tip 15 cm out, crooked, mouth up 20", r(near=1 - np.tanh(0.6), align=0.5, pitch=20)),
    ("up, tip 3 cm out, straight, mouth up 20", r(near=1 - np.tanh(0.12), pitch=20)),
    ("at the mouth, straight, mouth up 20", r(near=1.0, pitch=20)),
    ("in 5 mm", r(near=1.0, pitch=20, depth_mm=5)),
    ("in 25 mm", r(near=1.0, pitch=20, depth_mm=25)),
    ("in 40 mm", r(near=1.0, pitch=20, depth_mm=40)),
    ("in 46 mm, seated", r(near=1.0, pitch=20, depth_mm=46)),
    ("in 48 mm (full)", r(near=1.0, pitch=20, depth_mm=48)),
    ("48 mm, jaw half open", r(near=1.0, pitch=20, depth_mm=48, opened=0.5)),
    ("48 mm, jaw open", r(near=1.0, pitch=20, depth_mm=48, opened=1.0)),
    ("let go, arm at the grip point", r(prog=(3, reach(0)), up=False, pitch=20, depth_mm=48, opened=1, released=True)),
    ("let go, 2 cm away", r(prog=(3, reach(2)), up=False, pitch=20, depth_mm=48, opened=1, released=True, retreat_cm=2)),
    ("let go, 4 cm away: success", r(prog=(3, reach(4)), up=False, pitch=20, depth_mm=48, released=True, retreat_cm=4,
                                     success=True)),
]
for name, v in path:
    print(f"  {v:6.2f}  {name}")
P = [v for _, v in path]
held = lambda d, **k: r(near=1.0, pitch=20, depth_mm=d, **k)  # noqa: E731
check("retain: the intended path never decreases", all(a <= b + 1e-6 for a, b in zip(P, P[1:])))
check("retain: stage 2 at the mouth (straight, tilted) = 10 = stage 3 at 0 mm", abs(P[5] - 10) < 1e-3 and abs(held(0.01) - 10) < 0.01)
check("retain: held 25 mm = 14, held at full depth = 16", abs(P[7] - 14) < 1e-3 and abs(P[10] - 16) < 1e-3)
check("retain: depth pays all the way, 25 -> 48 mm strictly rising",
      all(held(a) < held(b) for a, b in zip(range(25, 48), range(26, 49))))
check("retain: success (18) beats every other state", P[-1] == 18 and max(P[:-1]) < 18)
check("retain: a seated peg the arm still holds (16) is 2 below success", abs((P[-1] - P[10]) - 2) < 1e-3)
check("retain: opening at 30 mm adds nothing", abs(held(30, opened=1) - held(30)) < 1e-6)
check("retain: opening seated, mouth level, adds nothing",
      abs(r(near=1, pitch=0, depth_mm=48, opened=1) - r(near=1, pitch=0, depth_mm=48)) < 1e-6)
check("retain: opening seated, mouth 10 deg down, adds nothing",
      abs(r(near=1, pitch=-10, depth_mm=48, opened=1) - r(near=1, pitch=-10, depth_mm=48)) < 1e-6)
check("retain: opening seated with the socket dropped adds nothing",
      abs(held(48, opened=1, socket_up=False) - held(48, socket_up=False)) < 1e-6)
check("retain: letting go seated with the mouth level or 10 deg down pays no more than holding",
      all(r(prog=(3, reach(2)), up=False, pitch=pt, depth_mm=48, opened=1, released=True, retreat_cm=2)
          <= r(near=1, pitch=pt, depth_mm=48) + 1e-6 for pt in (0, -10)))
fallen = r(prog=(3, reach(5)), up=False, pitch=20, depth_mm=None, released=True, retreat_cm=5)
check("retain: the peg falling out after the release scores below anything held and lifted", fallen < P[2], f"{fallen:.2f}")
check("retain: released at 30 mm and still in scores as stage 1 (no release pay)",
      r(prog=(3, reach(5)), up=False, pitch=20, depth_mm=30, released=True, retreat_cm=5) <= 4 + 1e-6)
check("retain: mouth tilt is worth at most 1 in stage 2", abs((P[3] - P[2]) - 1) < 1e-3)
v1 = [r(retain=False, near=1.0), r(retain=False, near=1.0, depth_mm=12), r(retain=False, near=1.0, depth_mm=24.9),
      r(retain=False, near=1.0, depth_mm=25, success=True)]
check("v1: mouth 10, 25 mm success 16, rising in between", v1 == sorted(v1) and abs(v1[0] - 10) < 1e-3 and v1[-1] == 16)
check("both: table -2, arms touching -1", abs(held(20) - held(20, table=True) - 2) < 1e-6
      and abs(held(20) - held(20, touching=True) - 1) < 1e-6)

# ---------------------------------------------------------------- 2. geometry
MOUTH = np.array([0.19, 0.03, 0.11])  # a point in mid-air between the arms' start poses, away from both arms


def rot(x):
    """Rotation matrix whose local x is x and whose local z is as close to world up as possible."""
    z = np.array([0.0, 0.0, 1.0]) - x * x[2]
    z /= np.linalg.norm(z)
    return np.stack([x, np.cross(z, x), z], 1)


def pose(p, q):
    return Pose.create_from_pq(p=torch.tensor(np.array([p]), dtype=torch.float32), q=torch.tensor(np.array([q]), dtype=torch.float32))


def place(e, pitch_deg=0.0, depth=0.01, off=(0.0, 0.0), roll_deg=0.0, flip=False):
    """Socket mouth at MOUTH, tilted mouth-up by pitch_deg; the peg's leading face `depth` inside the mouth plane,
    offset `off` along the socket's local (y, z), rolled about its own axis, and with its -x end leading if flip.
    Returns the socket pose."""
    p = np.deg2rad(pitch_deg)
    inward = np.array([0.0, np.cos(p), -np.sin(p)])  # into the hole; the socket's local x points the other way
    s_rot = rot(-inward)
    tip = MOUTH + inward * depth + s_rot[:, 1] * off[0] + s_rot[:, 2] * off[1]
    q = qmult(mat2quat(rot(inward)), axangle2quat([1, 0, 0], np.deg2rad(roll_deg)))
    if flip:
        q = qmult(q, axangle2quat([0, 0, 1], np.pi))
    S = pose(MOUTH + inward * float(e.socket_half_length[0]), mat2quat(s_rot))
    e.socket.set_pose(S)
    e.peg.set_pose(pose(tip - inward * float(e.peg_half_length[0]), q))
    return S


env = gym.make("SO101InsertPegRetain-v1", num_envs=1, obs_mode="state", sim_backend="cpu", domain_randomization=False)
env.reset(seed=0)
e = env.unwrapped


def ev(**k):
    place(e, **k)
    i = e.evaluate()
    return {n: float(i[n][0]) for n in ("in_hole", "seated", "axial", "fit", "align", "socket_pitch")}


g = ev(depth=0.010)
check("geometry: on the axis 10 mm in: in the hole, axial 10 mm, fit -5 mm",
      g["in_hole"] == 1 and abs(g["axial"] - 0.010) < 2e-4 and abs(g["fit"] + 0.005) < 2e-4, f"{g}")
g = ev(depth=0.010, off=(0.0065, 0.0))
check("geometry: 6.5 mm off along one socket axis (face overlaps the wall 1.5 mm): not in the hole", g["in_hole"] == 0, f"fit {g['fit']:.4f}")
g = ev(depth=0.010, off=(0.0049, 0.0049))
check("geometry: 4.9 mm off on both axes (a resting corner, 6.9 mm from the axis): in the hole", g["in_hole"] == 1, f"fit {g['fit']:.4f}")
g = ev(depth=0.010, roll_deg=45)
check("geometry: rolled 45 deg, centred (diagonal 31.1 mm in a 32 mm hole): in the hole", g["in_hole"] == 1, f"fit {g['fit']:.4f}")
g = ev(depth=0.010, roll_deg=45, off=(0.002, 0.0))
check("geometry: rolled 45 deg, 2 mm off: not in the hole", g["in_hole"] == 0, f"fit {g['fit']:.4f}")
g = ev(depth=0.046)
check("geometry: 46 mm in (past half the peg) reads 46 mm, pointing in", abs(g["axial"] - 0.046) < 2e-4 and g["align"] > 0.999, f"{g}")
g = ev(depth=0.020, flip=True)
check("geometry: the peg's other end leading reads the same", abs(g["axial"] - 0.020) < 2e-4 and g["align"] > 0.999, f"{g}")
g = ev(depth=0.058)
check("geometry: a tip 8 mm behind the hole floor is not in the hole", g["in_hole"] == 0, f"axial {g['axial']:.4f}")
g = ev(depth=0.046, pitch_deg=20)
check("geometry: seated at 46 mm; socket pitch reads +20", g["seated"] == 1 and abs(g["socket_pitch"] - 20) < 0.1, f"{g}")
g = ev(depth=0.044)
check("geometry: 44 mm is not seated (floor 50, margin 5)", g["seated"] == 0 and g["in_hole"] == 1)
env.close()

# ---------------------------------------------------------------- 3. physics


def release(e, pitch_deg, depth, seconds=2.0):
    S = place(e, pitch_deg=pitch_deg, depth=depth)
    z3 = torch.zeros((1, 3))
    e.peg.set_linear_velocity(z3)
    e.peg.set_angular_velocity(z3)
    for _ in range(int(seconds * e.sim_freq)):
        e.socket.set_pose(S)  # a perfect grip: the socket's pose and velocity reset before every physics substep
        e.socket.set_linear_velocity(z3)
        e.socket.set_angular_velocity(z3)
        e.scene.step()
    # read the actual poses: the last substep moves the socket slightly, and resetting it would shift the hole
    # around the peg (an earlier version of this probe read a resting peg as 9.5 mm into the wall that way)
    i = e.evaluate()
    return bool(i["seated"][0]), round(float(i["axial"][0]) * 1000, 1), round(float(i["fit"][0]) * 1000, 2)


phys = {}
for mu in (0.3, 0.7):
    env = gym.make("SO101InsertPegRetain-v1", num_envs=1, obs_mode="state", sim_backend="cpu", domain_randomization=True,
                   domain_randomization_config=dict(friction_range=[mu, mu], size_scale_range=[1.0, 1.0], density_range=[350, 350]))
    env.reset(seed=0)
    e = env.unwrapped
    for pitch in (-10, 0, 10, 20):
        for d in (25, 42, 46, 48):
            phys[(mu, pitch, d)] = release(e, pitch, d / 1000)
            print(f"  friction {mu} mouth {pitch:+3d} deg, released {d} mm -> seated, depth mm, fit mm: {phys[(mu, pitch, d)]}")
    env.close()
check("physics: released 25 mm deep, the peg is not seated after 2 s at any pitch or friction",
      not any(v[0] for (mu, p, d), v in phys.items() if d == 25))
check("physics: mouth up 10-20 deg, released 46-48 mm deep: still seated after 2 s at both frictions",
      all(v[0] for (mu, p, d), v in phys.items() if p >= 10 and d >= 46))
check("physics: mouth up 10-20 deg, released 42 mm deep (not yet seated): slides in and is seated after 2 s",
      all(phys[(mu, p, 42)][0] for mu in (0.3, 0.7) for p in (10, 20)))
check("physics: every seated peg passes the hole fit test (fit <= 1 mm)", all(v[2] <= 1.0 for v in phys.values() if v[0]))

# ---------------------------------------------------------------- 4. jaw
env = gym.make("SO101InsertPegRetain-v1", num_envs=1, obs_mode="state", sim_backend="cpu", domain_randomization=False)
env.reset(seed=0)
arm = env.unwrapped.arms[1]
q0 = arm.robot.get_qpos().clone()
errs = []
for jaw in (0.0, 0.1, 0.2, 0.3, 0.4):
    q = q0.clone()
    q[:, -1] = jaw
    arm.robot.set_qpos(q)
    env.unwrapped.scene.step()  # one substep so the link poses follow the new joint angles
    dist = float(torch.linalg.norm(arm.finger1_tip.pose.p - arm.finger2_tip.pose.p))
    errs.append(abs(dist - (JAW_TIP_DIST_0 + JAW_TIP_DIST_PER_RAD * jaw)) * 1000)
check("jaw: fingertip distance = 14.9 mm + 65 mm/rad within 0.5 mm at 0-0.4 rad", max(errs) < 0.5,
      " ".join(f"{x:.2f}" for x in errs) + " mm")
env.close()

print(f"{sum(checks)} of {len(checks)} pass")
print("ALL PASS" if all(checks) else "SOME FAIL")
sys.exit(0 if all(checks) else 1)
