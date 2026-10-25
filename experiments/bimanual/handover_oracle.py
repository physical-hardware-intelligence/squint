"""B001: a scripted handover in the simulator (docs/bimanual/experiments/B001-handover-env-check.md).

Top-down grasps from inverse kinematics on the same URDF the simulator loads (pytorch_kinematics). No learning.
Run from the repository root after `source env.sh`:
    .venv/bin/python experiments/bimanual/handover_oracle.py --seeds 1 2 3 4 5 6 7 8 9 10
Writes experiments/bimanual/out/B001_<time>.json (every run's flags, first step of each stage, rewards, run card).
"""

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import pytorch_kinematics as pk  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

import envs  # noqa: E402,F401

REPO = Path(__file__).resolve().parents[2]
CHAIN = pk.build_chain_from_urdf((REPO / "envs/robot/so101.urdf").read_bytes()).to(dtype=torch.float64)
DELTA = np.array([0.1] * 5 + [0.2])  # rad per step at action 1 (pd_joint_target_delta_pos, envs/robot/so101.py L97-106)
OPEN, CLOSED = np.deg2rad(45.0), np.deg2rad(-10.0)
GRASP_LIFT = 0.006  # TCP this far above the bar's centre: the URDF fingertips sit short of the jaws' real ends
STAGES = ("picker_grasp", "both_hold", "handed_over", "receiver_grasp", "bar_in_bin", "success",
          "arms_touching", "robot_touching_table", "robot_touching_bin")


def link_mats(q: np.ndarray) -> dict:
    out = CHAIN.forward_kinematics(torch.tensor(q[None], dtype=torch.float64))
    return {k: out[k].get_matrix()[0].numpy() for k in ("gripper_link", "finger1_tip", "finger2_tip")}


def tcp_and_rot(q: np.ndarray, T: np.ndarray):
    """The TCP (midpoint of the fingertips, as SO101.tcp_pos) and the gripper's rotation, in the world."""
    m = link_mats(q)
    tcp = (m["finger1_tip"][:3, 3] + m["finger2_tip"][:3, 3]) / 2
    return (T @ np.r_[tcp, 1.0])[:3], T[:3, :3] @ m["gripper_link"][:3, :3]


def ik(q0, T, limits, pos, jaw_across=None, carry=None, w=0.03):
    """Arm joints putting the TCP at pos with the fingers pointing straight down (gripper z up).
    jaw_across: the jaws close along the gripper's x axis (fingertip to fingertip, measured in sim), so x must be
    perpendicular to this direction (a bar's long axis) to close across the bar.
    carry: (v_g, want): a direction fixed in the gripper frame that must point along want (orients a held bar)."""
    lo, hi = limits[:5, 0], limits[:5, 1]

    def res(x):
        tcp, R = tcp_and_rot(np.r_[x, q0[5]], T)
        r = [tcp - pos, w * (R[:, 2] - np.array([0, 0, 1.0]))]
        if jaw_across is not None:
            r.append([w * float(R[:, 0] @ jaw_across)])
        if carry is not None:
            r.append(w * (R @ carry[0] - carry[1]))
        return np.concatenate([np.ravel(a) for a in r])

    best = None
    for roll in (0.0, np.pi, -np.pi):
        start = np.clip(q0[:5] + np.array([0, 0, 0, 0, roll]), lo + 1e-6, hi - 1e-6)
        sol = least_squares(res, start, bounds=(lo, hi), xtol=1e-10, ftol=1e-10)
        if best is None or sol.cost < best.cost:
            best = sol
    q = np.r_[best.x, q0[5]]
    return q, float(np.linalg.norm(tcp_and_rot(q, T)[0] - pos))


class Runner:
    """Drives each arm's controller target toward a goal at the controller's top speed, logging every step."""

    def __init__(self, env):
        self.env, self.u = env, env.unwrapped
        self.goal = [self.target(a) for a in range(2)]
        self.log = []

    def target(self, a):
        return self.u.arms[a].controller.get_state()["target_qpos"][0].numpy().copy()

    def qpos(self, a):
        return self.u.arms[a].robot.get_qpos()[0].numpy().copy()

    def step(self):
        act = {uid: torch.tensor(np.clip((self.goal[a] - self.target(a)) / DELTA, -1, 1)[None], dtype=torch.float32)
               for a, uid in enumerate(("so101-0", "so101-1"))}
        _, r, _, _, info = self.env.step(act)
        self.log.append((float(r[0]), {k: bool(v[0]) for k, v in info.items() if v.dtype == torch.bool}))
        return info

    def go(self, a, q, steps_max=60, settle=0.02):
        self.goal[a] = q.copy()
        for _ in range(steps_max):
            info = self.step()
            if np.max(np.abs(self.qpos(a)[:5] - q[:5])) < settle and np.max(np.abs(self.target(a) - q)) < 1e-4:
                break
        return info

    def wait(self, n):
        info = None
        for _ in range(n):
            info = self.step()
        return info


def run(seed: int) -> dict:
    env = gym.make("SO101HandoverBar-v1", num_envs=1, obs_mode="state_dict", sim_backend="cpu", max_episode_steps=600)
    env.reset(seed=seed)
    u = env.unwrapped
    limits = u.arms[0].robot.get_qlimits()[0].numpy()
    run_ = Runner(env)
    p, r = int(u.picker[0]), 1 - int(u.picker[0])
    Tp, Tr = (u.arms[a].robot.pose.to_transformation_matrix()[0].numpy() for a in (p, r))
    toward_r = np.array([0.0, -1.0 if p == 0 else 1.0, 0.0])
    errs = {}

    bar_p = u.bar.pose.p[0].numpy()
    axis = u.bar.pose.to_transformation_matrix()[0, :3, 0].numpy()
    hw, g = float(u.bar_half_widths[0]), float(u.bar_half_lengths[0]) - 0.012
    ends = [bar_p + axis * g, bar_p - axis * g]
    near = int(np.argmin([np.linalg.norm(tcp_and_rot(run_.qpos(p), Tp)[0] - e) for e in ends]))
    grasp_pt, free_pt = ends[near], ends[1 - near]

    # The picker grasps its end of the bar from above
    q = run_.qpos(p)
    q[5] = OPEN
    q, errs["picker_above"] = ik(q, Tp, limits, grasp_pt + [0, 0, 0.07], jaw_across=axis)
    run_.go(p, q)
    q, errs["picker_grasp"] = ik(q, Tp, limits, np.r_[grasp_pt[:2], hw + GRASP_LIFT], jaw_across=axis)
    run_.go(p, q)
    q[5] = CLOSED
    run_.go(p, q, steps_max=15)
    run_.wait(5)
    v_g = tcp_and_rot(run_.qpos(p), Tp)[1].T @ (free_pt - grasp_pt)  # the free end's direction, gripper frame
    v_g[2] = 0.0
    v_g /= np.linalg.norm(v_g)

    # Lift, then bring the bar to the handover point with its free end toward the receiver
    H = u.handover_point.numpy()
    q, errs["picker_lift"] = ik(q, Tp, limits, np.r_[grasp_pt[:2], 0.12], carry=(v_g, toward_r))
    run_.go(p, q)
    q, errs["picker_handover"] = ik(q, Tp, limits, H - toward_r * g, carry=(v_g, toward_r))
    run_.go(p, q)
    run_.wait(5)

    # The receiver grasps the free end from above
    q2 = run_.qpos(r)
    q2[5] = OPEN
    free_now = H + toward_r * g
    q2, errs["receiver_above"] = ik(q2, Tr, limits, free_now + [0, 0, 0.06], jaw_across=toward_r)
    run_.go(r, q2)
    q2, errs["receiver_grasp"] = ik(q2, Tr, limits, free_now + [0, 0, GRASP_LIFT], jaw_across=toward_r)
    run_.go(r, q2)
    q2[5] = CLOSED
    run_.go(r, q2, steps_max=15)
    run_.wait(5)

    # The picker lets go and goes back to its start pose
    q[5] = OPEN
    run_.go(p, q, steps_max=15)
    run_.wait(5)
    q, errs["picker_retreat"] = ik(q, Tp, limits, tcp_and_rot(run_.qpos(p), Tp)[0] + [0, 0, 0.08])
    run_.go(p, q)
    rest = u.rest_qpos.numpy().copy()
    run_.go(p, rest)

    # The receiver carries the bar over the bin, along the bin's long side, lowers it and lets go
    c_g = tcp_and_rot(run_.qpos(r), Tr)[1].T @ (-toward_r)  # bar centre direction from the receiver's grip
    c_g[2] = 0.0
    c_g /= np.linalg.norm(c_g)
    bin_p = u.bin.pose.p[0].numpy()
    bin_long = u.bin.pose.to_transformation_matrix()[0, :3, 1].numpy()
    best = None
    for want in (bin_long, -bin_long):
        tgt = bin_p - want * g
        qq, e = ik(q2, Tr, limits, np.r_[tgt[:2], 0.10], carry=(c_g, want))
        if best is None or e < best[1]:
            best = (qq, e, want)
    q2, errs["receiver_over_bin"], want = best
    run_.go(r, q2)
    tgt = bin_p - want * g
    q2, errs["receiver_lower"] = ik(q2, Tr, limits, np.r_[tgt[:2], float(u.bin_thickness) + hw + GRASP_LIFT + 0.004],
                                     carry=(c_g, want))
    run_.go(r, q2)
    q2[5] = OPEN
    run_.go(r, q2, steps_max=15)
    run_.wait(5)
    q2, errs["receiver_retreat"] = ik(q2, Tr, limits, tcp_and_rot(run_.qpos(r), Tr)[0] + [0, 0, 0.08])
    run_.go(r, q2)
    run_.go(r, rest)
    final = run_.wait(10)

    first = {}
    for t, (_, flags) in enumerate(run_.log):
        for k in STAGES:
            if flags.get(k) and k not in first:
                first[k] = t
    rewards = [x for x, _ in run_.log]
    env.close()
    return dict(seed=seed, picker="left" if p == 0 else "right", steps=len(run_.log), success=bool(final["success"][0]),
                first_step=first, reward_max=round(max(rewards), 4), reward_end=round(rewards[-1], 4),
                ik_error_mm={k: round(v * 1000, 2) for k, v in errs.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    args = ap.parse_args()
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    runs = []
    for s in args.seeds:
        runs.append(run(s))
        print(json.dumps(runs[-1]))
    out = dict(experiment="B001", date=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
               commit=git("rev-parse", "HEAD"), dirty=git("status", "--porcelain"),
               command="python experiments/bimanual/handover_oracle.py --seeds " + " ".join(map(str, args.seeds)),
               successes=sum(r["success"] for r in runs), n=len(runs), runs=runs)
    path = REPO / "experiments/bimanual/out" / f"B001_{dt.datetime.now():%Y%m%d-%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"{out['successes']} of {out['n']} succeeded; wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
