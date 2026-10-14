"""
Score real SO101ReachCube episodes from a flight recording (deploy.py --record_dir), with Squint's own success test
(envs/reach.py evaluate): tip within 2 cm of a point 2 cm above the cube's top, arm static, gripper >= 90% closed.

  * cube: its red pixels at step 0, back-projected through the env's camera at the logged joint angles onto the table
    plane at the cube's half height. Accuracy depends on how well the camera pose matches far from the gripper:
    check a few positions with a ruler before trusting it to the centimetre.
  * tip: Squint's sim robot set to each logged joint angle (the state's first six numbers), agent.tcp_pose.
  * static: no joint moved more than STATIC_DEG between consecutive steps.
  * gripper: logged gripper angle below 3 deg sim (closeness >= 90% of the -10..120 deg range).

Usage (repo root): source env.sh && python deploy_utils/score_reach.py <record_dir>/flight [--positions cubes.csv]
  --positions: CSV with columns ep,x_cm,y_cm (e.g. 0,30,-5): the cube positions measured with a ruler, base frame
               (x forward from the pan axis, y to the robot's left). Used instead of the estimate, which is still
               reported, so the CSV also measures the estimate's error.
  --cube-xy X Y: one measured position in metres for every episode.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import gymnasium as gym
import envs  # noqa: F401

GOAL_ABOVE_TOP = 0.02      # reach.py above_item_goal_height
REACH_THRESH = 0.02        # reach.py target_goal_thresh
STATIC_DEG = 0.5           # [judgement] max joint change between consecutive recorded steps to count as still
GRIPPER_CLOSED_DEG = 3.0   # sim degrees: closeness >= 90% of the -10..120 deg range
SPAWN = {"x": (0.20, 0.40), "y": (-0.10, 0.10)}   # reach.py spawn_box_pos [0.3, 0], half size 0.1


def cube_from_frame(u, rgb, qpos):
    """Back-project the red pixels of a 128x128 policy image to the table plane. None if no cube is visible."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(int)
    red = ((hsv[..., 0] <= 10) | (hsv[..., 0] >= 170)) & (hsv[..., 1] >= 60) & (hsv[..., 2] >= 60)
    if red.sum() < 20:
        return None
    vs, us = np.nonzero(red)
    u.agent.robot.set_qpos(qpos)
    u._update_wrist_camera_pose()
    u.scene.update_render()
    p = u.scene.sensors["base_camera"].get_params()
    K, E = p["intrinsic_cv"][0].cpu().numpy(), p["extrinsic_cv"][0].cpu().numpy()
    R, t = E[:, :3], E[:, 3]
    C = -R.T @ t
    d = R.T @ (np.linalg.inv(K) @ np.array([us.mean(), vs.mean(), 1.0]))
    half = float(u.item_half_sizes[0])
    P = C + (half - C[2]) / d[2] * d
    return float(P[0]), float(P[1]), int(red.sum())


def main():
    ap = argparse.ArgumentParser(description="score real ReachCube episodes from a flight recording")
    ap.add_argument("flight")
    ap.add_argument("--cube-xy", nargs=2, type=float, default=None, metavar=("X", "Y"))
    ap.add_argument("--positions", default=None, help="CSV ep,x_cm,y_cm of ruler-measured cube positions")
    a = ap.parse_args()
    measured = {}
    if a.positions:
        for r in csv.DictReader(open(a.positions)):
            measured[int(r["ep"])] = (float(r["x_cm"]) / 100, float(r["y_cm"]) / 100)

    env = gym.make("SO101ReachCube-v1", obs_mode="rgb+segmentation", render_mode="sensors", num_envs=1,
                   domain_randomization=False, sensor_configs={"width": 128, "height": 128})
    env.reset(seed=0)
    u = env.unwrapped
    half = float(u.item_half_sizes[0])
    report = {}
    for ep in sorted(p for p in Path(a.flight).iterdir() if p.is_dir() and p.name.startswith("ep")):
        rows = list(csv.DictReader(open(ep / "log.csv")))
        q = np.array([[float(r[c]) for c in list(r)[2:8]] for r in rows], dtype=np.float32)   # real qpos, sim radians
        rgb = cv2.cvtColor(cv2.imread(str(ep / "step000.png")), cv2.COLOR_BGR2RGB)
        est = cube_from_frame(u, rgb, torch.from_numpy(q[:1]))
        ep_idx = int(ep.name[2:])
        if ep_idx in measured or a.cube_xy:
            cx, cy = measured.get(ep_idx, tuple(a.cube_xy or (0, 0)))
            src = "measured"
        elif est is not None:
            cx, cy, src = est[0], est[1], f"estimated from {est[2]} red pixels"
        else:
            report[ep.name] = {"error": "no cube visible at step 0"}
            print(f"{ep.name}: no cube visible at step 0, add it to --positions")
            continue
        est_err = (round(100 * float(np.hypot(est[0] - cx, est[1] - cy)), 1) if (est is not None and src == "measured") else None)
        goal = np.array([cx, cy, 2 * half + GOAL_ABOVE_TOP])
        dist, tips = [], []
        for k in range(len(q)):
            u.agent.robot.set_qpos(torch.from_numpy(q[k:k + 1]))
            tip = u.agent.tcp_pose.p[0].cpu().numpy()
            tips.append(tip.tolist())
            dist.append(float(np.linalg.norm(tip - goal)))
        dq = np.degrees(np.abs(np.diff(q, axis=0))).max(axis=1) if len(q) > 1 else np.zeros(0)
        static = np.concatenate([[False], dq <= STATIC_DEG])
        closed = np.degrees(q[:, 5]) < GRIPPER_CLOSED_DEG
        reached = np.array(dist) <= REACH_THRESH
        success = reached & static & closed
        first = lambda m: int(np.argmax(m)) if m.any() else None
        in_spawn = SPAWN["x"][0] <= cx <= SPAWN["x"][1] and SPAWN["y"][0] <= cy <= SPAWN["y"][1]
        report[ep.name] = {
            "cube_xy_cm": [round(100 * cx, 1), round(100 * cy, 1)], "cube_source": src, "cube_in_training_area": in_spawn,
            "estimate_error_cm": est_err,
            "steps": len(q), "start_dist_cm": round(100 * dist[0], 1), "min_dist_cm": round(100 * min(dist), 1),
            "min_dist_step": int(np.argmin(dist)), "end_dist_cm": round(100 * dist[-1], 1),
            "first_reached_step": first(reached), "first_closed_step": first(closed), "first_success_step": first(success),
            "success_at_end": bool(success[-1]), "success_any": bool(success.any()),
            "per_step": {"dist_cm": [round(100 * d, 2) for d in dist], "tip_m": tips}}
        e = report[ep.name]
        print(f"{ep.name}: cube ({e['cube_xy_cm'][0]}, {e['cube_xy_cm'][1]}) cm {'inside' if in_spawn else 'OUTSIDE'} the training "
              f"area [{src}] | tip-goal {e['start_dist_cm']} -> min {e['min_dist_cm']} (step {e['min_dist_step']}) -> "
              f"end {e['end_dist_cm']} cm | reached {e['first_reached_step']}, closed {e['first_closed_step']}, "
              f"success {e['first_success_step']} | SUCCESS at end: {e['success_at_end']}"
              + (f" | camera estimate off by {est_err} cm" if est_err is not None else ""))
    json.dump(report, open(Path(a.flight) / "reach_score.json", "w"), indent=1)
    env.close()


if __name__ == "__main__":
    main()
