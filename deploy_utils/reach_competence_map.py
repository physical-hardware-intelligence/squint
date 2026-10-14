"""
Where on the table does a ReachCube policy succeed? Sim, deploy settings (action x0.15, deterministic), one rollout
per cube position on a grid that covers the training spawn area (x 20-40 cm, y -10..10 cm) and beyond it.

Writes <out>.json and <out>.png: one cell per position, green = success at the last step, colour shade = closest
approach of the tip to the goal (2 cm above the cube's top); the training area is outlined.
Usage (repo root): source env.sh && python deploy_utils/reach_competence_map.py --checkpoint runs/<run>/ckpt.pt
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import cv2
import numpy as np
import torch
import gymnasium as gym
import envs  # noqa: F401
from mani_skill.utils.structs import Pose
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from train_squint import DeployAgent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--action-scale", type=float, default=0.15)
    ap.add_argument("--xs", type=float, nargs="+", default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
    ap.add_argument("--ys", type=float, nargs="+", default=[0.15, 0.10, 0.05, 0.0, -0.05, -0.10, -0.15])
    ap.add_argument("--out", default=os.path.join(ROOT, "tune_snapshots", "reach_competence_map"))
    a = ap.parse_args()

    env = FlattenRGBDObservationWrapper(gym.make("SO101ReachCube-v1", obs_mode="rgb+segmentation", render_mode="sensors",
          num_envs=1, domain_randomization=False, sensor_configs={"width": 128, "height": 128}), rgb=True, depth=False, state=True)
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    agent = DeployAgent(env, sample_obs=obs, device=torch.device("cpu"))
    agent.load_checkpoint(a.checkpoint)
    half = float(u.item_half_sizes[0])

    cells = []
    for y in a.ys:
        for x in a.xs:
            env.reset(seed=0)
            u.item.set_pose(Pose.create_from_pq(p=[[x, y, half]]))
            u.goal_site.set_pose(Pose.create_from_pq(p=[[x, y, 2 * half + 0.02]]))
            obs = env.observation(u.get_obs())
            dmin, succ = 9.0, False
            for _ in range(a.steps):
                act = agent.get_action(obs).cpu().numpy()
                obs, _, _, _, info = env.step(np.clip(act * a.action_scale, -1, 1))
                dmin = min(dmin, float(torch.linalg.norm(u.goal_site.pose.p - u.agent.tcp_pose.p, axis=1)[0]))
                succ = bool(info["success"][0])
            cells.append({"x_cm": round(100 * x, 1), "y_cm": round(100 * y, 1), "success_at_end": succ, "min_dist_cm": round(100 * dmin, 1)})
            print(f"  x {100 * x:5.1f}  y {100 * y:+5.1f}  {'OK  ' if succ else 'fail'}  closest {100 * dmin:5.1f} cm", flush=True)

    # picture: rows = y (left at the top), columns = x (forward to the right)
    cw, ch, m = 90, 60, 70
    img = np.full((m + ch * len(a.ys) + 40, m + cw * len(a.xs) + 20, 3), 255, np.uint8)
    for i, y in enumerate(a.ys):
        for j, x in enumerate(a.xs):
            c = next(c for c in cells if c["x_cm"] == round(100 * x, 1) and c["y_cm"] == round(100 * y, 1))
            shade = min(1.0, c["min_dist_cm"] / 20.0)
            col = (90, 200, 90) if c["success_at_end"] else (int(80 + 150 * (1 - shade)), int(80 + 60 * (1 - shade)), 230)
            x0, y0 = m + j * cw, m + i * ch
            cv2.rectangle(img, (x0 + 2, y0 + 2), (x0 + cw - 2, y0 + ch - 2), col, -1)
            cv2.putText(img, f"{c['min_dist_cm']:.1f}", (x0 + 20, y0 + 37), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
        cv2.putText(img, f"y {100 * y:+.0f}", (5, m + i * ch + 37), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
    for j, x in enumerate(a.xs):
        cv2.putText(img, f"x {100 * x:.0f}", (m + j * cw + 22, m - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
    # training area outline: x 20..40, y -10..10 (cell centres at the grid values)
    def px(xv): return int(m + (np.interp(xv, a.xs, range(len(a.xs))) + 0.5) * cw)
    def py(yv): return int(m + (np.interp(yv, a.ys[::-1], range(len(a.ys))[::-1]) + 0.5) * ch)
    cv2.rectangle(img, (px(0.20) - cw // 2, py(0.10) - ch // 2), (px(0.40) + cw // 2, py(-0.10) + ch // 2), (0, 0, 0), 2)
    cv2.putText(img, "green = success; number = closest approach (cm); box = training spawn area; robot base at x 0, y 0",
                (5, img.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cv2.imwrite(a.out + ".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    json.dump({"checkpoint": a.checkpoint, "steps": a.steps, "action_scale": a.action_scale, "cells": cells},
              open(a.out + ".json", "w"), indent=1)
    inside = [c for c in cells if 20 <= c["x_cm"] <= 40 and -10 <= c["y_cm"] <= 10]
    outside = [c for c in cells if c not in inside]
    print(f"success inside the training area: {sum(c['success_at_end'] for c in inside)}/{len(inside)} | "
          f"outside: {sum(c['success_at_end'] for c in outside)}/{len(outside)} | saved {a.out}.png")
    env.close()


if __name__ == "__main__":
    main()
