"""
What did the policy see, and did the background change its decisions? Reads a flight_recorder.py recording.

For every recorded step:
  * robot outline: Squint's sim rendered at the logged joint angles with the env's camera, dilated ROBOT_DILATE_PX
  * cube: red-to-pink hue with real saturation (so its highlights count), +2 px edge
  * cloth level: 25th-percentile brightness of the pixels that are neither robot nor cube (a low percentile stays
    on the cloth even when a bright wall fills much of the view)
  * room: those pixels more than ROOM_MARGIN brighter than the cloth level (walls, floor, lights beyond the cloth)
  * two counterfactuals, each scored as |policy action change| (deterministic, actions in -1..1):
      room -> cloth : the room painted with the cloth colour (does the view beyond the cloth matter?)
      all -> black  : every non-robot, non-cube pixel set to 0, training's background (does the cloth's grey matter?)
  * replay check: the action on the recorded frame must equal the logged action
Writes <flight>/analysis.json and a contact sheet per episode: real 128 | real 16x16 | sim at same pose | room mask.
Usage (repo root): source env.sh && python deploy_utils/analyze_flight.py <record_dir>/flight --checkpoint runs/<run>/ckpt.pt
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import gymnasium as gym
import envs  # noqa: F401
from mani_skill.utils.structs import Link
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from train_squint import DeployAgent

ROOM_MARGIN = 40             # luminance levels (0-255) above the cloth level that count as "room"
ROBOT_DILATE_PX = 6          # at 128 px: absorbs a few pixels of real/sim misalignment of the jaws


def to16(img):
    return cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA)


def up(img, n=128):
    return cv2.resize(img, (n, n), interpolation=cv2.INTER_NEAREST)


def main(flight_dir, checkpoint):
    flight = Path(flight_dir)
    env = FlattenRGBDObservationWrapper(gym.make(
        "SO101ReachCube-v1", obs_mode="rgb+segmentation", render_mode="sensors", num_envs=1,
        domain_randomization=False, sensor_configs={"width": 128, "height": 128}), rgb=True, depth=False, state=True)
    obs0, _ = env.reset(seed=0)
    u = env.unwrapped
    robot_ids = torch.tensor([i for i, v in u.segmentation_id_map.items() if isinstance(v, Link)])
    agent = DeployAgent(env, sample_obs=obs0, device=torch.device("cpu"))
    agent.load_checkpoint(checkpoint)
    kernel = np.ones((2 * ROBOT_DILATE_PX + 1,) * 2, np.uint8)

    report = {}
    for ep in sorted(p for p in flight.iterdir() if p.is_dir() and p.name.startswith("ep")):
        rows = list(csv.DictReader(open(ep / "log.csv")))
        steps, tiles = [], []
        for r in rows:
            k = int(r["step"])
            rgb = cv2.cvtColor(cv2.imread(str(ep / f"step{k:03d}.png")), cv2.COLOR_BGR2RGB)
            state = torch.tensor([[float(r[c]) for c in list(r)[2:14]]], dtype=torch.float32)
            # sim at the same pose: robot outline + the view training would have produced
            u.agent.robot.set_qpos(state[:, :6])
            u._update_wrist_camera_pose()
            sd = u.get_obs()["sensor_data"]["base_camera"]
            sim_rgb = sd["rgb"][0].cpu().numpy()
            robot = torch.isin(sd["segmentation"][0, ..., 0].cpu(), robot_ids).numpy().astype(np.uint8)
            robot = cv2.dilate(robot, kernel).astype(bool)
            s = rgb.astype(int)
            lum = s.mean(2)
            # the cube by hue, so its pink highlights count as cube too: red-to-pink hue, real saturation, +2 px edge
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(int)
            red = ((hsv[..., 0] <= 10) | (hsv[..., 0] >= 170)) & (hsv[..., 1] >= 60) & (hsv[..., 2] >= 60)
            red = cv2.dilate(red.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            bg = ~red & ~robot
            cloth_level = float(np.percentile(lum[bg], 25)) if bg.any() else 0.0
            room = bg & (lum > cloth_level + ROOM_MARGIN)
            cloth = bg & ~room
            cloth_value = np.median(rgb[cloth], axis=0) if cloth.any() else np.zeros(3)
            room_painted = rgb.copy(); room_painted[room] = cloth_value.astype(np.uint8)
            all_black = rgb.copy(); all_black[bg] = 0

            def act(img):
                with torch.no_grad():
                    return agent.get_action({"rgb": torch.from_numpy(img)[None], "state": state}).numpy()[0]

            a_real, a_room, a_black = act(rgb), act(room_painted), act(all_black)
            logged = np.array([float(r[f"action_{i}"]) for i in range(6)])
            change16 = lambda img: round(float(np.abs(to16(rgb).astype(int) - to16(img).astype(int)).mean()), 2)
            steps.append({
                "step": k, "cloth_level": round(cloth_level, 1), "room_frac_128": round(float(room.mean()), 4),
                "input_change_16_room": change16(room_painted), "action_delta_room_max": round(float(np.abs(a_real - a_room).max()), 4),
                "input_change_16_black": change16(all_black), "action_delta_black_max": round(float(np.abs(a_real - a_black).max()), 4),
                "replay_error": round(float(np.abs(a_real - logged).max()), 6)})
            mask_img = np.zeros_like(rgb); mask_img[room] = (255, 60, 60); mask_img[robot] = (90, 90, 90)
            tiles.append(np.hstack([rgb, up(to16(rgb)), sim_rgb, mask_img]))
        # contact sheet: every step, 4 panels each, captioned
        sheet_rows = []
        for t, st in zip(tiles, steps):
            t = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
            cv2.putText(t, f"{st['step']}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(t, f"room {100 * st['room_frac_128']:.0f}% da {st['action_delta_room_max']:.2f}", (390, 124),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            sheet_rows.append(t)
        cols = 3
        while len(sheet_rows) % cols:
            sheet_rows.append(np.zeros_like(sheet_rows[0]))
        grid = np.vstack([np.hstack(sheet_rows[i:i + cols]) for i in range(0, len(sheet_rows), cols)])
        head = np.full((22, grid.shape[1], 3), 30, np.uint8)
        for c in range(cols):
            for x, lab in [(4, "REAL (policy in)"), (132, "REAL 16x16"), (260, "SIM same pose"), (388, "ROOM = red")]:
                cv2.putText(head, lab, (c * 512 + x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)
        cv2.imwrite(str(flight / f"{ep.name}_contact_sheet.png"), np.vstack([head, grid]))
        mean = lambda key: round(float(np.mean([s[key] for s in steps])), 4)
        peak = lambda key: round(float(np.max([s[key] for s in steps])), 4)
        report[ep.name] = {
            "steps": len(steps), "replay_max_error": max(s["replay_error"] for s in steps),
            "cloth_level_mean": mean("cloth_level"),
            "room_frac_128_mean": mean("room_frac_128"), "room_frac_128_max": peak("room_frac_128"),
            "steps_with_room_over_5pct": int(sum(s["room_frac_128"] > 0.05 for s in steps)),
            "action_delta_room_mean": mean("action_delta_room_max"), "action_delta_room_max": peak("action_delta_room_max"),
            "action_delta_black_mean": mean("action_delta_black_max"), "action_delta_black_max": peak("action_delta_black_max"),
            "per_step": steps}
        e = report[ep.name]
        print(f"{ep.name}: {len(steps)} steps | replay error {e['replay_max_error']:.1e} | cloth level {e['cloth_level_mean']:.1f} "
              f"| room mean {100 * e['room_frac_128_mean']:.1f}% max {100 * e['room_frac_128_max']:.1f}% "
              f"(steps >5%: {e['steps_with_room_over_5pct']}) | action change: room->cloth mean {e['action_delta_room_mean']:.3f} "
              f"max {e['action_delta_room_max']:.3f}; all->black mean {e['action_delta_black_mean']:.3f} max {e['action_delta_black_max']:.3f}")
    json.dump(report, open(flight / "analysis.json", "w"), indent=1)
    env.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("flight", help="<record_dir>/flight, as written by deploy.py --record_dir")
    ap.add_argument("--checkpoint", required=True, help="the checkpoint that was deployed")
    a = ap.parse_args()
    main(a.flight, a.checkpoint)
