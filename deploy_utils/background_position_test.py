"""Phi 2026-10-13 (after run 3): sim-only. (1) where was the real cube? back-project its red pixels at step 0 through
the sim camera at the logged joint angles onto the table. (2) does the background colour alone break the policy?
(3) does cube position (left/centre/right) matter? Deploy settings: action x0.15, 60 steps, deterministic."""
import os, sys, csv, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np, torch, gymnasium as gym, envs  # noqa: F401
from mani_skill.utils.structs import Pose
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from train_squint import DeployAgent

import argparse
_ap = argparse.ArgumentParser(description="background colour and cube position, in sim")
_ap.add_argument("--checkpoint", required=True)
_ap.add_argument("--flight", default=None, help="a real recording (<record_dir>/flight): estimate where its cube was")
_A = _ap.parse_args()
F = _A.flight
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env = FlattenRGBDObservationWrapper(gym.make("SO101ReachCube-v1", obs_mode="rgb+segmentation", render_mode="sensors",
      num_envs=1, domain_randomization=False, sensor_configs={"width": 128, "height": 128}), rgb=True, depth=False, state=True)
obs, _ = env.reset(seed=0); u = env.unwrapped
agent = DeployAgent(env, sample_obs=obs, device=torch.device("cpu")); agent.load_checkpoint(_A.checkpoint)
BLACK = u._rgb_overlay_image.clone()
half = float(u.item_half_sizes[0])

# (1) cube position from the real step-0 frames
est = {}
for ep in (sorted(os.listdir(F)) if F else []):
    if not ep.startswith("ep") or not os.path.isdir(f"{F}/{ep}"): continue
    row = next(csv.DictReader(open(f"{F}/{ep}/log.csv")))
    q = torch.tensor([[float(row[c]) for c in list(row)[2:8]]], dtype=torch.float32)
    rgb = cv2.cvtColor(cv2.imread(f"{F}/{ep}/step000.png"), cv2.COLOR_BGR2RGB).astype(int)
    red = (rgb[..., 0] > 1.4 * rgb[..., 1]) & (rgb[..., 0] > 1.4 * rgb[..., 2]) & (rgb[..., 0] > 60)
    vs, us = np.nonzero(red)
    u.agent.robot.set_qpos(q); u._update_wrist_camera_pose(); u.scene.update_render()
    p = u.scene.sensors["base_camera"].get_params()
    K, E = p["intrinsic_cv"][0].cpu().numpy(), p["extrinsic_cv"][0].cpu().numpy()
    R, t = E[:, :3], E[:, 3]
    C = -R.T @ t; d = R.T @ (np.linalg.inv(K) @ np.array([us.mean(), vs.mean(), 1.0]))
    lam = (half - C[2]) / d[2]; P = C + lam * d
    est[ep] = (float(P[0]), float(P[1]))
    print(f"run 3 {ep}: cube at about x {100*P[0]:.1f} cm forward, y {100*P[1]:+.1f} cm (+ = left) | {red.sum()} red px at ({us.mean():.0f},{vs.mean():.0f})")
print(f"spawn square in training: x 20..40 cm, y -10..+10 cm | start-pose tip: x 23.0, y -4.0")

def rollout(bg, seed, cube_xy=None, scale=0.15, steps=60):
    obs, _ = env.reset(seed=seed)
    u._rgb_overlay_image = BLACK.clone() if bg is None else torch.tensor(bg, dtype=torch.uint8).expand_as(BLACK).clone()
    if cube_xy is not None:
        x, y = cube_xy
        u.item.set_pose(Pose.create_from_pq(p=[[x, y, half]]))
        u.goal_site.set_pose(Pose.create_from_pq(p=[[x, y, 2 * half + 0.02]]))
        obs = env.observation(u.get_obs())
    dmin, succ = 9.0, False
    for _ in range(steps):
        a = agent.get_action(obs).cpu().numpy()
        obs, _, _, _, info = env.step(np.clip(a * scale, -1, 1))
        dmin = min(dmin, float(torch.linalg.norm(u.goal_site.pose.p - u.agent.tcp_pose.p, axis=1)[0]))
        succ = bool(info["success"][0])
    return succ, dmin

res = {"cube_estimates_cm": {k: [round(100*v[0], 1), round(100*v[1], 1)] for k, v in est.items()}}
BGS = {"black (training)": None, "grey 10 (morning lock)": (10, 10, 10), "grey run3 (32,37,34)": (32, 37, 34)}
print("\n(2) background colour, 10 training-distribution cube positions (seeds 0-9):")
for name, bg in BGS.items():
    r = [rollout(bg, s) for s in range(10)]
    res[f"bg {name}"] = {"success": sum(x[0] for x in r), "min_dist_cm": [round(100*x[1], 1) for x in r]}
    print(f"  {name:24s}: success {sum(x[0] for x in r)}/10 | closest approach (cm): {[round(100*x[1],1) for x in r]}")
print("\n(3) cube position, x 25 cm:")
for name, bg in [("black", None), ("grey run3", (32, 37, 34))]:
    line = []
    for y in (0.08, 0.0, -0.08):
        s, d = rollout(bg, 0, (0.25, y)); line.append(f"y {100*y:+.0f}: {'OK' if s else 'fail'} ({100*d:.1f} cm)")
        res[f"pos {name} y{100*y:+.0f}"] = [s, round(100*d, 1)]
    for ep, xy in est.items():
        s, d = rollout(bg, 0, xy); line.append(f"run3 {ep}: {'OK' if s else 'fail'} ({100*d:.1f} cm)")
        res[f"pos {name} {ep}"] = [s, round(100*d, 1)]
    print(f"  {name:10s}: " + " | ".join(line))
os.makedirs(os.path.join(ROOT, "tune_snapshots"), exist_ok=True)
json.dump(res, open(os.path.join(ROOT, "tune_snapshots", "background_position_test.json"), "w"), indent=1)
