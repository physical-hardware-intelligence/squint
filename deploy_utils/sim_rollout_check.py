"""Phi 2026-10-13: when does the policy reach, stop and close its gripper IN SIM? Training settings (action x1.0,
50 steps) vs our deploy settings (action x0.15, 60 steps). Nominal sim (no randomisation), deterministic actions."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, gymnasium as gym, envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from train_squint import DeployAgent

import argparse
_ap = argparse.ArgumentParser(description="when does the policy reach, stop and close its gripper in sim?")
_ap.add_argument("--checkpoint", required=True)
_ap.add_argument("--seeds", type=int, default=5)
_A = _ap.parse_args()
CKPT = _A.checkpoint
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env = FlattenRGBDObservationWrapper(gym.make("SO101ReachCube-v1", obs_mode="rgb+segmentation", render_mode="sensors",
      num_envs=1, domain_randomization=False, sensor_configs={"width": 128, "height": 128}), rgb=True, depth=False, state=True)
obs, _ = env.reset(seed=0)
agent = DeployAgent(env, sample_obs=obs, device=torch.device("cpu")); agent.load_checkpoint(CKPT)
u = env.unwrapped
results = {}
for label, scale, steps in [("train x1.0, 50 steps", 1.0, 50), ("deploy x0.15, 60 steps", 0.15, 60)]:
    runs = []
    for seed in range(_A.seeds):
        obs, _ = env.reset(seed=seed)
        first = {"reached": None, "gripper_closed": None, "success": None}
        for t in range(steps):
            a = agent.get_action(obs).cpu().numpy()
            obs, _, _, _, info = env.step(np.clip(a * scale, -1, 1))
            d = float(torch.linalg.norm(u.goal_site.pose.p - u.agent.tcp_pose.p, axis=1)[0])
            g = float(np.degrees(u.agent.robot.get_qpos()[0, -1].item()))
            for k, cond in (("reached", d <= 0.02), ("gripper_closed", g < 3.0), ("success", bool(info["success"][0]))):
                if cond and first[k] is None:
                    first[k] = t + 1
        runs.append({"seed": seed, **{f"first_{k}": v for k, v in first.items()},
                     "final_dist_cm": round(100 * d, 1), "final_gripper_deg": round(g, 1), "final_success": bool(info["success"][0])})
    results[label] = runs
    ok = sum(r["final_success"] for r in runs)
    print(f"\n{label}: success at the last step {ok}/{_A.seeds}")
    for r in runs:
        print(f"  seed {r['seed']}: reached at step {r['first_reached']}, gripper closed at {r['first_gripper_closed']}, "
              f"success at {r['first_success']} | end: {r['final_dist_cm']} cm, gripper {r['final_gripper_deg']} deg, success {r['final_success']}")
os.makedirs(os.path.join(ROOT, "tune_snapshots"), exist_ok=True)
json.dump(results, open(os.path.join(ROOT, "tune_snapshots", "sim_rollout_check.json"), "w"), indent=1)
