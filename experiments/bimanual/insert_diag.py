"""How far does an insertion checkpoint get? Runs it on N simulated episodes and counts, per episode, every stage reached
(at any step) on the way to a retained insertion, then what happened after the peg arm let go.

    python experiments/bimanual/insert_diag.py --checkpoint CKPT [--env_id SO101InsertPegRetain-v1] [--num_envs 256]
Domain randomization and backgrounds as training (0-50 +-6), GPU simulator. Works for both task 2 envs: the retain
counts (released, fell out) are read from the same flags, and `success` is the env's own test.
Prints and writes experiments/bimanual/out/insert_<run>_<time>.json (or --out).
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper  # noqa: E402

import envs  # noqa: E402,F401
from train_squint import DeployAgent  # noqa: E402

FLAGS = ("socket_grasp", "peg_grasp", "both_lifted", "in_hole", "seated", "success", "arms_touching", "robot_touching_table")

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--env_id", default="SO101InsertPegRetain-v1")
ap.add_argument("--num_envs", type=int, default=256)
ap.add_argument("--seed", type=int, default=8)
ap.add_argument("--out", default=None)
args = ap.parse_args()
dev = torch.device("cuda")
env = gym.make(args.env_id, num_envs=args.num_envs, obs_mode="rgb+segmentation", sim_backend="gpu",
               sensor_configs=dict(width=128, height=128), domain_randomization=True,
               domain_randomization_config={"background_level_range": [0, 50], "background_tint": 6})
env = FlattenActionSpaceWrapper(FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True))
obs, _ = env.reset(seed=args.seed)
agent = DeployAgent(env, sample_obs=obs, device=dev)
agent.load_checkpoint(args.checkpoint)
T = gym.spec(args.env_id).max_episode_steps  # the wrapped env's spec has no step limit
keys = FLAGS + ("released", "socket_up")
R = {k: [] for k in keys}
axial, pitch, retreat = [], [], []
with torch.no_grad():
    for t in range(T):
        obs, _, _, _, info = env.step(agent.get_action(obs))
        for k in keys:
            R[k].append(info[k].to(dev))
        axial.append(info["axial"].to(dev))
        pitch.append(info["socket_pitch"].to(dev))
        retreat.append(info["retreat"].to(dev))
R = {k: torch.stack(v) for k, v in R.items()}  # (T, n)
axial, pitch, retreat = torch.stack(axial), torch.stack(pitch), torch.stack(retreat)
n = args.num_envs
nan = torch.full((n,), float("nan"), device=dev)

# The release: the first step the peg arm is off the peg while the peg is seated in the held-up socket
let_go = R["released"] & R["seated"] & R["socket_up"]
did = let_go.any(0)
first = torch.where(did, let_go.float().argmax(0), torch.full((n,), T - 1, device=dev))
after = torch.arange(T, device=dev)[:, None] >= first[None]
fell = (after & ~R["in_hole"]).any(0) & did  # left the hole at any step after the release
at = lambda x: torch.where(did, x.gather(0, first[None]).squeeze(0), nan)  # noqa: E731  value at the release step


def q(x):
    """10th, 50th, 90th percentiles over the episodes where x is defined."""
    x = x[~torch.isnan(x)]
    return [round(float(x.quantile(p)), 1) for p in (0.1, 0.5, 0.9)] if len(x) else None


out = dict(
    checkpoint=args.checkpoint, env_id=args.env_id, episodes=n, seed=args.seed, steps=T,
    reached={k: int(R[k].any(0).sum()) for k in FLAGS},
    at_end={k: int(R[k][-1].sum()) for k in FLAGS},
    released_seated=int(did.sum()),
    fell_out_after_release=int(fell.sum()),
    success_last_10_steps=int(R["success"][-10:].all(0).sum()),
    first_success_step=q(torch.where(R["success"].any(0), R["success"].float().argmax(0).float(), nan)),
    max_depth_mm=q(torch.where(R["in_hole"].any(0), torch.where(R["in_hole"], axial, -1.0).max(0).values * 1000, nan)),
    socket_pitch_deg_while_both_up=q(torch.where(R["both_lifted"].any(0),
                                                 (pitch * R["both_lifted"]).sum(0) / R["both_lifted"].sum(0).clamp(min=1), nan)),
    depth_at_release_mm=q(at(axial) * 1000),
    pitch_at_release_deg=q(at(pitch)),
    retreat_at_end_cm=q(torch.where(did, retreat[-1] * 100, nan)),
)
print(json.dumps(out, indent=1))
path = Path(args.out) if args.out else (Path(__file__).parent / "out" /
                                        f"insert_{Path(args.checkpoint).parent.name}_{dt.datetime.now():%Y%m%d-%H%M%S}.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out, indent=1))
print("wrote", path)
