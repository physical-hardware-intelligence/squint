"""How far does a handover checkpoint get? Runs it on N simulated episodes and counts, per episode, every stage reached.

    python experiments/bimanual/stage_counts.py --checkpoint runs/<run>/ckpt.pt [--num_envs 128] [--seed 7]
Same environment settings as training's evaluation (domain randomization, backgrounds 0-50 +-6), GPU simulator.
Prints and writes experiments/bimanual/out/stages_<run>_<time>.json.
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

FLAGS = ("picker_grasp", "both_hold", "handed_over", "receiver_grasp", "bar_in_bin", "success",
         "arms_touching", "robot_touching_table", "robot_touching_bin")
BANDS = ((0, 2, "reaching"), (2, 4, "picker holds, carrying"), (4, 6, "at handover point, receiver reaching"),
         (6, 7, "both hold"), (7, 9, "receiver carries to bin"), (9, 15.01, "in bin"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num_envs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("SO101HandoverBar-v1", num_envs=args.num_envs, obs_mode="rgb+segmentation", sim_backend="gpu",
                   sensor_configs=dict(width=128, height=128), domain_randomization=True,
                   domain_randomization_config={"background_level_range": [0, 50], "background_tint": 6})
    env = FlattenActionSpaceWrapper(FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True))
    obs, info = env.reset(seed=args.seed)
    agent = DeployAgent(env, sample_obs=obs, device=dev)
    agent.load_checkpoint(args.checkpoint)
    u = env.unwrapped
    n, steps = args.num_envs, u.spec.max_episode_steps or 120
    ever = {k: torch.zeros(n, dtype=torch.bool, device=dev) for k in FLAGS}
    best = torch.full((n,), -1e9, device=dev)
    min_recv_gap = torch.full((n,), 1e9, device=dev)  # receiver TCP to the bar centre, closest approach (m)
    for _ in range(steps):
        obs, r, term, trunc, info = env.step(agent.get_action(obs))
        for k in FLAGS:
            ever[k] |= info[k].to(dev)
        best = torch.maximum(best, u.compute_dense_reward(None, None, info).to(dev))
        tcp_r = u._by_role(u.arms[0].tcp_pos, u.arms[1].tcp_pos, "receiver")
        min_recv_gap = torch.minimum(min_recv_gap, torch.linalg.norm(tcp_r - u.bar.pose.p, dim=1).to(dev))
    out = dict(checkpoint=args.checkpoint, episodes=n, seed=args.seed,
               reached={k: int(v.sum()) for k, v in ever.items()},
               best_stage={name: int(((best >= lo) & (best < hi)).sum()) for lo, hi, name in BANDS},
               receiver_closest_cm=dict(median=round(float(min_recv_gap.median()) * 100, 1),
                                        under_3cm=int((min_recv_gap < 0.03).sum()), under_6cm=int((min_recv_gap < 0.06).sum())))
    print(json.dumps(out, indent=1))
    path = Path(__file__).resolve().parent / "out" / f"stages_{Path(args.checkpoint).parent.name}_{dt.datetime.now():%Y%m%d-%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
