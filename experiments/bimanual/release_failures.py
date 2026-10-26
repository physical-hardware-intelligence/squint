"""Where does a handover policy's release fail? For the episodes that reach both hold but never a carry, the state at the
last step: still both holding (frozen), picker alone holding (the receiver let go), receiver gripping with the picker
still touching, or nobody holding (dropped). Also the steps spent in both hold.

    python experiments/bimanual/release_failures.py --checkpoint runs/b011_wrist_s1/ckpt_3M.pt [--seed 7]
Same settings as stage_counts.py (domain randomization, backgrounds 0-50 +-6, GPU, deterministic actions, 120 steps).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper  # noqa: E402

import envs  # noqa: E402,F401
from train_squint import DeployAgent  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--env_id", default="SO101HandoverBarV2-v1")
    ap.add_argument("--num_envs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    dev = torch.device("cuda")
    env = gym.make(args.env_id, num_envs=args.num_envs, obs_mode="rgb+segmentation", sim_backend="gpu",
                   sensor_configs=dict(width=128, height=128), domain_randomization=True,
                   domain_randomization_config={"background_level_range": [0, 50], "background_tint": 6})
    env = FlattenActionSpaceWrapper(FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True))
    obs, _ = env.reset(seed=args.seed)
    agent = DeployAgent(env, sample_obs=obs, device=dev)
    agent.load_checkpoint(args.checkpoint)
    n = args.num_envs
    ever_hold = torch.zeros(n, dtype=torch.bool, device=dev)
    ever_carry = torch.zeros(n, dtype=torch.bool, device=dev)
    hold_steps = torch.zeros(n, device=dev)
    for _ in range(env.unwrapped.spec.max_episode_steps or 120):
        obs, r, term, trunc, info = env.step(agent.get_action(obs))
        hold = info["both_hold"].to(dev)
        ever_hold |= hold
        hold_steps += hold.float()
        ever_carry |= (info["handed_over"] & info["receiver_grasp"] & ~info["picker_touch_bar"]).to(dev)
    f = {k: info[k].to(dev) for k in ("both_hold", "picker_grasp", "receiver_grasp", "picker_touch_bar", "bar_lifted")}
    failed = ever_hold & ~ever_carry
    end = dict(
        frozen_both_hold=f["both_hold"],
        receiver_grips_picker_still_touching=f["receiver_grasp"] & f["picker_touch_bar"] & ~f["both_hold"],
        picker_alone_holds=f["picker_grasp"] & ~f["receiver_grasp"],
        nobody_holds=~f["picker_grasp"] & ~f["receiver_grasp"],
    )
    out = dict(checkpoint=f"runs/{Path(args.checkpoint).parent.name}/{Path(args.checkpoint).name}", seed=args.seed,
               episodes=n, reached_both_hold=int(ever_hold.sum()), reached_carry=int(ever_carry.sum()),
               both_hold_no_carry=int(failed.sum()),
               of_those_at_the_last_step={k: int((v & failed).sum()) for k, v in end.items()},
               of_those_other=int((failed & ~torch.stack(list(end.values())).any(0)).sum()),
               both_hold_steps_median=dict(no_carry=float(hold_steps[failed].median()) if failed.any() else None,
                                           carry=float(hold_steps[ever_carry].median()) if ever_carry.any() else None))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
