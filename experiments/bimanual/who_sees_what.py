"""Which wrist camera sees the bar and the bin, in each phase of a handover checkpoint's episodes?

    python experiments/bimanual/who_sees_what.py --checkpoint runs/<run>/ckpt.pt [--env_id ...] [--num_envs 128] [--seed 7]
Exact: counts the bar's and the bin's pixels in each 128 x 128 segmentation image (occlusion included). "Seen" means at
least 64 pixels, one pixel's worth of the policy's 16 x 16 input. The policy gets both images, so "either" is what it has.
Prints and writes experiments/bimanual/out/sees_<run>_<time>.json.
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

MIN_PIXELS = 64  # 8 x 8 raw pixels = one pixel after the 128 -> 16 area downsample


class KeepSegmentation(FlattenRGBDObservationWrapper):
    """The flattening wrapper drops segmentation; keep each camera's segmentation image from the last step."""

    def observation(self, observation):
        self.seg = {k: v["segmentation"][..., 0] for k, v in observation["sensor_data"].items()}
        return super().observation(observation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--env_id", default="SO101HandoverBar-v1")
    ap.add_argument("--num_envs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    dev = torch.device("cuda")
    env = gym.make(args.env_id, num_envs=args.num_envs, obs_mode="rgb+segmentation", sim_backend="gpu",
                   sensor_configs=dict(width=128, height=128), domain_randomization=True,
                   domain_randomization_config={"background_level_range": [0, 50], "background_tint": 6})
    keep = KeepSegmentation(env, rgb=True, depth=False, state=True)
    env = FlattenActionSpaceWrapper(keep)
    obs, info = env.reset(seed=args.seed)
    agent = DeployAgent(env, sample_obs=obs, device=dev)
    agent.load_checkpoint(args.checkpoint)
    u = env.unwrapped
    ids = {"bar": u.bar.per_scene_id.to(dev).view(-1, 1, 1), "bin": u.bin.per_scene_id.to(dev).view(-1, 1, 1)}
    phases = ("start", "picker holds", "both hold", "receiver carries")
    keys = [f"{cam} sees {obj}" for cam in ("picker cam", "receiver cam", "either cam") for obj in ("bar", "bin")]
    tally = {ph: {k: 0 for k in keys + ["steps"]} for ph in phases}

    def count(phase_mask):
        seen = {}
        for obj, i in ids.items():
            px = {arm: (keep.seg[f"{arm}_wrist"].to(dev) == i).flatten(1).sum(1) >= MIN_PIXELS for arm in ("left", "right")}
            seen[("picker cam", obj)] = u._by_role(px["left"], px["right"], "picker").to(dev)
            seen[("receiver cam", obj)] = u._by_role(px["left"], px["right"], "receiver").to(dev)
            seen[("either cam", obj)] = px["left"] | px["right"]
        return {f"{c} sees {o}": int((v & phase_mask).sum()) for (c, o), v in seen.items()}

    def add(name, mask):
        for k, v in count(mask).items():
            tally[name][k] += v
        tally[name]["steps"] += int(mask.sum())

    add("start", torch.ones(args.num_envs, dtype=torch.bool, device=dev))  # the images before the first action
    for _ in range(u.spec.max_episode_steps or 120):
        obs, r, term, trunc, info = env.step(agent.get_action(obs))
        hold = info["both_hold"].to(dev)
        add("picker holds", (info["picker_grasp"] & ~info["handed_over"]).to(dev) & ~hold)
        add("both hold", hold)
        add("receiver carries", (info["handed_over"] & info["receiver_grasp"] & ~info["picker_touch_bar"]).to(dev))
    out = dict(checkpoint=args.checkpoint, env_id=args.env_id, episodes=args.num_envs, seed=args.seed,
               min_pixels=MIN_PIXELS, steps_where_seen=tally)
    print(json.dumps(out, indent=1))
    path = Path(__file__).resolve().parent / "out" / f"sees_{Path(args.checkpoint).parent.name}_{dt.datetime.now():%Y%m%d-%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
