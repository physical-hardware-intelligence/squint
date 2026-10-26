"""B013: chain two trained policies on the whole task. The handover policy acts from a normal start until the receiver
has carried the bar alone for `--settle_steps` steps after its first carrying step (handover latched, receiver
gripping, picker's jaws off the bar): the point at which B008's saved carry starts were taken
(`harvest_carry_starts.py`). From the next action on, the carry policy acts until the episode ends; no switching back.
Without `--carry`, the handover policy acts alone (the control: it should reproduce its own stage counts).

    python experiments/bimanual/chain_eval.py --handover runs/b011_wrist_s1/ckpt_3M.pt \
        --carry runs/b008_wrist_s1/ckpt_1M.pt [--env_id SO101HandoverBarV2-v1] [--seed 7] [--num_envs 128]
A policy is given the first C image channels, C read from its checkpoint (6: the two wrist cameras; 9: plus the top
camera), so a top-camera carry policy needs an env that renders the top camera (`SO101HandoverBarV2TopCamLeHome-v1`,
B009's camera). The channel order is checked against the raw camera images at the first step.
Domain randomization and backgrounds as training (0-50 +-6), GPU simulator, no colour jitter, deterministic actions,
no scene rebuild between episodes (as stage_counts.py). Outcomes are counted at each of `--report_steps` (the task's
120 steps, and more: a late handover leaves the carry policy less than the 60 steps it was trained for).
Prints and writes experiments/bimanual/out/chain_<handover run>_<carry run>_seed<seed>_<time>.json.
`--video DIR` also records every env as training's evaluation videos do (cameras plus a side view; use few envs).
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
from mani_skill.utils.wrappers.record import RecordEpisode  # noqa: E402

import envs  # noqa: E402,F401
from train_squint import DeployAgent  # noqa: E402

CAMS = ("left_wrist", "right_wrist", "top")


class KeepRGB(FlattenRGBDObservationWrapper):
    """Keeps each camera's raw rgb from the last observation, to check which flattened channels belong to which camera."""

    def observation(self, observation):
        self.raw = {c: d["rgb"].clone() for c, d in observation["sensor_data"].items()}
        return super().observation(observation)


def channels_of(checkpoint):
    """Image channels the policy was trained on: the input width of its encoder's first convolution."""
    return int(torch.load(checkpoint, map_location="cpu")["encoder"]["conv.0.weight"].shape[1])


def short(checkpoint):
    """`runs/<run>/<file>`: no machine paths in the written results."""
    p = Path(checkpoint)
    return f"runs/{p.parent.name}/{p.name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--handover", required=True, help="checkpoint that acts from the start")
    ap.add_argument("--carry", default=None, help="checkpoint that takes over once the receiver carries alone")
    ap.add_argument("--env_id", default="SO101HandoverBarV2-v1")
    ap.add_argument("--num_envs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--settle_steps", type=int, default=5, help="as harvest_carry_starts.py")
    ap.add_argument("--report_steps", type=int, nargs="+", default=[120, 180])
    ap.add_argument("--video", default=None, help="directory for a video of all envs (as training's evaluation videos)")
    args = ap.parse_args()
    dev = torch.device("cuda")
    steps = max(args.report_steps)

    env = gym.make(args.env_id, num_envs=args.num_envs, obs_mode="rgb+segmentation", sim_backend="gpu",
                   sensor_configs=dict(width=128, height=128), domain_randomization=True,
                   domain_randomization_config={"background_level_range": [0, 50], "background_tint": 6},
                   max_episode_steps=steps,
                   **(dict(render_mode="all", human_render_camera_configs=dict(shader_pack="default", width=128, height=128))
                      if args.video else {}))
    keep = KeepRGB(env, rgb=True, depth=False, state=True)
    env = FlattenActionSpaceWrapper(keep)
    if args.video:
        env = RecordEpisode(env, output_dir=args.video, save_trajectory=False, save_video=True, max_steps_per_video=steps,
                            video_fps=20)
    u = env.unwrapped
    n = args.num_envs
    obs, _ = env.reset(seed=args.seed)

    # The flattened image stacks the cameras in sensor order: check it, then give each policy its first C channels
    present = [c for c in CAMS if c in keep.raw]
    assert torch.equal(obs["rgb"], torch.cat([keep.raw[c] for c in present], dim=-1)), "camera order in the observation"
    assert present[:2] == ["left_wrist", "right_wrist"], present
    total = obs["rgb"].shape[-1]

    def make_agent(checkpoint):
        c = channels_of(checkpoint)
        assert c in (6, 9) and c <= total, f"{checkpoint} takes {c} channels; {args.env_id} gives {total} ({present})"
        sample = {"rgb": obs["rgb"][..., :c], "state": obs["state"]}
        agent = DeployAgent(env, sample_obs=sample, device=dev)
        agent.load_checkpoint(checkpoint)  # strict: a different state size fails here
        return agent, c

    agent_h, c_h = make_agent(args.handover)
    agent_c, c_c = make_agent(args.carry) if args.carry else (None, None)

    flags = ("picker_grasp", "both_hold", "bar_in_bin", "success", "robot_touching_table", "arms_touching")
    ever = {k: torch.zeros(n, dtype=torch.bool, device=dev) for k in flags + ("carrying", "switched")}
    run_len = torch.zeros(n, dtype=torch.long, device=dev)  # consecutive carrying steps, current run
    switched = torch.zeros(n, dtype=torch.bool, device=dev)
    switch_t = torch.full((n,), -1, dtype=torch.long, device=dev)
    first_bin = torch.full((n,), -1, dtype=torch.long, device=dev)
    reports = {}
    for t in range(steps):
        action = agent_h.get_action({"rgb": obs["rgb"][..., :c_h], "state": obs["state"]})
        if agent_c is not None and switched.any():
            a_c = agent_c.get_action({"rgb": obs["rgb"][..., :c_c], "state": obs["state"]})
            action = torch.where(switched[:, None], a_c, action)
        obs, r, term, trunc, info = env.step(action)
        assert bool((u.elapsed_steps == t + 1).all()), "an env was reset mid-run"
        for k in flags:
            ever[k] |= info[k].to(dev)
        carrying = (info["handed_over"] & info["receiver_grasp"] & ~info["picker_touch_bar"]).to(dev)
        ever["carrying"] |= carrying
        run_len = torch.where(carrying, run_len + 1, torch.zeros_like(run_len))
        if agent_c is not None:
            # harvest_carry_starts.py saved the state at the first carrying step + settle_steps: settle_steps + 1 in a row
            new = ~switched & (run_len >= args.settle_steps + 1)
            switch_t[new] = t
            switched |= new
            ever["switched"] = switched.clone()
        first_bin = torch.where((first_bin < 0) & info["bar_in_bin"].to(dev), torch.full_like(first_bin, t), first_bin)
        if t + 1 in args.report_steps:
            on = switched if agent_c is not None else ever["carrying"]
            reports[t + 1] = dict(
                reached={k: int(v.sum()) for k, v in ever.items()},
                at_end={k: int(info[k].sum()) for k in ("bar_in_bin", "success")},
                after_switch=dict(episodes=int(on.sum()), bar_in_bin=int((on & ever["bar_in_bin"]).sum()),
                                  success=int((on & ever["success"]).sum()),
                                  bar_in_bin_at_end=int((on & info["bar_in_bin"].to(dev)).sum())))

    has = switch_t >= 0
    out = dict(handover=short(args.handover), carry=short(args.carry) if args.carry else None, env_id=args.env_id,
               channels=dict(handover=c_h, carry=c_c, env=total, cameras=present), episodes=n, seed=args.seed,
               settle_steps=args.settle_steps, steps=steps, reports=reports,
               switch_step=dict(median=int(switch_t[has].median()) if has.any() else None,
                                over_100=int((switch_t >= 100).sum())),
               first_in_bin_step_median=int(first_bin[first_bin >= 0].median()) if (first_bin >= 0).any() else None)
    print(json.dumps(out, indent=1))
    name = f"chain_{Path(args.handover).parent.name}_{Path(args.carry).parent.name if args.carry else 'alone'}_seed{args.seed}"
    path = Path(__file__).resolve().parent / "out" / f"{name}_{dt.datetime.now():%Y%m%d-%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    env.close()  # writes the video, if any


if __name__ == "__main__":
    main()
