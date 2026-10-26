"""Save carry starts for B008: run trained handover policies and, once in an episode the receiver has held the bar
alone (picker's jaws off it, after a mid-air handover) for `--settle_steps`, save that env's simulator state, its controller
targets, which arm picked, its background colour and gripper drive (drawn per episode), and the two wrist images from
the episode's first step.

    python experiments/bimanual/harvest_carry_starts.py --checkpoints runs/a/ckpt_3M.pt runs/b/ckpt_3M.pt \
        [--episodes 10] [--num_envs 512] [--out runs/carry_starts.pt]
Domain randomization and backgrounds as training (0-50 +-6); no colour jitter (an observation wrapper, not the scene).
Use the training env count:
saved states only fit the env index they came from, and the carry envs check the object sizes at load.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper  # noqa: E402

import envs  # noqa: E402,F401
from envs.handover_carry import CarryStartMixin, controllers_of  # noqa: E402
from train_squint import DeployAgent  # noqa: E402

CAMS = ("left_wrist", "right_wrist")


class KeepSensors(FlattenRGBDObservationWrapper):
    """Keeps each wrist camera's raw rgb and segmentation from the last observation (the flattening drops them)."""

    def observation(self, observation):
        self.raw = {c: {k: observation["sensor_data"][c][k].clone() for k in ("rgb", "segmentation")} for c in CAMS}
        return super().observation(observation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--num_envs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--out", default="runs/carry_starts.pt")
    ap.add_argument("--env_id", default="SO101HandoverBar-v1", help="its grasp flags decide what counts as a carry or holding")
    ap.add_argument("--settle_steps", type=int, default=5, help="save this many steps into a carry")
    ap.add_argument("--confirm_steps", type=int, default=10, help="keep it only if the carry is unbroken this many steps later")
    ap.add_argument("--at", choices=("carry", "holding"), default="carry",
                    help="carry: picker's jaws off the bar (B008, B014). holding: receiver grips, picker no longer squeezing, "
                         "its jaws possibly still on the bar; saved when that has held settle_steps + 1 steps in a row, "
                         "the same rule as chain_eval.py --switch_on holding, with no check on what happens after")
    args = ap.parse_args()
    dev = torch.device("cuda")
    env = gym.make(args.env_id, num_envs=args.num_envs, obs_mode="rgb+segmentation", sim_backend="gpu",
                   sensor_configs=dict(width=128, height=128), domain_randomization=True,
                   domain_randomization_config={"background_level_range": [0, 50], "background_tint": 6})
    keep = KeepSensors(env, rgb=True, depth=False, state=True)
    env = FlattenActionSpaceWrapper(keep)
    u = env.unwrapped
    n = args.num_envs
    saved = dict(env_index=[], picker=[], actors={}, articulations={}, targets=[{}, {}], first={c: {"rgb": [], "segmentation": []} for c in CAMS})

    episode = 0
    for ck in args.checkpoints:
        obs, _ = env.reset(seed=args.seed)
        agent = DeployAgent(env, sample_obs=obs, device=dev)
        agent.load_checkpoint(ck)
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            episode += 1
            first = {c: {k: v.clone() for k, v in keep.raw[c].items()} for c in CAMS}
            # A start is saved `settle` steps into a carry (a grip that has held, not the barely-held instant of
            # release), and kept only if the carry is still unbroken `confirm` steps after that. Restoring cannot
            # carry over PhysX's contact cache, and barely-held grips were the ones it shook loose (B008 preflight)
            start_t = torch.full((n,), -1, dtype=torch.long, device=dev)
            alive = torch.zeros(n, dtype=torch.bool, device=dev)
            pending = torch.zeros(n, dtype=torch.bool, device=dev)
            done = torch.zeros(n, dtype=torch.bool, device=dev)
            buf = None
            run = torch.zeros(n, dtype=torch.long, device=dev)
            for t in range(u.spec.max_episode_steps or 120):
                obs, r, term, trunc, info = env.step(agent.get_action(obs))
                if args.at == "holding":
                    # The current run of the holding condition, reset when it breaks (as chain_eval.py's switch)
                    cond = (info["handed_over"] & info["receiver_grasp"] & ~info["picker_grasp"]).to(dev)
                    run = torch.where(cond, run + 1, torch.zeros_like(run))
                    snap = (run == args.settle_steps + 1) & ~done & ~pending
                    start_t[snap] = t - args.settle_steps
                    alive = cond
                else:
                    carry = (info["handed_over"] & info["receiver_grasp"] & ~info["picker_touch_bar"]).to(dev)
                    began = carry & (start_t < 0)
                    start_t[began] = t
                    alive = (alive | began) & carry
                    snap = alive & (t == start_t + args.settle_steps) & ~done
                if snap.any():
                    state = u.scene.get_sim_state()
                    if buf is None:
                        buf = {"actors": {k: torch.zeros_like(v) for k, v in state["actors"].items()},
                               "articulations": {k: torch.zeros_like(v) for k, v in state["articulations"].items()},
                               "targets": [{uid: torch.zeros_like(c._target_qpos) for uid, c in controllers_of(arm)
                                            if hasattr(c, "_target_qpos")} for arm in u.arms]}
                    for group in ("actors", "articulations"):
                        for name, v in state[group].items():
                            buf[group][name][snap] = v[snap]
                    for a, arm in enumerate(u.arms):
                        for uid, ctrl in controllers_of(arm):
                            if hasattr(ctrl, "_target_qpos"):
                                buf["targets"][a][uid][snap] = ctrl._target_qpos[snap]
                    pending |= snap
                pending &= alive
                confirm = pending & (t == start_t + args.settle_steps + args.confirm_steps)
                if confirm.any():
                    idx = torch.nonzero(confirm)[:, 0]
                    for group in ("actors", "articulations"):
                        for name, v in buf[group].items():
                            saved[group].setdefault(name, []).append(v[idx].clone())
                    for a in range(len(u.arms)):
                        for uid, v in buf["targets"][a].items():
                            saved["targets"][a].setdefault(uid, []).append(v[idx].clone())
                    for c in CAMS:
                        for k in ("rgb", "segmentation"):
                            saved["first"][c][k].append(first[c][k][idx].cpu())
                    if u._background_colors is not None:
                        saved.setdefault("background", []).append(u._background_colors[idx].clone())
                    saved.setdefault("gripper_stiffness", []).append(u._gripper_stiffness[:, idx].T.clone())
                    saved.setdefault("gripper_damping", []).append(u._gripper_damping[:, idx].T.clone())
                    saved["env_index"].append(idx.clone())
                    saved["picker"].append(u.picker[idx].clone())
                    done |= confirm
                    pending &= ~confirm
            print(f"{ck} episode {ep + 1}/{args.episodes}: carry starts {int(done.sum())} of {n}", flush=True)

    out = dict(
        env_index=torch.cat(saved["env_index"]).cpu(),
        picker=torch.cat(saved["picker"]).cpu(),
        actors={k: torch.cat(v).cpu() for k, v in saved["actors"].items()},
        articulations={k: torch.cat(v).cpu() for k, v in saved["articulations"].items()},
        targets=[{k: torch.cat(v).cpu() for k, v in t.items()} for t in saved["targets"]],
        first={c: {k: torch.cat(v) for k, v in d.items()} for c, d in saved["first"].items()},
        geometry={k: getattr(u, k).cpu().clone() for k in CarryStartMixin.GEOMETRY_KEYS},
        **{k: torch.cat(saved[k]).cpu() for k in ("background", "gripper_stiffness", "gripper_damping") if k in saved},
        meta=dict(checkpoints=args.checkpoints, episodes_per_checkpoint=args.episodes, num_envs=n, seed=args.seed,
                  settle_steps=args.settle_steps, confirm_steps=args.confirm_steps, at=args.at),
    )
    per_env = torch.bincount(out["env_index"], minlength=n)
    print(f"saved {len(out['env_index'])} carry starts; env indices with none: {int((per_env == 0).sum())} of {n}; "
          f"median per index {float(per_env.float().median()):.0f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)


if __name__ == "__main__":
    main()
