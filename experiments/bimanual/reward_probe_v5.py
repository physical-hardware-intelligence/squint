"""B020's checks on reward v5, before any training (B016's and B012's lessons). The current policy (B011 seed 1 at 3M) runs
its own episodes under the corrected grasp check; at every step the same state is scored under v2 and under v5. "Reach
steps" are the steps v5 can change: the picker holds the bar at the handover point and both do not hold it.

  --design   distributions used to set OPEN_APPROACH_FULL (seed 9, not a check seed)
  (default)  the checks; exit 1 if any fails (seeds 7 and 8):
    C1 v5 never pays more than v2, and pays the same off reach steps and on reach steps with the receiver's gripper open
       (at least OPEN_APPROACH_FULL)
    C2 real handovers (class 1): median over episodes of the mean v2 - v5 on their reach steps at most 0.1, and on the 3
       steps before their first both-hold v2 - v5 at most 0.1 in at least 90% of them (no dip at the grip)
    C3 shut beside the bar (class 5): the same median at least 0.5
    C4 v2 - v5 at most 2 * OPEN_APPROACH_WEIGHT everywhere: the reach band never falls below stage 2's top

    python experiments/bimanual/reward_probe_v5.py --checkpoint runs/b011_wrist_s1/ckpt_3M.pt --seed 7 [--design]
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
from experiments.bimanual.probe_receiver_grasp import CLASSES, GraspClasses  # noqa: E402
from train_squint import DeployAgent  # noqa: E402

EPS = 1e-4


def pct(x, qs):
    x = x[~torch.isnan(x)].float().cpu()
    if not len(x):
        return None
    return {f"p{int(q * 100)}": round(float(torch.quantile(x, q)), 4) for q in qs} | {"n": len(x)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--env_id", default="SO101HandoverBarV5Grip-v1")
    ap.add_argument("--num_envs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--design", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda")
    env = gym.make(args.env_id, num_envs=args.num_envs, obs_mode="rgb+segmentation", sim_backend="gpu",
                   sensor_configs=dict(width=128, height=128), domain_randomization=True,
                   domain_randomization_config={"background_level_range": [0, 50], "background_tint": 6})
    env = FlattenActionSpaceWrapper(FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True))
    obs, _ = env.reset(seed=args.seed)
    agent = DeployAgent(env, sample_obs=obs, device=dev)
    agent.load_checkpoint(args.checkpoint)
    u = env.unwrapped
    assert u.GRASP_NEEDS_OPEN_JAWS and u.REWARD_VERSION == 5, "run on the v5 env with the corrected grasp check"
    g = GraspClasses(u, dev)
    rec = {k: [] for k in ("r2", "r5", "reach", "open", "off_axis", "depth", "both_hold", "shut_touch")}
    for _ in range(u.spec.max_episode_steps or 120):
        obs, r, term, trunc, info = env.step(agent.get_action(obs))
        g.update(info)
        scores = {}
        for v in (2, 5):
            u.REWARD_VERSION = v
            scores[v] = u.compute_dense_reward(None, None, info).to(dev)
        u.REWARD_VERSION = 5
        _, off_axis, depth = (x.to(dev) for x in u._receiver_grip_geometry())
        opn = u._by_role(u._openness(u.arms[0]), u._openness(u.arms[1]), "receiver").to(dev)
        at_handover = info["picker_grasp"].to(dev) & (torch.linalg.norm(u.bar.pose.p - u.handover_point, dim=1).to(dev) < 0.05)
        both = info["both_hold"].to(dev)
        rec["r2"].append(scores[2]); rec["r5"].append(scores[5])
        rec["reach"].append(at_handover & ~both)
        rec["open"].append(opn); rec["off_axis"].append(off_axis); rec["depth"].append(depth)
        rec["both_hold"].append(both)
        rec["shut_touch"].append(info["receiver_touch_bar"].to(dev) & (opn < 0.03))
    R = {k: torch.stack(v) for k, v in rec.items()}  # (steps, envs)
    d = R["r2"] - R["r5"]
    cls = g.classes()
    real, beside = cls[0], cls[4]
    T, n = d.shape
    nan = torch.full_like(d, float("nan"))
    # The 3 steps before each episode's first both-hold
    first_hold = torch.where(R["both_hold"].any(0), R["both_hold"].float().argmax(0), torch.full((n,), -1, device=dev))
    t_idx = torch.arange(T, device=dev)[:, None]
    pre_grip = (first_hold[None] >= 0) & (t_idx < first_hold[None]) & (t_idx >= first_hold[None] - 3)
    out = dict(checkpoint=f"runs/{Path(args.checkpoint).parent.name}/{Path(args.checkpoint).name}", seed=args.seed,
               episodes=n, classes={name: int(c.sum()) for name, c in zip(CLASSES, cls)},
               constants=dict(OPEN_APPROACH_WEIGHT=u.OPEN_APPROACH_WEIGHT, OPEN_APPROACH_FULL=u.OPEN_APPROACH_FULL))

    if args.design:
        qs = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
        out["design"] = dict(
            real_reach_open=pct(torch.where(R["reach"] & real[None], R["open"], nan), qs),
            real_arrival_open=pct(torch.where(real, g.arrival_open, float("nan")), qs),
            real_pre_grip_off_axis_cm=pct(torch.where(pre_grip & real[None], R["off_axis"] * 100, nan), qs + (1.0,)),
            real_pre_grip_depth_cm=pct(torch.where(pre_grip & real[None], R["depth"] * 100, nan), (0.0, 0.05, 0.1, 0.5)),
            real_pre_grip_open=pct(torch.where(pre_grip & real[None], R["open"], nan), qs),
            beside_shut_touching_off_axis_cm=pct(torch.where(R["reach"] & R["shut_touch"] & beside[None], R["off_axis"] * 100, nan),
                                                 (0.0, 0.01, 0.05, 0.1, 0.25, 0.5)),
            any_shut_touching_off_axis_cm=pct(torch.where(R["shut_touch"], R["off_axis"] * 100, nan), (0.0, 0.01, 0.05, 0.1)),
        )
        print(json.dumps(out, indent=1))
        return

    def per_episode_mean(mask):
        m = mask.float()
        return torch.where(m.sum(0) > 0, (d * m).sum(0) / m.sum(0).clamp(min=1), torch.full((n,), float("nan"), device=dev))

    unchanged = ~R["reach"] | (R["open"] >= u.OPEN_APPROACH_FULL)
    reach_drop = per_episode_mean(R["reach"])
    pre_grip_max = torch.where(pre_grip, d, torch.full_like(d, -1e9)).max(0).values
    real_with_pre = real & (first_hold >= 3)
    checks = {
        "C1 v5 never above v2": float(d.min()) >= -EPS,
        "C1 v5 = v2 off reach steps and when open": float(d[unchanged].abs().max()) <= EPS,
        "C2 real: median reach-step drop <= 0.1": float(reach_drop[real].nanmedian()) <= 0.1,
        "C2 real: no dip at the grip in >= 90%": float((pre_grip_max[real_with_pre] <= 0.1).float().mean()) >= 0.9,
        "C3 beside: median reach-step drop >= 0.5": float(reach_drop[beside].nanmedian()) >= 0.5,
        "C4 drop <= 2 * weight": float(d.max()) <= 2 * u.OPEN_APPROACH_WEIGHT + EPS,
    }
    out["numbers"] = dict(
        min_v2_minus_v5=round(float(d.min()), 5), max_v2_minus_v5=round(float(d.max()), 4),
        max_abs_where_unchanged=round(float(d[unchanged].abs().max()), 6),
        reach_step_drop_median={name: (round(float(reach_drop[c].nanmedian()), 3) if c.any() else None) for name, c in zip(CLASSES, cls)},
        real_no_dip_at_grip=f"{int((pre_grip_max[real_with_pre] <= 0.1).sum())} of {int(real_with_pre.sum())}",
    )
    out["checks"] = {k: "PASS" if v else "FAIL" for k, v in checks.items()}
    print(json.dumps(out, indent=1))
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
