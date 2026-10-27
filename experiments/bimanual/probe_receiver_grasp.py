"""Why does the receiver miss the grasp? No learning (B019 amendment A2): a handover checkpoint runs its own episodes under
the corrected grasp check, and each episode goes in the first class that fits:
  1 real handover; 2 the bar never reaches the handover point; 3 the receiver's grip point never comes within 3 cm of its
  end of the bar; 4 it does, but the receiver's gripper never closes below 15% open there; 5 it closes there and shuts
  (below 3% within 5 steps) touching the bar: beside it; 6 shuts not touching it: on air; 7 anything else.
At the receiver's first close near the bar: the grip point's distance from the bar's centre line, its depth in from the
bar's end, the angle between the jaws' closing direction and the bar (90 = across it); also whether the gripper was already
closed when it first came within 3 cm, and the closes per episode. Settings as stage_counts.py.

    python experiments/bimanual/probe_receiver_grasp.py --checkpoint runs/b011_wrist_s1/ckpt_3M.pt --seed 7
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

NEAR, CLOSED, SHUT, SHUT_WITHIN = 0.03, 0.15, 0.03, 5  # m, openness, openness, steps
CLASSES = ("1 real handover", "2 bar never at the handover point", "3 receiver never within 3 cm",
           "4 near but never closes", "5 shuts beside the bar", "6 shuts on air", "7 other")


def quartiles(x):
    x = x[~torch.isnan(x)].float().cpu()
    if not len(x):
        return None
    q = torch.quantile(x, torch.tensor([0.25, 0.5, 0.75]))
    return f"{q[1]:.2f} [{q[0]:.2f}-{q[2]:.2f}] n={len(x)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--env_id", default="SO101HandoverBarV2Grip-v1")
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
    u = env.unwrapped
    assert u.GRASP_NEEDS_OPEN_JAWS, "run under the corrected grasp check"
    n = args.num_envs
    left, right = u.arms
    nan = lambda: torch.full((n,), float("nan"), device=dev)
    no = lambda: torch.zeros(n, dtype=torch.bool, device=dev)
    real, reached, near, shut_touch, shut_air, prev_closed = no(), no(), no(), no(), no(), no()
    arrival_open = nan()  # receiver openness the first time its grip point is within 3 cm
    close_t = torch.full((n,), -1, dtype=torch.long, device=dev)
    closes = torch.zeros(n, device=dev)
    feats = {k: nan() for k in ("off_axis_cm", "depth_cm", "angle_deg", "openness")}
    for t in range(u.spec.max_episode_steps or 120):
        obs, r, term, trunc, info = env.step(agent.get_action(obs))
        bar_p, axis = u.bar.pose.p, u._bar_axis()
        # The receiver's target as in the reward: a point on the centre line 1.2 cm in from the end away from the picker
        grip_offset = (u.bar_half_lengths - 0.012)[:, None]
        end_a, end_b = bar_p + axis * grip_offset, bar_p - axis * grip_offset
        tcp_p = u._by_role(left.tcp_pos, right.tcp_pos, "picker")
        tcp_r = u._by_role(left.tcp_pos, right.tcp_pos, "receiver")
        a_nearer = torch.linalg.norm(tcp_p - end_a, dim=1) <= torch.linalg.norm(tcp_p - end_b, dim=1)
        gap = torch.linalg.norm(tcp_r - torch.where(a_nearer[:, None], end_b, end_a), dim=1)
        rel = tcp_r - bar_p
        s = (rel * axis).sum(1)
        off_axis = torch.linalg.norm(rel - s[:, None] * axis, dim=1)
        depth = u.bar_half_lengths - s.abs()
        # The jaws' closing direction, as is_grasping takes it (the fixed jaw's y axis)
        close_dir = u._by_role(left.finger1_link.pose.to_transformation_matrix()[..., :3, 1],
                               right.finger1_link.pose.to_transformation_matrix()[..., :3, 1], "receiver")
        angle = torch.rad2deg(torch.acos((close_dir * axis).sum(1).abs().clamp(max=1)))
        opn = u._by_role(u._openness(left), u._openness(right), "receiver")

        is_near = info["reached_handover"] & (gap < NEAR)
        real |= info["both_hold"]
        reached |= info["reached_handover"]
        arrival_open = torch.where(is_near & ~near, opn, arrival_open)
        near |= is_near
        closed = is_near & (opn < CLOSED)
        closes += (closed & ~prev_closed).float()
        prev_closed = closed
        first = closed & (close_t < 0)
        close_t = torch.where(first, torch.full_like(close_t, t), close_t)
        for k, v in (("off_axis_cm", off_axis * 100), ("depth_cm", depth * 100), ("angle_deg", angle), ("openness", opn)):
            feats[k] = torch.where(first, v, feats[k])
        shut = (close_t >= 0) & (t - close_t <= SHUT_WITHIN) & (opn < SHUT)
        shut_touch |= shut & info["receiver_touch_bar"]
        shut_air |= shut & ~info["receiver_touch_bar"]

    closed_once = close_t >= 0
    rest = ~real
    cls = [real, rest & ~reached, rest & reached & ~near, rest & near & ~closed_once,
           rest & closed_once & shut_touch, rest & closed_once & ~shut_touch & shut_air]
    cls.append(rest & ~torch.stack(cls[1:]).any(0))
    out = dict(checkpoint=f"runs/{Path(args.checkpoint).parent.name}/{Path(args.checkpoint).name}", env_id=args.env_id,
               seed=args.seed, episodes=n, classes={name: int(c.sum()) for name, c in zip(CLASSES, cls)},
               at_first_close_near_the_bar={}, arrived_already_closed={}, closes_per_episode={})
    for name, c in zip(CLASSES, cls):
        if not c.any():
            continue
        f = {k: quartiles(torch.where(c, v, float("nan"))) for k, v in feats.items()}
        if any(f.values()):
            out["at_first_close_near_the_bar"][name] = f
        arrived = c & ~torch.isnan(arrival_open)
        if arrived.any():
            out["arrived_already_closed"][name] = f"{int((arrived & (arrival_open < CLOSED)).sum())} of {int(arrived.sum())}"
            out["closes_per_episode"][name] = quartiles(torch.where(arrived, closes, float("nan")))
    print(json.dumps(out, indent=1))
    path = Path(__file__).resolve().parent / "out" / f"receiver_grasp_{Path(args.checkpoint).parent.name}_seed{args.seed}_{dt.datetime.now():%Y%m%d-%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
