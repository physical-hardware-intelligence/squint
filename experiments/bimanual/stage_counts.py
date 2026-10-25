"""How far does a handover checkpoint get? Runs it on N simulated episodes and counts, per episode, every stage reached.

    python experiments/bimanual/stage_counts.py --checkpoint runs/<run>/ckpt.pt [--num_envs 128] [--seed 7]
Domain randomization and backgrounds as training (0-50 +-6), GPU simulator. Unlike training's evaluation: no colour
jitter, and the scene is not rebuilt between episodes (fixed object sizes per env index).
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
# Reward bands of the current envs/handover.py (B003: stage 5 from 8, stage 6 from 10, success 16). Penalties can
# push an episode into a lower band, so the flags above and `carrying` below are the authoritative counts.
BANDS = ((0, 2, "reaching"), (2, 4, "picker holds, carrying"), (4, 6, "at handover point, receiver reaching"),
         (6, 8, "both hold"), (8, 10, "receiver carries to bin"), (10, 16.01, "in bin"))


def bin_points_in_view(u, width=128, height=128):
    """Per arm: how many of 5 bin points (centre, 4 rim corners at the bin origin's height) project inside that arm's
    wrist image. Geometry only, occlusion ignored, so it is an upper bound on what the camera can see. (left, right)."""
    hx, hy = u.bin_half_sizes_x, u.bin_half_sizes_y
    zero = torch.zeros_like(hx)
    local = torch.stack([torch.stack([zero, zero]), torch.stack([hx, hy]), torch.stack([hx, -hy]),
                         torch.stack([-hx, hy]), torch.stack([-hx, -hy])])  # (5, 2, n)
    local = torch.cat([local.permute(2, 0, 1), torch.zeros(len(hx), 5, 1, device=hx.device),
                       torch.ones(len(hx), 5, 1, device=hx.device)], dim=2)  # (n, 5, 4)
    world = local @ u.bin.pose.to_transformation_matrix().transpose(1, 2)  # (n, 5, 4)
    counts = []
    for arm in ("left", "right"):
        cam = u.scene.sensors[f"{arm}_wrist"].camera
        xc = world @ cam.get_extrinsic_matrix().transpose(1, 2)  # (n, 5, 3), OpenCV camera frame
        uv = xc @ cam.get_intrinsic_matrix().transpose(1, 2)
        z = uv[..., 2]
        px, py = uv[..., 0] / z.clamp(min=1e-6), uv[..., 1] / z.clamp(min=1e-6)
        counts.append(((z > 0) & (px >= 0) & (px < width) & (py >= 0) & (py < height)).sum(dim=1))
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--env_id", default="SO101HandoverBar-v1")
    ap.add_argument("--num_envs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(args.env_id, num_envs=args.num_envs, obs_mode="rgb+segmentation", sim_backend="gpu",
                   sensor_configs=dict(width=128, height=128), domain_randomization=True,
                   domain_randomization_config={"background_level_range": [0, 50], "background_tint": 6})
    env = FlattenActionSpaceWrapper(FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True))
    obs, info = env.reset(seed=args.seed)
    agent = DeployAgent(env, sample_obs=obs, device=dev)
    agent.load_checkpoint(args.checkpoint)
    u = env.unwrapped
    n, steps = args.num_envs, u.spec.max_episode_steps or 120
    ever = {k: torch.zeros(n, dtype=torch.bool, device=dev) for k in FLAGS + ("carrying",)}
    best = torch.full((n,), -1e9, device=dev)
    min_recv_gap = torch.full((n,), 1e9, device=dev)  # receiver TCP to the bar centre, closest approach (m)
    # Bin in the receiver's wrist view (any of the 5 bin points projected inside the image), by phase
    view = {k: torch.zeros(n, device=dev) for k in ("start", "both_hold_steps", "both_hold_seen", "carry_steps", "carry_seen")}
    # Bin in the receiver's view before the first action (the observation the policy starts from)
    left0, right0 = bin_points_in_view(u)
    view["start"] = (u._by_role(left0, right0, "receiver").to(dev) > 0).float()
    # Bar to bin centre, horizontal (m): at the first carrying step and the closest during the carry
    carry_start = torch.full((n,), float("nan"), device=dev)
    carry_min = torch.full((n,), float("nan"), device=dev)
    for _ in range(steps):
        obs, r, term, trunc, info = env.step(agent.get_action(obs))
        for k in FLAGS:
            ever[k] |= info[k].to(dev)
        # Stage 5's condition (handover.py): latched handover, receiver holds, picker no longer touches the bar
        ever["carrying"] |= (info["handed_over"] & info["receiver_grasp"] & ~info["picker_touch_bar"]).to(dev)
        best = torch.maximum(best, u.compute_dense_reward(None, None, info).to(dev))
        left, right = bin_points_in_view(u)
        centre_seen = u._by_role(left, right, "receiver").to(dev) > 0
        hold = info["both_hold"].to(dev)
        carry = (info["handed_over"] & info["receiver_grasp"] & ~info["picker_touch_bar"]).to(dev)
        view["both_hold_steps"] += hold.float()
        view["both_hold_seen"] += (hold & centre_seen).float()
        view["carry_steps"] += carry.float()
        d_xy = torch.linalg.norm((u.bar.pose.p - u.bin.pose.p)[:, :2], dim=1).to(dev)
        carry_start = torch.where(carry & torch.isnan(carry_start), d_xy, carry_start)
        carry_min = torch.where(carry, torch.fmin(carry_min, d_xy), carry_min)
        view["carry_seen"] += (carry & centre_seen).float()
        tcp_r = u._by_role(u.arms[0].tcp_pos, u.arms[1].tcp_pos, "receiver")
        min_recv_gap = torch.minimum(min_recv_gap, torch.linalg.norm(tcp_r - u.bar.pose.p, dim=1).to(dev))
    out = dict(checkpoint=args.checkpoint, episodes=n, seed=args.seed,
               reached={k: int(v.sum()) for k, v in ever.items()},
               best_stage={name: int(((best >= lo) & (best < hi)).sum()) for lo, hi, name in BANDS},
               receiver_closest_cm=dict(median=round(float(min_recv_gap.median()) * 100, 1),
                                        under_3cm=int((min_recv_gap < 0.03).sum()), under_6cm=int((min_recv_gap < 0.06).sum())),
               carry_bar_to_bin_cm=dict(
                   episodes=int((~torch.isnan(carry_start)).sum()),
                   start_median=round(float(carry_start.nanmedian()) * 100, 1) if (~torch.isnan(carry_start)).any() else None,
                   closest_median=round(float(carry_min.nanmedian()) * 100, 1) if (~torch.isnan(carry_min)).any() else None,
                   closest_under_10cm=int((carry_min < 0.10).sum()), closest_under_5cm=int((carry_min < 0.05).sum())),
               bin_in_receiver_view=dict(
                   at_start_episodes=int(view["start"].sum()),
                   both_hold_steps=int(view["both_hold_steps"].sum()), of_which_bin_seen=int(view["both_hold_seen"].sum()),
                   carry_steps=int(view["carry_steps"].sum()), of_which_bin_seen_carry=int(view["carry_seen"].sum())))
    print(json.dumps(out, indent=1))
    path = Path(__file__).resolve().parent / "out" / f"stages_{Path(args.checkpoint).parent.name}_{dt.datetime.now():%Y%m%d-%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
