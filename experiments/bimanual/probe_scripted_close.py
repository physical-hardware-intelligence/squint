"""Is hovering a choice? No learning (B023 step 0). A handover checkpoint runs its own episodes; the first time an episode
hovers (the picker holds the bar at the handover point, the receiver's grip point within 3 cm of its end of the bar, its
gripper at least 15% open, no both-hold, for 5 steps in a row), that episode's receiver is taken over: arm held still,
gripper closing at the full rate, for 10 steps, while the picker keeps the policy's actions. Counts per outcome: a real
grasp (both hold under the corrected check), shut beside the bar, shut on air, other. High real grasps: the hover is in
place and the policy chooses not to close; low: the hover is in the wrong place. Also, at each takeover, where the
receiver's grip point is (gap to its target, distance from the bar's centre line, depth in from the bar's end, the
jaws' closing direction against the bar) and how open its gripper is, by outcome. Settings as stage_counts.py.

    python experiments/bimanual/probe_scripted_close.py --checkpoint runs/b021_s1/ckpt_6M.pt --seed 7
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

NEAR, OPEN, HOVER_STEPS, CLOSE_STEPS, SHUT = 0.03, 0.15, 5, 10, 0.03


def quartiles(x):
    x = x[~torch.isnan(x)].float().cpu()
    if not len(x):
        return None
    q = torch.quantile(x, torch.tensor([0.25, 0.5, 0.75]))
    return f"{q[1]:.2f} [{q[0]:.2f}-{q[2]:.2f}] n={len(x)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--env_id", default="SO101HandoverBarV6Grip-v1")
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
    assert env.action_space.shape[-1] == 12  # left arm's 6 (5 joints, gripper), then the right arm's 6
    n = args.num_envs
    receiver = (1 - u.picker.to(dev).long())
    run = torch.zeros(n, dtype=torch.long, device=dev)
    taken = torch.full((n,), -1, dtype=torch.long, device=dev)  # step the takeover began
    real = torch.zeros(n, dtype=torch.bool, device=dev)
    shut_touch = torch.zeros(n, dtype=torch.bool, device=dev)
    shut_air = torch.zeros(n, dtype=torch.bool, device=dev)
    ever_hold = torch.zeros(n, dtype=torch.bool, device=dev)  # both-hold before any takeover (not a hover episode)
    feats = {k: torch.full((n,), float("nan"), device=dev) for k in ("gap_cm", "off_axis_cm", "depth_cm", "angle_deg", "openness")}
    for t in range(u.spec.max_episode_steps or 120):
        a = agent.get_action(obs).clone()
        active = (taken >= 0) & (t - taken < CLOSE_STEPS)
        for side in (0, 1):
            m = active & (receiver == side)
            a[m, side * 6:side * 6 + 5] = 0.0  # arm held still
            a[m, side * 6 + 5] = -1.0  # gripper closing at the full rate
        obs, r, te, tr, info = env.step(a)
        gap, off_axis, depth = (x.to(dev) for x in u._receiver_grip_geometry())
        left, right = u.arms
        close_dir = u._by_role(left.finger1_link.pose.to_transformation_matrix()[..., :3, 1],
                               right.finger1_link.pose.to_transformation_matrix()[..., :3, 1], "receiver")
        angle = torch.rad2deg(torch.acos((close_dir * u._bar_axis()).sum(1).abs().clamp(max=1))).to(dev)
        opn = u._by_role(u._openness(u.arms[0]), u._openness(u.arms[1]), "receiver").to(dev)
        bar_at = torch.linalg.norm(u.bar.pose.p - u.handover_point, dim=1).to(dev) < 0.05
        hold = info["both_hold"].to(dev)
        ever_hold |= hold & (taken < 0)
        hovering = info["picker_grasp"].to(dev) & bar_at & (gap < NEAR) & (opn >= OPEN) & ~hold
        run = torch.where(hovering, run + 1, torch.zeros_like(run))
        start = (run >= HOVER_STEPS) & (taken < 0) & ~ever_hold
        taken = torch.where(start, torch.full_like(taken, t + 1), taken)
        for k, v in (("gap_cm", gap * 100), ("off_axis_cm", off_axis * 100), ("depth_cm", depth * 100), ("angle_deg", angle), ("openness", opn)):
            feats[k] = torch.where(start, v, feats[k])
        in_window = (taken >= 0) & (t >= taken) & (t - taken < CLOSE_STEPS)
        real |= in_window & hold
        shut = in_window & (opn < SHUT)
        shut_touch |= shut & info["receiver_touch_bar"].to(dev)
        shut_air |= shut & ~info["receiver_touch_bar"].to(dev)
    took = taken >= 0
    done = took & (taken + CLOSE_STEPS <= (u.spec.max_episode_steps or 120))
    out = dict(checkpoint=f"runs/{Path(args.checkpoint).parent.name}/{Path(args.checkpoint).name}", env_id=args.env_id,
               seed=args.seed, episodes=n, hover_takeovers=int(done.sum()),
               real_grasp=int((done & real).sum()), shut_beside=int((done & ~real & shut_touch).sum()),
               shut_on_air=int((done & ~real & ~shut_touch & shut_air).sum()),
               other=int((done & ~real & ~shut_touch & ~shut_air).sum()),
               both_hold_before_any_hover=int(ever_hold.sum()))
    outcomes = dict(real_grasp=done & real, shut_on_air=done & ~real & ~shut_touch & shut_air, shut_beside=done & ~real & shut_touch)
    out["at_takeover"] = {name: {k: quartiles(torch.where(m, v, float("nan"))) for k, v in feats.items()}
                          for name, m in outcomes.items() if m.any()}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
