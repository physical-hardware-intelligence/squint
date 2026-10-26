"""Is the release physically possible from the saved starts? No learning: from each restored start the picker opens its
gripper (3 steps), then moves its arm back toward the episode's start pose (12 steps, at most 0.1 rad per joint per
step), while the receiver holds still. Counts, after the 15 steps: the receiver still gripping the lifted bar with the
picker off it (a clean release), the bar dropped (no grip, or not lifted), and the receiver gripping while the picker
still touches.

    SQUINT_CARRY_BANK=runs/carry_starts_holding_b011.pt python experiments/bimanual/probe_scripted_release.py
Run it on a clean-carry bank (B014's) as the control: there the picker is already off, so nearly all should hold.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper  # noqa: E402

import envs  # noqa: E402,F401

N, RESETS, OPEN_STEPS, BACK_STEPS = 128, 4, 3, 12
dev = torch.device("cuda")
env = gym.make(sys.argv[1] if len(sys.argv) > 1 else "SO101HandoverCarryTopCamLeHome-v2", num_envs=N, obs_mode="rgb+segmentation", sim_backend="gpu",
               sensor_configs=dict(width=128, height=128), domain_randomization=True,
               domain_randomization_config={"background_level_range": [0, 50], "background_tint": 6})
env = FlattenActionSpaceWrapper(FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True))
u = env.unwrapped
assert env.action_space.shape[-1] == 12  # left arm's 6 (5 joints, gripper), then the right arm's 6
start_q = u.rest_qpos.to(dev)  # the episode start pose (SO101 keyframe "start")
tot = dict(starts=0, clean=0, dropped=0, still_touching=0, other=0)
rows = {}
for ep in range(RESETS):
    obs, _ = env.reset(seed=1000 + ep)
    has = torch.tensor([len(r) > 0 for r in u._bank_rows], device=dev)
    picker = u.picker.to(dev).long()
    # Grip features at the start: where along the bar the receiver grips, off its centre line, the bar's tilt, the
    # receiver's gripper drive (randomized per episode) and how closed it is
    rside = 1 - picker
    tcp_r = torch.where(rside[:, None] == 0, u.arms[0].tcp_pos, u.arms[1].tcp_pos).to(dev)
    bar_p, axis = u.bar.pose.p.to(dev), u._bar_axis().to(dev)
    s_along = ((tcp_r - bar_p) * axis).sum(1)
    feats = dict(
        depth_cm=(u.bar_half_lengths.to(dev)[:N] - s_along.abs()) * 100,
        off_axis_cm=torch.linalg.norm((tcp_r - bar_p) - s_along[:, None] * axis, dim=1) * 100,
        tilt_deg=torch.rad2deg(torch.asin(axis[:, 2].abs().clamp(max=1))),
        stiffness=torch.stack([u._gripper_stiffness[int(rs), e] for e, rs in enumerate(rside.tolist())]).to(dev),
        closed=1 - torch.stack([(u.arms[int(rs)].robot.get_qpos()[e, -1] - u.arms[int(rs)].robot.get_qlimits()[e, -1, 0])
                                / (u.arms[int(rs)].robot.get_qlimits()[e, -1, 1] - u.arms[int(rs)].robot.get_qlimits()[e, -1, 0])
                                for e, rs in enumerate(rside.tolist())]).to(dev),
    )
    for t in range(OPEN_STEPS + BACK_STEPS):
        a = torch.zeros((N, 12), device=dev)
        for side in (0, 1):
            is_p = picker == side
            q = u.arms[side].robot.get_qpos().to(dev)
            arm = torch.zeros((N, 6), device=dev)
            arm[:, 5] = 1.0  # gripper: open at the full rate
            if t >= OPEN_STEPS:
                arm[:, :5] = torch.clamp((start_q[:5] - q[:, :5]) / 0.1, -1, 1)
            a[is_p, side * 6:(side + 1) * 6] = arm[is_p]
        obs, r, te, tr, info = env.step(a)
    rg, lifted, ptouch = (info[k].to(dev) for k in ("receiver_grasp", "bar_lifted", "picker_touch_bar"))
    clean = has & rg & lifted & ~ptouch
    dropped = has & (~rg | ~lifted)
    touching = has & rg & lifted & ptouch
    tot["starts"] += int(has.sum()); tot["clean"] += int(clean.sum()); tot["dropped"] += int(dropped.sum())
    tot["still_touching"] += int(touching.sum())
    for k, v in feats.items():
        rows.setdefault(k, {"clean": [], "dropped": []})
        rows[k]["clean"].append(v[clean].cpu()); rows[k]["dropped"].append(v[dropped].cpu())
n = max(tot["starts"], 1)
print(f"scripted release from {tot['starts']} restored starts: clean {tot['clean']} ({100 * tot['clean'] / n:.0f}%), "
      f"dropped {tot['dropped']} ({100 * tot['dropped'] / n:.0f}%), receiver grips but picker still touching "
      f"{tot['still_touching']} ({100 * tot['still_touching'] / n:.0f}%)")
print("grip at the start, by outcome: median [25th-75th percentile]")
for k, d in rows.items():
    parts = []
    for o in ("clean", "dropped"):
        x = torch.cat(d[o]).float()
        if len(x):
            q = torch.quantile(x, torch.tensor([0.25, 0.5, 0.75]))
            parts.append(f"{o} {q[1]:.2f} [{q[0]:.2f}-{q[2]:.2f}]")
    print(f"  {k:12s} " + " | ".join(parts))
