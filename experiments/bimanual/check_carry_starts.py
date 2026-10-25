"""Do B008's restored episodes start as carries? Checks one carry env against its bank.

    SQUINT_CARRY_BANK=runs/carry_starts.pt python experiments/bimanual/check_carry_starts.py SO101HandoverCarry-v1 [--num_envs 64]
For the env indices that have saved states: at reset the receiver holds the bar alone (latched handover); holding still
for 10 steps keeps it held and the bar within 1 cm; the reset frame equals a fresh render; the first-frame variant shows
the saved opening images; the top-camera variant has 9 channels; the bin variant has the bin's 4 numbers in its state.
Rates are over the envs that have a saved start only (envs without one are not counted as passing).
GPU simulation. Writes the top camera's first frame to experiments/bimanual/out/top_camera_view.png for the top variant.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper  # noqa: E402

import envs  # noqa: E402,F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env_id")
    ap.add_argument("--num_envs", type=int, default=64)
    args = ap.parse_args()
    base = gym.make(args.env_id, num_envs=args.num_envs, obs_mode="rgb+segmentation", sim_backend="gpu",
                    sensor_configs=dict(width=128, height=128), domain_randomization=True,
                    domain_randomization_config={"background_level_range": [0, 50], "background_tint": 6})
    flat = FlattenRGBDObservationWrapper(base, rgb=True, depth=False, state=True)
    env = FlattenActionSpaceWrapper(flat)
    u = env.unwrapped
    ok = []

    def check(name, cond, detail=""):
        ok.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))

    obs, _ = env.reset(seed=1)
    has = torch.tensor([len(r) > 0 for r in u._bank_rows], device=u.device)
    rows = u._restored_rows.clone()
    saved_bar = u._bank["actors"]["bar"][rows.cpu()][:, :7].to(u.device)
    err = (u.bar.pose.raw_pose - saved_bar).abs().max(dim=1).values
    check("restored bar pose = saved pose", bool((err[has] < 1e-4).all()), f"max error {float(err[has].max()):.1e}")
    check("reset frame = fresh render", torch.equal(obs["rgb"], flat.observation(u.get_obs())["rgb"]))
    bank = u._bank
    if "gripper_stiffness" in bank:
        want = bank["gripper_stiffness"][rows.cpu()].T.to(u.device)
        check("restored gripper stiffness = saved", torch.allclose(u._gripper_stiffness[:, has], want[:, has]))
    if "background" in bank and u._background_colors is not None:
        want = bank["background"][rows.cpu()].to(u.device)
        check("restored background colour = saved", torch.allclose(u._background_colors[has], want[has]))
    # Grasp flags read contact forces, which exist only after a physics step: judge the carry after one still step
    zero = torch.zeros((args.num_envs, 12), device=u.device)
    obs, r, term, trunc, info = env.step(zero)
    carry = info["handed_over"] & info["receiver_grasp"] & ~info["picker_touch_bar"]
    check("restored envs are carries after one still step (receiver holds alone, handover latched)",
          bool(carry[has].float().mean() >= 0.95), f"{int((carry & has).sum())} of {int(has.sum())}")
    tcp_r = lambda: u._by_role(u.arms[0].tcp_pos, u.arms[1].tcp_pos, "receiver")
    rel0 = u.bar.pose.p - tcp_r()
    for _ in range(10):
        obs, r, term, trunc, info = env.step(zero)
    carry = info["handed_over"] & info["receiver_grasp"] & ~info["picker_touch_bar"]
    slip = torch.linalg.norm((u.bar.pose.p - tcp_r()) - rel0, dim=1)
    check("holding still for 10 more steps keeps the carry", bool(carry[has].float().mean() >= 0.9),
          f"{int((carry & has).sum())} of {int(has.sum())}")
    check("bar stays within 1 cm of the receiver's gripper (no slip)", bool((slip[has] < 0.01).float().mean() >= 0.9),
          f"median slip {float(slip[has].median()) * 100:.2f} cm, max {float(slip[has].max()) * 100:.2f} cm")
    c = obs["rgb"].shape[-1]
    if "FirstFrame" in args.env_id:
        bank = u._bank["first"]
        saved = torch.cat([bank[cam]["rgb"][rows.cpu()] for cam in ("left_wrist", "right_wrist")], dim=-1).to(obs["rgb"].device)
        check("first-frame channels = the saved opening images", torch.equal(obs["rgb"][has][..., 6:12], saved[has]))
        check("12 image channels", c == 12, f"{c}")
    elif "TopCam" in args.env_id:
        check("9 image channels (left, right, top)", c == 9, f"{c}")
        img = obs["rgb"][0, ..., 6:9].cpu().numpy().astype(np.uint8)
        out = Path(__file__).resolve().parent / "out" / "top_camera_view.png"
        from PIL import Image
        Image.fromarray(img).resize((384, 384), Image.NEAREST).save(out)
        print(f"top camera view saved: {out}")
    elif "BinInState" in args.env_id:
        check("28 state numbers", obs["state"].shape[-1] == 28, f"{obs['state'].shape[-1]}")
    else:
        check("6 image channels", c == 6, f"{c}")
    print(f"{args.env_id}: {sum(ok)} of {len(ok)} passed; envs without a saved start: {int((~has).sum())}")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
