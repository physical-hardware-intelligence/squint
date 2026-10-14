"""
How much does a camera-pose difference matter at the 16x16 the policy sees? (Phi, 2026-10-13)

Renders the sim wrist camera at 128x128 (training size) -> area-downsample to 16x16, the same chain as training,
for: Squint's default camera, our tuned camera, and random draws inside the training jitter window (+-2 mm, +-1 deg).
Grabs real frames through deploy's chain (centre crop -> resize 128 -> area 16, optional 180 flip).
No arm access: the joint angles come from a snapshot JSON (the arm must not have moved since).

Metrics, all on the 16x16 policy image:
  L1   mean absolute pixel difference (0-255) between two 16x16 RGB images
  IoU  overlap of the jaw masks (bright, unsaturated pixels), at 16x16 and at 128x128
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from deploy_utils.tune_camera_nomotion import NoMotionTuner, JOINTS


class SnapshotArm:
    """Stands in for ReadOnlyArm: fixed joint angles from a snapshot, never opens the serial port."""

    def __init__(self, snapshot):
        snap = json.loads(Path(snapshot).read_text())
        self.deg = {j: float(snap["qpos_deg"][j]) for j in JOINTS}
        self.servo = snap.get("gripper_servo_deg")
        self.torque_start = {j: 0 for j in JOINTS}
        self.calibration_name = "snapshot"

    def qpos_deg(self):
        return dict(self.deg), self.servo

    def close(self):
        return None


def to16(img128):
    return cv2.resize(img128, (16, 16), interpolation=cv2.INTER_AREA)   # = F.interpolate(mode='area'), factor 8


def jaw_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return (hsv[..., 2] > 70) & (hsv[..., 1] < 70)                      # bright + unsaturated: jaws, not the red cube


def iou(a, b):
    return float((a & b).sum() / max(1, (a | b).sum()))


def l1(a, b):
    return float(np.abs(a.astype(int) - b.astype(int)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--tuned", nargs=6, type=float, required=True, metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
                    help="metres and degrees, as printed by the tuner")
    ap.add_argument("--rotate-180", action="store_true")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--jitter-draws", type=int, default=50)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "tune_snapshots" / "camera_fit"))
    args = ap.parse_args()

    t = NoMotionTuner.__new__(NoMotionTuner)
    if args.no_camera:
        NoMotionTuner._setup_real_robot = lambda self: setattr(self, "cap", type("C", (), {"release": lambda s: None})())
    NoMotionTuner.__init__(t, "SO101ReachCube-v1", 128, 128, SnapshotArm(args.snapshot), rotate_180=args.rotate_180,
                           headless=True)
    default = (t.cam_x, t.cam_y, t.cam_z, t.cam_roll, t.cam_pitch, t.cam_yaw)

    def render(pose):
        t.cam_x, t.cam_y, t.cam_z, t.cam_roll, t.cam_pitch, t.cam_yaw = pose
        t._update_camera()
        return t._get_sim_image()

    sim_def, sim_tun = render(default), render(tuple(args.tuned))
    rng = np.random.default_rng(0)
    jitter = []
    for _ in range(args.jitter_draws):
        d = rng.uniform(-1, 1, 6) * np.array([0.002, 0.002, 0.002, 1, 1, 1])
        jitter.append(render(tuple(np.array(default) + d)))
    render(default)

    res = {"default": dict(zip("x y z roll pitch yaw".split(), map(float, default))),
           "tuned": dict(zip("x y z roll pitch yaw".split(), args.tuned)),
           "sim_default_vs_tuned": {"L1_16": l1(to16(sim_def), to16(sim_tun)),
                                    "IoU_16": iou(jaw_mask(to16(sim_def)), jaw_mask(to16(sim_tun)))},
           "training_jitter_vs_default": {
               "L1_16_mean": float(np.mean([l1(to16(sim_def), to16(j)) for j in jitter])),
               "L1_16_max": float(np.max([l1(to16(sim_def), to16(j)) for j in jitter])),
               "IoU_16_min": float(np.min([iou(jaw_mask(to16(sim_def)), jaw_mask(to16(j))) for j in jitter]))}}

    if not args.no_camera:
        reals = [r for r in (t._get_real_image() for _ in range(args.frames)) if r is not None]
        real = np.median(np.stack(reals), axis=0).astype(np.uint8)            # median of frames: less sensor noise
        r16 = to16(real)
        noise = float(np.mean([l1(to16(a), to16(b)) for a, b in zip(reals[:-1], reals[1:])]))
        luma = lambda im: float(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[jaw_mask(im)].mean())
        res["jaw_luma_128"] = {"real": luma(real), "sim_tuned": luma(sim_tun), "sim_default": luma(sim_def)}
        res["jaw_luma_16"] = {"real": luma(r16), "sim_tuned": luma(to16(sim_tun))}
        res["real"] = {
            "frames": len(reals), "frame_to_frame_L1_16": noise,
            "vs_default": {"L1_16": l1(r16, to16(sim_def)), "IoU_16": iou(jaw_mask(r16), jaw_mask(to16(sim_def))),
                           "IoU_128": iou(jaw_mask(real), jaw_mask(sim_def))},
            "vs_tuned": {"L1_16": l1(r16, to16(sim_tun)), "IoU_16": iou(jaw_mask(r16), jaw_mask(to16(sim_tun))),
                         "IoU_128": iou(jaw_mask(real), jaw_mask(sim_tun))}}
        up = lambda im: cv2.resize(im, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask = lambda im: cv2.cvtColor(jaw_mask(im).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        row1 = np.hstack([up(r16), up(to16(sim_def)), up(to16(sim_tun))])
        row2 = np.hstack([up(mask(r16)), up(mask(to16(sim_def))), up(mask(to16(sim_tun)))])
        bar = np.full((28, row1.shape[1], 3), 40, np.uint8)
        for x, s in [(8, "REAL 16x16 (flip)"), (264, "SIM 16x16 default cam"), (520, "SIM 16x16 tuned cam")]:
            cv2.putText(bar, s, (x, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f"{args.out}.png", np.vstack([bar, row1, row2]))
    Path(f"{args.out}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    t._cleanup()


if __name__ == "__main__":
    main()
