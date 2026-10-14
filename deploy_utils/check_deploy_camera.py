"""Phi 2026-10-13: does deploy's camera path (lerobot OpenCVCamera from robot_config.py) give the flipped wrist view,
and do the uvcc-locked settings survive lerobot opening the camera? Camera only: the servo bus is never opened."""
import os, sys, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from deploy_utils.robot_config import create_real_robot

import argparse
_ap = argparse.ArgumentParser(description="deploy's camera path vs a direct capture; optional uvcc settings check")
_ap.add_argument("--uvcc", default=None, help="path to the uvcc binary, to check camera settings before/after")
_ap.add_argument("--vendor", default="10643"); _ap.add_argument("--product", default="2136")
_A = _ap.parse_args()
UVCC = _A.uvcc
def uvcc_export():
    if not UVCC: return None
    r = subprocess.run([UVCC, "--vendor", _A.vendor, "--product", _A.product, "export"], capture_output=True, text=True, timeout=10)
    return json.loads(r.stdout) if r.returncode == 0 else r.stderr.strip()

before = uvcc_export()
cfg = create_real_robot().config.cameras["base_camera"]     # the exact config deploy.py uses; robot not connected
cam = OpenCVCamera(cfg); cam.connect()
frames = [cam.async_read() for _ in range(15)]
cam.disconnect()
rgb = frames[-1]
print("lerobot frame:", rgb.shape, rgb.dtype, "(RGB) | rotation", cfg.rotation)
# reference: the tuner's path (direct AVFoundation + cv2.rotate 180), BGR
cap = cv2.VideoCapture(cfg.index_or_path, cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY); cap.set(3, 640); cap.set(4, 480)
for _ in range(10): cap.read()
ok, bgr = cap.read(); cap.release()
ref = cv2.rotate(bgr, cv2.ROTATE_180)
diff = np.abs(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(int) - ref.astype(int)).mean()
flip_diff = np.abs(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(int) - bgr.astype(int)).mean()
print(f"mean |lerobot - tuner(flipped)| = {diff:.1f}   vs  unflipped = {flip_diff:.1f}  (lower = same orientation)")
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tune_snapshots", "deploy_camera_check.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
cv2.imwrite(out, np.hstack([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), ref]))
after = uvcc_export()
if isinstance(before, dict) and isinstance(after, dict):
    changed = {k: (before[k], after.get(k)) for k in before if before[k] != after.get(k)}
    print("uvcc settings changed by lerobot opening the camera:", changed or "none")
else:
    print("uvcc:", before, after)
print("saved", out)
