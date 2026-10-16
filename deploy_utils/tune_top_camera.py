"""
Fit the sim's fixed top camera to the real one (Phi, 2026-10-15). Nothing moves: the arm is read with ReadOnlyArm
(tune_camera_nomotion.py, every servo write disabled) and the sim robot copies its joint angles every frame.

Window: Real | Sim | 50/50 blend. The real frame goes through the SAME capture (640x480) and centre-square crop as
deploy.py, so the fitted FOV is the one the policy will see. The sim shows only the robot (cube and bin hidden) on
black, like the policy's image. Slide until the sim robot sits on the real one, then press p and send the numbers.

Trackbars: eye x/y/z and aim x/y/z in cm (URDF frame: x forward from the base, y to the robot's left, z up),
FOV in degrees. Image rotation: SQUINT_TOP_CAMERA_ROTATION (0/90/180/-90) or the o key (cycles 0, 90, 180, 270).
Keys: p print  c save snapshot+json to tune_snapshots/  o rotate real image  n next camera index  q quit

Which index is the overhead camera: OpenCV's macOS index changes with what is plugged in and in what order
(phi/src/phi/utils/camera_realign.py records two different orders), so it is chosen by looking: press n until the
left panel shows the overhead view. The index in use is printed on the panel and saved with c.

Run (repo root, a Terminal with Camera permission, arm powered, torque state left as it is):
  source env.sh && python deploy_utils/tune_top_camera.py          # starts at index SQUINT_TOP_CAMERA or 0
Check without hardware:  python deploy_utils/tune_top_camera.py --selftest
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs import Pose

import envs  # noqa: F401  registers SO101PlaceCubeV2FarTop-v1
import envs.place_v2 as v2

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SIZE = 480
ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
SNAP_DIR = Path(__file__).resolve().parent.parent / "tune_snapshots"
# trackbar name -> (cm min, cm max); value stored in metres
SLIDERS = {"eye x": (-60, 120), "eye y": (-80, 80), "eye z": (5, 150), "aim x": (-20, 80), "aim y": (-40, 40), "aim z": (-10, 30)}


class TopCameraFitter:
    def __init__(self, cam, rotation, arm, headless=False, cam_index=None):
        self.cam, self.arm, self.headless, self.cam_index = cam, arm, headless, cam_index
        self.rot = rotation
        self.eye = list(v2.TOP_CAMERA_EYE)
        self.aim = list(v2.TOP_CAMERA_TARGET)
        self.fov = float(np.degrees(v2.TOP_CAMERA_FOV))
        self.env = gym.make("SO101PlaceCubeV2FarTop-v1", obs_mode="rgb+segmentation", render_mode="sensors", num_envs=1,
                            domain_randomization=False, sensor_configs={"width": SIZE, "height": SIZE})
        self.env.reset(seed=0)
        self.u = self.env.unwrapped
        far = Pose.create_from_pq(p=[[5.0, 5.0, 0.0125]])
        self.u.item.set_pose(far)            # WHY: the real cube and bin sit wherever; only the robot is a fixed reference
        self.u.bin.set_pose(Pose.create_from_pq(p=[[-5.0, 5.0, 0.0]]))
        self.qpos_deg = {j: float(np.degrees(v)) for j, v in zip(JOINTS, self.u.rest_qpos.cpu().numpy())}

    # --- real side ---
    def real_image(self):
        if self.cam is None:   # selftest: a "camera" mounted rotated by self.rot; undoing it must give the sim view back
            ok, bgr = True, self._rotate_back(self.sim_image())
        else:
            ok, bgr = self.cam.read()
        if not ok:
            return np.zeros((SIZE, SIZE, 3), np.uint8)
        if ROTATIONS[self.rot] is not None:
            bgr = cv2.rotate(bgr, ROTATIONS[self.rot])
        h, w = bgr.shape[:2]
        s = min(h, w)                         # = deploy.py create_wrist_camera_preprocessor: centre square, resize
        bgr = bgr[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
        return cv2.resize(bgr, (SIZE, SIZE))

    def _rotate_back(self, img):
        inv = {0: None, 90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}[self.rot]
        return img if inv is None else cv2.rotate(img, inv)

    # --- sim side ---
    def sync_arm(self):
        if self.arm is not None:
            from deploy_utils.tune_camera_nomotion import JOINTS as J
            deg, _ = self.arm.qpos_deg()
            self.qpos_deg = {j: deg[j] for j in J}
        robot = self.u.agent.robot
        q = torch.deg2rad(torch.tensor([[self.qpos_deg[j] for j in JOINTS]], dtype=torch.float32))
        lim = robot.get_qlimits()[0].cpu()
        robot.set_qpos(torch.max(torch.min(q, lim[:, 1]), lim[:, 0]))

    def sim_image(self):
        eye = np.array(self.eye, dtype=np.float64)
        aim = np.array(self.aim, dtype=np.float64)
        if np.linalg.norm((aim - eye)[:2]) < 1e-3:   # WHY: look_at with up=z is undefined for a perfectly vertical view
            eye[0] -= 1e-3
        self.u.camera_mount.set_pose(sapien_utils.look_at(eye.tolist(), aim.tolist()))
        self.u.scene.sensors["top_camera"].camera.set_fovy(np.deg2rad(self.fov))
        rgb = self.u.get_obs()["sensor_data"]["top_camera"]["rgb"][0].cpu().numpy()
        return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)

    # --- ui ---
    def frame(self):
        self.sync_arm()
        sim = self.sim_image()
        real = self.real_image()
        blend = cv2.addWeighted(real, 0.5, sim, 0.5, 0)
        row = np.concatenate([real, sim, blend], 1)
        for i, t in enumerate([f"real: index {self.cam_index}, rot {self.rot}", "sim", "blend"]):
            cv2.putText(row, t, (10 + i * SIZE, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return row

    def params(self):
        r = lambda v: [round(x, 4) for x in v]
        return {"TOP_CAMERA_EYE": r(self.eye), "TOP_CAMERA_TARGET": r(self.aim), "TOP_CAMERA_FOV_DEG": round(self.fov, 1),
                "SQUINT_TOP_CAMERA_ROTATION": {270: -90}.get(self.rot, self.rot), "SQUINT_TOP_CAMERA": self.cam_index,
                "qpos_deg": self.qpos_deg}

    def print_params(self):
        p = self.params()
        print(f"\nTOP_CAMERA_EYE = {p['TOP_CAMERA_EYE']}\nTOP_CAMERA_TARGET = {p['TOP_CAMERA_TARGET']}\n"
              f"TOP_CAMERA_FOV = np.deg2rad({p['TOP_CAMERA_FOV_DEG']})\nSQUINT_TOP_CAMERA={p['SQUINT_TOP_CAMERA']}  "
              f"SQUINT_TOP_CAMERA_ROTATION={p['SQUINT_TOP_CAMERA_ROTATION']}")

    def save(self, img):
        SNAP_DIR.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(str(SNAP_DIR / f"{stamp}_top_camera.png"), img)
        (SNAP_DIR / f"{stamp}_top_camera.json").write_text(json.dumps(self.params(), indent=1))
        print("saved", SNAP_DIR / f"{stamp}_top_camera.json")

    def run(self):
        win = "top camera fit | p print  c save  o rotate  q quit"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        for k, (lo, hi) in SLIDERS.items():
            idx, i = (self.eye, "xyz".index(k[-1])) if k.startswith("eye") else (self.aim, "xyz".index(k[-1]))
            cv2.createTrackbar(k + " cm", win, int(round(idx[i] * 100)) - lo, hi - lo,
                               lambda v, idx=idx, i=i, lo=lo: idx.__setitem__(i, (v + lo) / 100))
        cv2.createTrackbar("FOV deg", win, int(self.fov), 120, lambda v: setattr(self, "fov", float(max(v, 20))))
        while True:
            img = self.frame()
            cv2.imshow(win, img)
            k = cv2.waitKey(30) & 0xFF
            if k == ord("q"):
                break
            if k == ord("p"):
                self.print_params()
            if k == ord("c"):
                self.save(img)
            if k == ord("o"):
                self.rot = (self.rot + 90) % 360
            if k == ord("n") and self.cam is not None:
                self.cam.release()
                self.cam_index, self.cam = open_next_camera(self.cam_index + 1)
        self.print_params()


def open_camera(index):
    """The deploy capture (robot_config.py _top_camera: 640x480 @ 30), or None if nothing answers at this index."""
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    ok = cap.isOpened() and any(cap.read()[0] for _ in range(10))   # WHY: first frames are often empty
    if not ok:
        cap.release()
        return None
    return cap


def open_next_camera(start, max_index=8):
    """First index from `start` (wrapping) that delivers frames."""
    for k in range(max_index):
        i = (start + k) % max_index
        cap = open_camera(i)
        if cap is not None:
            print(f"camera index {i}")
            return i, cap
    raise SystemExit("no camera delivered frames: check this Terminal's Camera permission")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="no camera, no arm: check the sim side and save one frame")
    ap.add_argument("--no-arm", action="store_true", help="use the start pose instead of reading the arm")
    a = ap.parse_args()
    rot = int(os.environ.get("SQUINT_TOP_CAMERA_ROTATION", "0")) % 360
    if a.selftest:
        f = TopCameraFitter(None, rot, None, headless=True)
        img = f.frame()
        sim, real = img[:, SIZE:2 * SIZE], img[:, :SIZE]
        robot_px = int((sim.max(-1) > 20).sum())
        assert robot_px > 500, f"sim top view shows almost nothing ({robot_px} px): camera pose wrong?"
        assert np.abs(sim[40:].astype(int) - real[40:].astype(int)).max() == 0, "selftest real frame should equal the sim frame"   # rows < 40 hold the labels
        SNAP_DIR.mkdir(exist_ok=True)
        cv2.imwrite(str(SNAP_DIR / "top_camera_selftest.png"), img)
        print(f"selftest OK: robot {robot_px} px in the sim top view; saved {SNAP_DIR / 'top_camera_selftest.png'}")
        f.print_params()
        return
    idx, cap = open_next_camera(int(os.environ.get("SQUINT_TOP_CAMERA", "0")))
    arm = None
    if not a.no_arm:
        from deploy_utils.tune_camera_nomotion import ReadOnlyArm
        arm = ReadOnlyArm()
    try:
        fitter = TopCameraFitter(cap, rot, arm, cam_index=idx)
        fitter.run()
    finally:
        try:
            fitter.cam.release()
        except Exception:
            pass
        if arm is not None:
            print("arm unchanged check:", arm.close())


if __name__ == "__main__":
    main()
