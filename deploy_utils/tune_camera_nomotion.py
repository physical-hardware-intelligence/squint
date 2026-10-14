"""
No-motion wrist camera tuner for SO101 sim2real (Phi, 2026-10-13).

Same window, trackbars and printed numbers as tune_camera.py (it subclasses LiveCameraTuner), with one inversion:
the real arm never moves. The SIM copies the REAL arm's joint angles every frame, the way deploy.py syncs them.

  * The arm is read through a bare Feetech bus. No LeRobot Robot object, so no configure(): no PID or torque-limit
    writes and no torque-on. Every write or torque call on the bus raises.
  * Torque_Enable and Goal_Position are read at start and checked unchanged at exit.
  * Exit, Ctrl-C and the r/s keys of tune_camera.py never send a goal (r/s are removed).
  * With torque off you can move the arm BY HAND to view the gripper from other poses; the sim follows.

Keys: p=print params  o=toggle 180 deg image flip  c=save snapshot  f=apply FOV  q=quit
Run (repo root):  source env.sh && python deploy_utils/tune_camera_nomotion.py [--rotate-180]
Check without a window:  ... tune_camera_nomotion.py --selftest
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import atexit
import json
import signal
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from deploy_utils.tune_camera import LiveCameraTuner

# rig: the same SQUINT_* variables as robot_config.py (see rig.example.sh)
PORT = os.environ.get("SQUINT_PORT", "/dev/ttyACM0")
CALIBRATION = (Path(os.environ.get("SQUINT_CALIBRATION_DIR", Path(__file__).resolve().parent))
               / f"{os.environ.get('SQUINT_ROBOT_ID', 'so101_follower_arm')}.json")
_cam = os.environ.get("SQUINT_CAMERA", "0")
CAM_INDEX = int(_cam) if _cam.isdigit() else _cam
ROTATE_180_DEFAULT = os.environ.get("SQUINT_CAMERA_ROTATION", "0") == "180"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# manipulator.py's SO-101 gripper map, servo degrees -> sim degrees.
GRIPPER_SERVO = (-62.5, 64.62)
GRIPPER_SIM = (-10.0, 120.0)


class ReadOnlyArm:
    """Joint angles from the SO101 follower without ever writing to a servo."""

    WRITES = ("write", "sync_write", "enable_torque", "disable_torque", "torque_disabled", "write_calibration",
              "configure_motors", "reset_calibration", "setup_motor", "set_half_turn_homings",
              "record_ranges_of_motion", "set_baudrate")

    def __init__(self, port=PORT, calibration=CALIBRATION):
        from lerobot.motors import Motor, MotorCalibration, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus

        cal = json.loads(Path(calibration).read_text())
        self.bus = FeetechMotorsBus(
            port=port,
            motors={j: Motor(cal[j]["id"], "sts3215", MotorNormMode.DEGREES) for j in JOINTS},
            calibration={j: MotorCalibration(**cal[j]) for j in JOINTS},
        )
        self.bus.connect(handshake=True)   # open port + ping + firmware read; lerobot writes nothing here
        for name in self.WRITES:
            if hasattr(self.bus, name):
                setattr(self.bus, name, self._refuse(name))

        self.calibration_name = Path(calibration).name
        regs = self.bus.read_calibration()
        self.calibration_diff = {j: (regs[j].homing_offset - cal[j]["homing_offset"],
                                     regs[j].range_min - cal[j]["range_min"],
                                     regs[j].range_max - cal[j]["range_max"]) for j in JOINTS}
        self.torque_start = self.read_raw("Torque_Enable")
        self.goal_start = self.read_raw("Goal_Position")
        self.pos_start = self.read_raw("Present_Position")
        self._closed = False

    @staticmethod
    def _refuse(name):
        def refuse(*args, **kwargs):
            raise RuntimeError(f"no-motion tuner: bus.{name}() is disabled, this tool never writes to a servo")
        return refuse

    def read_raw(self, register):
        return self.bus.sync_read(register, normalize=False, num_retry=2)

    def qpos_deg(self):
        """Servo degrees for the arm joints, sim degrees for the gripper (manipulator.py's intended map)."""
        deg = self.bus.sync_read("Present_Position", num_retry=2)
        servo = deg["gripper"]
        deg["gripper"] = ((servo - GRIPPER_SERVO[0]) / (GRIPPER_SERVO[1] - GRIPPER_SERVO[0])
                          * (GRIPPER_SIM[1] - GRIPPER_SIM[0]) + GRIPPER_SIM[0])
        return {j: float(deg[j]) for j in JOINTS}, float(servo)

    def close(self):
        """Disconnect without touching torque; report whether anything on the arm changed."""
        if self._closed:
            return None
        self._closed = True
        torque, goal, pos = self.read_raw("Torque_Enable"), self.read_raw("Goal_Position"), self.read_raw("Present_Position")
        self.bus.disconnect(disable_torque=False)
        return {
            "torque_start": self.torque_start, "torque_end": torque,
            "goal_unchanged": goal == self.goal_start,
            "position_drift_ticks": {j: int(pos[j]) - int(self.pos_start[j]) for j in JOINTS},
        }


class NoMotionTuner(LiveCameraTuner):
    def __init__(self, env_id, sim_width, sim_height, arm, cam_index=CAM_INDEX, rotate_180=False,
                 headless=False, snapshot_dir=None):
        self.arm, self.cam_index, self.rotate_180 = arm, cam_index, rotate_180
        self.headless = headless
        self.snapshot_dir = Path(snapshot_dir or Path(__file__).resolve().parent.parent / "tune_snapshots")
        self.last_qpos_deg, self.last_gripper_servo, self.clamped = {}, None, {}
        self._closed = False
        self.arm_report = None
        super().__init__(env_id, sim_width, sim_height)
        names = [j.name for j in self.sim_env.unwrapped.agent.robot.active_joints]
        if names != JOINTS:
            raise SystemExit(f"sim joint order {names} != {JOINTS}; refusing to map angles")

    # --- Real side: camera only, the arm is ReadOnlyArm ---

    def _setup_real_robot(self):
        self.real_robot = self.real_agent = None
        self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)    # = robot_config.py OpenCVCameraConfig
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            raise SystemExit("Could not open the camera. Run this in a Terminal that has Camera permission.")
        for _ in range(10):                            # let the stream settle before the first comparison
            self.cap.read()

    def _get_real_image(self):
        ok, bgr = self.cap.read()
        if not ok:
            return None
        if self.rotate_180:                            # = lerobot Cv2Rotation.ROTATE_180 in robot_config.py
            bgr = cv2.rotate(bgr, cv2.ROTATE_180)
        h, w = bgr.shape[:2]                           # same centre crop + resize as tune_camera.py
        s = min(h, w)
        c = (max(h, w) - s) // 2
        bgr = bgr[c:c + s, :, :] if h > w else bgr[:, c:c + s, :]
        return cv2.resize(bgr, (self.sim_width, self.sim_height))

    # --- Sim follows real (replaces "real moves to sim") ---

    def _move_real_to_sim_pose(self):
        self._sync_sim_to_real()

    def _sync_sim_to_real(self):
        deg, self.last_gripper_servo = self.arm.qpos_deg()
        robot = self.sim_env.unwrapped.agent.robot
        q = torch.deg2rad(torch.tensor([[deg[j] for j in JOINTS]], dtype=torch.float32))
        lim = robot.get_qlimits()[0].cpu()
        q_clamped = torch.max(torch.min(q[0], lim[:, 1]), lim[:, 0])
        self.clamped = {j: float(np.degrees(q[0, i] - q_clamped[i])) for i, j in enumerate(JOINTS)
                        if abs(float(q[0, i] - q_clamped[i])) > 1e-6}
        robot.set_qpos(q_clamped.unsqueeze(0))
        robot.set_qvel(torch.zeros_like(q))
        env = self.sim_env.unwrapped
        if env.gpu_sim_enabled:
            env.scene._gpu_apply_all()
        self.last_qpos_deg = deg

    def _update_camera(self):
        self._sync_sim_to_real()
        super()._update_camera()

    # --- UI and exit: no motion anywhere ---

    def _setup_ui(self):
        if not self.headless:
            super()._setup_ui()
            self.win_title = "No-motion tuner | p:print o:flip180 c:snapshot f:FOV q:quit"
            cv2.setWindowTitle(self.win, self.win_title)

    def _setup_exit(self):
        def cleanup(sig=None, frame=None):
            if not self._closed:
                self._closed = True
                for step in (self.cap.release, self._close_arm, self.sim_env.close):
                    try:
                        step()
                    except Exception as e:
                        print(f"cleanup: {e}")
            if sig is not None:
                sys.exit(0)

        signal.signal(signal.SIGINT, cleanup)
        atexit.register(cleanup)
        self._cleanup = cleanup

    def _close_arm(self):
        self.arm_report = self.arm.close()
        if self.arm_report:
            r = self.arm_report
            moved = {j: d for j, d in r["position_drift_ticks"].items() if abs(d) > 2}
            print(f"\nArm check: torque {r['torque_start']} -> {r['torque_end']} | goal unchanged: {r['goal_unchanged']}"
                  f" | joints moved >2 ticks since start: {moved or 'none'}")

    def params(self):
        return {"WRIST_CAMERA_BASE_POS": [round(self.cam_x, 4), round(self.cam_y, 4), round(self.cam_z, 4)],
                "WRIST_CAMERA_BASE_ROT_DEG": [round(self.cam_roll, 1), round(self.cam_pitch, 1), round(self.cam_yaw, 1)],
                "WRIST_CAMERA_FOV_DEG": round(self.cam_fov, 1), "rotate_180": self.rotate_180}

    def frame(self):
        self._update_camera()
        comp = self._make_comparison(self._get_real_image(), self._get_sim_image())
        if comp is None:
            return None
        short = {"shoulder_pan": "pan", "shoulder_lift": "lift", "elbow_flex": "elbow", "wrist_flex": "flex",
                 "wrist_roll": "roll", "gripper": "grip"}
        lines = [f"SIM FOLLOWS REAL ARM (read-only, torque {sum(self.arm.torque_start.values())}/6 on)"
                 f"   flip180 {'ON' if self.rotate_180 else 'off'}",
                 "real deg: " + "  ".join(f"{short[j]} {self.last_qpos_deg.get(j, 0):+.0f}" for j in JOINTS)]
        if self.clamped:
            lines.append("clamped to sim limits: " + ", ".join(f"{short[j]} {d:+.1f}" for j, d in self.clamped.items()))
        for i, text in enumerate(lines):
            for color, width in (((0, 0, 0), 3), ((120, 255, 120), 1)):
                cv2.putText(comp, text, (10, 125 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, width)
        return comp

    def snapshot(self, comp, tag="snap"):
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        stem = self.snapshot_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{tag}"
        cv2.imwrite(f"{stem}.png", comp)
        stem.with_suffix(".json").write_text(json.dumps(
            {**self.params(), "qpos_deg": self.last_qpos_deg, "gripper_servo_deg": self.last_gripper_servo,
             "clamped_deg": self.clamped}, indent=2))
        return stem

    def run(self):
        print(__doc__)
        while True:
            comp = self.frame()
            if comp is not None:
                if self.cam_fov != self._last_fov:
                    fov_text = f"FOV: {self._last_fov:.0f}->{self.cam_fov:.0f} (press 'f')"
                    cv2.putText(comp, fov_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
                    cv2.putText(comp, fov_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.imshow(self.win, comp)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                self.print_params()
            elif key == ord("o"):
                self.rotate_180 = not self.rotate_180
                print(f"180 deg flip {'ON' if self.rotate_180 else 'off'}")
            elif key == ord("c") and comp is not None:
                print(f"saved {self.snapshot(comp)}.png/.json")
            elif key == ord("f") and self._fov_pending:
                self._create_sim_env(preserve_fov=True)
                self._fov_pending = False
        cv2.destroyAllWindows()
        self.print_params()
        self._cleanup()


def selftest(args, arm):
    """Headless: render real|sim|blend with and without the flip, prove the trackbars move the sim, prove the arm
    was not touched."""
    t = NoMotionTuner(args.env_id, args.sim_width, args.sim_height, arm, args.cam_index, headless=True,
                      snapshot_dir=args.snapshot_dir)
    print("calibration file:", arm.calibration_name, "| servo register diff vs file (homing, min, max):",
          {j: d for j, d in arm.calibration_diff.items() if any(d)} or "none (exact)")
    print("torque at start:", arm.torque_start)
    robot = t.sim_env.unwrapped.agent.robot
    lim = np.degrees(robot.get_qlimits()[0].cpu().numpy())
    print("sim joint limits (deg):", {j: (round(float(lim[i, 0]), 1), round(float(lim[i, 1]), 1)) for i, j in enumerate(JOINTS)})

    comp = t.frame()
    deg = t.last_qpos_deg
    sim_deg = np.degrees(robot.get_qpos()[0].cpu().numpy())
    print("real arm (deg, gripper in sim deg):", {j: round(v, 2) for j, v in deg.items()},
          "| gripper servo deg:", round(t.last_gripper_servo, 2))
    print("sim qpos after sync (deg):", {j: round(float(v), 2) for j, v in zip(JOINTS, sim_deg)},
          "| clamped:", t.clamped or "none")
    stems = [t.snapshot(comp, "selftest_noflip")]
    t.rotate_180 = True
    stems.append(t.snapshot(t.frame(), "selftest_flip180"))
    t.rotate_180 = False

    before = t._get_sim_image().astype(int)
    x0 = t.cam_x
    t.cam_x += 0.02                                   # +20 mm along the camera-mount x axis
    t._update_camera()
    after = t._get_sim_image().astype(int)
    t.cam_x = x0
    print(f"trackbar check: +20 mm X changes {np.mean(np.abs(after - before) > 10) * 100:.1f}% of sim pixels")
    print("snapshots:", *[f"{s}.png" for s in stems], sep="\n  ")
    print("defaults:", t.params())
    t._cleanup()
    r = t.arm_report
    ok = r and r["torque_start"] == r["torque_end"] and r["goal_unchanged"] and all(
        abs(d) <= 2 for d in r["position_drift_ticks"].values())
    print("SELFTEST", "PASS: arm untouched" if ok else f"FAIL: {r}")
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="No-motion wrist camera tuning (SO101): the sim follows the real arm")
    parser.add_argument("--env-id", default="SO101ReachCube-v1")
    parser.add_argument("--sim-width", type=int, default=480)
    parser.add_argument("--sim-height", type=int, default=480)
    parser.add_argument("--port", default=PORT)
    parser.add_argument("--calibration", default=str(CALIBRATION))
    parser.add_argument("--cam-index", type=int, default=CAM_INDEX)
    parser.add_argument("--rotate-180", action="store_true", help="flip the real image 180 deg (Cv2Rotation.ROTATE_180)")
    parser.add_argument("--snapshot-dir", default=None)
    parser.add_argument("--selftest", action="store_true", help="no window: save comparisons, verify the arm was not touched")
    args = parser.parse_args()

    arm = ReadOnlyArm(args.port, args.calibration)
    if args.selftest:
        sys.exit(selftest(args, arm))
    NoMotionTuner(args.env_id, args.sim_width, args.sim_height, arm, args.cam_index, args.rotate_180 or ROTATE_180_DEFAULT,
                  snapshot_dir=args.snapshot_dir).run()
