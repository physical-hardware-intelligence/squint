"""Robot configuration for real deployment.

Set your rig with environment variables (see rig.example.sh), or edit the defaults below. The defaults are
upstream Squint's:

  SQUINT_PORT              serial port of the follower arm             default /dev/ttyACM0
  SQUINT_CAMERA            wrist camera: OpenCV index or device path   default /dev/video0
  SQUINT_CAMERA_ROTATION   0, 90, 180 or -90 degrees                   default 0
  SQUINT_ROBOT_ID          calibration file name, without .json        default so101_follower_arm
  SQUINT_CALIBRATION_DIR   folder holding that file                    default: this folder

On top of upstream, every robot created here (docs/so101-phi.md):
  * holds its position when torque comes on (goal = present position first),
  * retries bus reads and writes, and turns every servo back off if torque-on fails,
  * caps how far each goal may lead the present position (MAX_GOAL_LEAD_DEG).
"""

import os
from pathlib import Path

from lerobot.cameras.configs import Cv2Rotation
from lerobot.robots.robot import Robot
from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig, SO100FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
# Each goal at most this many degrees ahead of the present position (LeRobot's default: no limit). Bounds how hard
# a blocked joint pushes and how hard the gripper squeezes; normal policy steps (< 1 degree) pass unchanged.
MAX_GOAL_LEAD_DEG = {**{j: 8.0 for j in ARM_JOINTS}, "gripper": 5.0}
BUS_RETRIES = 2   # extra attempts per bus read/write (LeRobot's default: 0)


def _camera(value: str):
    return int(value) if value.isdigit() else value


def create_real_robot() -> Robot:
    """Create and configure a real robot with the specified camera.
    Returns:
        Configured Robot instance
    """
    robot_config = SO101FollowerConfig(
        port=os.environ.get("SQUINT_PORT", "/dev/ttyACM0"),
        use_degrees=True,
        max_relative_target=MAX_GOAL_LEAD_DEG,
        cameras={"base_camera": OpenCVCameraConfig(
            index_or_path=_camera(os.environ.get("SQUINT_CAMERA", "/dev/video0")),
            fps=30,
            width=640,
            height=480,
            rotation=Cv2Rotation(int(os.environ.get("SQUINT_CAMERA_ROTATION", "0"))),
        )},
        # cameras={"base_camera": RealSenseCameraConfig(
        #     serial_number_or_name="053645021390",
        #     fps=30,
        #     width=640,
        #     height=480
        # )},
        id=os.environ.get("SQUINT_ROBOT_ID", "so101_follower_arm"),
        calibration_dir=Path(os.environ.get("SQUINT_CALIBRATION_DIR", Path(__file__).parent)),
    )

    robot = make_robot_from_config(robot_config)
    retry_bus_io(robot)
    hold_position_on_torque_enable(robot)
    return robot


def retry_bus_io(robot: Robot) -> None:
    """Retry every bus read and write up to BUS_RETRIES more times.

    One lost status packet ("Failed to write 'Torque_Enable' on id_=2 ... There is no status packet!") is enough
    to abort a connect, although the same bus then read 1,000 times without a failure. Writing a register again
    with the same value is idempotent, so a retry is safe.
    """
    bus = robot.bus
    for name in ("read", "write", "sync_read", "sync_write"):
        original = getattr(bus, name)

        def with_retries(*args, _original=original, **kwargs):
            kwargs["num_retry"] = max(int(kwargs.get("num_retry", 0) or 0), BUS_RETRIES)
            return _original(*args, **kwargs)

        setattr(bus, name, with_retries)


def hold_position_on_torque_enable(robot: Robot) -> None:
    """Write goal = present position right before every torque-on, so the arm holds where it is.

    Goal_Position is a RAM register: after power-up it read 0 on all six servos of our arm while the joints sat
    799-3095 ticks away. LeRobot's connect() runs configure() inside bus.torque_disabled(), which ends with
    bus.enable_torque() and never sets the goal first.

    LeRobot also enables torque servo by servo, so a failure part-way leaves the earlier servos holding. If torque-on
    fails, every servo is turned back off before the error is raised.
    """
    bus = robot.bus
    original_enable = bus.enable_torque

    def enable_torque_holding(motors=None, num_retry: int = 0) -> None:
        present = bus.sync_read("Present_Position", motors, normalize=False, num_retry=BUS_RETRIES)
        bus.sync_write("Goal_Position", present, normalize=False, num_retry=BUS_RETRIES)
        try:
            original_enable(motors, num_retry=max(num_retry, BUS_RETRIES))
        except Exception as e:
            silent = []
            for m in bus.motors:
                try:
                    bus.write("Torque_Enable", m, 0, normalize=False)
                    bus.write("Lock", m, 0, normalize=False)
                except Exception:
                    silent.append(m)
            after = ("every servo's torque was turned back off" if not silent else
                     f"{', '.join(silent)} did not confirm torque off: support the arm, then cut its power")
            raise ConnectionError(f"Torque-on failed ({e}). Rolled back: {after}. Check the servo cables and the "
                                  "12 V supply, then run again.") from e

    bus.enable_torque = enable_torque_holding
