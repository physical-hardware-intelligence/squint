"""
Deploy-time safety for the SO-101 (docs/so101-phi.md).

  * exit_qpos(): the pose deploy.py drives to on exit. Squint's rest keyframe, but with wrist roll at the start
    keyframe's -90 deg instead of -180 deg: on a wrist_roll calibrated over the full circle (range 0..4095),
    -180 deg is raw tick 0, the encoder's wrap point. Exit-only, so the policy never sees this pose.
  * guard_until_safe_exit(): torque is on from connect(), but deploy.py installs its exit handler only after the
    Sim2RealEnv is built, which already moves the arm to the start pose. A crash in between left our arm standing,
    holding torque, until its power was cut. The guard folds the arm to exit_qpos() and releases torque; if the
    fold itself fails it keeps torque on (releasing would drop a raised arm) and says so.
"""

import atexit

import numpy as np

from envs.robot.so101 import SO101


def exit_qpos() -> np.ndarray:
    """Squint's rest keyframe with the start keyframe's wrist roll (-90 deg), in the sim joint order."""
    names = SO101.arm_joint_names + SO101.gripper_joint_names
    q = np.array(SO101.keyframes["rest"].qpos, dtype=np.float32).copy()
    roll = names.index("wrist_roll")
    q[roll] = SO101.keyframes["start"].qpos[roll]
    return q


def guard_until_safe_exit(real_agent, real_robot) -> dict:
    """Fold + release on an early exit. Set the returned dict's "armed" to False once deploy's own handler is in."""
    state = {"armed": True}

    def early_exit():
        if not state["armed"]:
            return
        print("\n[safety] deploy exited before its own exit handler was set up. Folding the arm to the exit pose...")
        try:
            real_agent.reset(exit_qpos())
        except Exception as e:
            print(f"[safety] fold FAILED ({e}). Torque is still ON and holding: support the arm, then cut its power.")
            return
        try:
            real_robot.disconnect()   # torque off at the folded pose
            print("[safety] folded, torque off.")
        except Exception as e:
            print(f"[safety] folded, but releasing torque failed ({e}): the arm may still be holding.")

    atexit.register(early_exit)
    return state
