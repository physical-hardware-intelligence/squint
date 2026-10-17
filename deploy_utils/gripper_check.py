"""
Phi 2026-10-16: the gripper check Squint PR #9's author asked for (aalmuzairee/squint#9).

Everything goes through deploy_utils/manipulator.py's LeRobotRealAgent, which is byte-identical to the PR branch's
(fix/so101-gripper-lerobot-0.4.3): it switches the gripper to degrees and maps servo [-62.5, 64.62] <-> sim [-10, 120].
The wrist camera is not used.

  --read    torque untouched, no servo writes. Close the jaws fully by hand, Enter; open fully, Enter. Prints the
            servo degrees (bus, DEGREES mode), the raw ticks and the sim degrees LeRobotRealAgent.get_qpos() reports.
  --drive   torque on (robot_config: goal = present first, 5 deg gripper lead cap), rise to the start pose exactly
            like deploy.py's reset, then the gripper alone, 1 sim degree per command at 15 Hz: sim -10 (closed),
            120 (open), back to 60 (the start pose's opening). Pauses for Enter (and photos) at each end. Stops if
            the gripper does not reach a target. Folds to the exit pose and releases torque at the end.
  --fake    with --drive: deploy_dryrun's perfect-servo fake arm instead of the real one (no hardware).

Run with a person watching, hand near the power (repo root): source env.sh && python deploy_utils/gripper_check.py --read
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from deploy_utils.manipulator import LeRobotRealAgent
from envs.robot.so101 import SO101

ROOT = Path(__file__).resolve().parent.parent
REACHED_DEG = 4.0          # servo degrees: farther than this from the target after settling = did not reach it
STEP_SIM_DEG, HZ, SETTLE_S = 1.0, 15, 1.5
# Blocked = over the last STALL_STEPS commands the jaw moved less than STALL_MOVE_DEG while its target led it by more
# than STALL_LEAD_DEG (servo degrees). Then the target is set to where the jaw is, so the servo stops pushing.
# Why (2026-10-16): a first by-hand reading stopped at servo +61.19, 3.4 deg short of the map's 64.62; fully opened,
# our jaw reached +65.23. The guard keeps a short stop from turning into a servo pushing against it.
STALL_STEPS, STALL_MOVE_DEG, STALL_LEAD_DEG = 5, 0.3, 2.0


def sim_from_servo(agent, servo_deg):
    """LeRobotRealAgent.get_qpos's servo -> sim gripper formula."""
    return (servo_deg - agent._gripper_servo_min) / agent._gripper_servo_range * agent._gripper_sim_range + agent._gripper_sim_min


def servo_target(agent, sim_deg):
    """The servo degrees LeRobotRealAgent.set_target_qpos sends for a sim gripper angle (its own formula)."""
    return (sim_deg - agent._gripper_sim_min) / agent._gripper_sim_range * agent._gripper_servo_range + agent._gripper_servo_min


def read_gripper(robot, agent, n=10):
    """Mean over n reads: servo degrees, raw ticks, sim degrees (the PR's get_qpos), raw load."""
    servo, ticks, sim, load = [], [], [], []
    for _ in range(n):
        servo.append(robot.bus.sync_read("Present_Position", num_retry=2)["gripper"])
        ticks.append(robot.bus.sync_read("Present_Position", normalize=False, num_retry=2)["gripper"])
        sim.append(math.degrees(float(agent.get_qpos()[0, -1])))
        try:
            load.append(robot.bus.sync_read("Present_Load", normalize=False, num_retry=2)["gripper"])
        except Exception:
            pass
        time.sleep(0.02)
    out = {"servo_deg": round(float(np.mean(servo)), 2), "ticks": int(round(np.mean(ticks))),
           "sim_deg": round(float(np.mean(sim)), 2)}
    if load:
        out["load_raw"] = int(round(np.mean(load)))
    return out


def read_mode(save):
    from deploy_utils.robot_config import create_real_robot
    from deploy_utils.tune_camera_nomotion import ReadOnlyArm
    robot = create_real_robot()
    robot.cameras = {}                                  # not needed; connect() would open the wrist cam
    agent = LeRobotRealAgent(robot, use_cached_qpos=False)
    robot.bus.connect(handshake=True)                   # open port + ping + firmware read; lerobot writes nothing here
    for name in ReadOnlyArm.WRITES:                     # make any write fail loudly
        if hasattr(robot.bus, name):
            setattr(robot.bus, name, ReadOnlyArm._refuse(name))
    result = {"mode": "read (torque untouched)", "robot_name": robot.name,
              "map": {"servo": [agent._gripper_servo_min, agent._gripper_servo_max],
                      "sim": [agent._gripper_sim_min, agent._gripper_sim_max]}}
    try:
        torque = robot.bus.sync_read("Torque_Enable", normalize=False, num_retry=2)
        result["torque_start"] = torque
        if any(int(v) for v in torque.values()):
            print(f"WARNING: torque is ON on {[j for j, v in torque.items() if int(v)]}; the jaws will not move by hand.")
        print(f"robot.name = {robot.name!r}; gripper norm mode: {robot.bus.motors['gripper'].norm_mode}")
        print("now:", read_gripper(robot, agent))
        input("\nClose the jaws FULLY by hand and hold them there, then press Enter ")
        result["closed"] = read_gripper(robot, agent)
        print("closed:", result["closed"], "  <- photo now")
        input("\nOpen the jaws FULLY by hand and hold them there, then press Enter ")
        result["open"] = read_gripper(robot, agent)
        print("open:  ", result["open"], "  <- photo now")
        result["torque_end"] = robot.bus.sync_read("Torque_Enable", normalize=False, num_retry=2)
    finally:
        robot.bus.disconnect(disable_torque=False)
    save(result)


def drive_mode(save, fake, fake_open_stop=None):
    from deploy_utils.safety import exit_qpos
    if fake:
        import cv2
        import tempfile
        from deploy_utils.deploy_dryrun import FakeRobot
        frame = os.path.join(tempfile.mkdtemp(), "black.png")
        cv2.imwrite(frame, np.zeros((480, 640, 3), np.uint8))     # FakeCamera wants a file; the camera is unused
        robot = FakeRobot(camera="frame", frame=frame)
        if fake_open_stop is not None:                             # a jaw that stops opening at this servo angle
            send = robot.send_action

            def send_with_stop(action):
                out = send(action)
                robot.bus.pos["gripper"] = min(robot.bus.pos["gripper"], fake_open_stop)
                return out
            robot.send_action = send_with_stop
    else:
        from deploy_utils.robot_config import create_real_robot
        robot = create_real_robot()
    robot.cameras = {}
    agent = LeRobotRealAgent(robot, use_cached_qpos=False)
    ask = (lambda prompt: print(prompt + " [fake: auto]")) if fake else input
    result = {"mode": "drive" + (" (FAKE arm)" if fake else ""), "robot_name": robot.name, "ends": {}}

    ask("Arm resting or held, someone watching, hand near the power? Enter to turn torque on (Ctrl-C to abort) ")
    agent.start()                                       # robot.connect(): configure, goal = present, torque on
    folded = False
    try:
        agent.get_qpos()                                # sets the motor order LeRobotRealAgent sends in
        result["at_torque_on"] = read_gripper(robot, agent, n=3)
        start = torch.tensor(SO101.keyframes["start"].qpos, dtype=torch.float32)
        ask("\nEnter: rise to the START pose (all joints together, like deploy.py's reset; gripper to sim 60) ")
        agent.reset(start.clone())
        time.sleep(SETTLE_S)
        result["ends"]["start_60"] = read_gripper(robot, agent)
        print("at the start pose:", result["ends"]["start_60"])
        q = start.clone()
        for label, target in (("closed_-10", -10.0), ("open_120", 120.0), ("back_60", 60.0)):
            ask(f"\nEnter: gripper ONLY to sim {target:+.0f} deg (servo {servo_target(agent, target):+.2f}), "
                f"{STEP_SIM_DEG:.0f} deg per command at {HZ} Hz ")
            cur = math.degrees(float(q[-1]))
            n = max(1, int(math.ceil(abs(target - cur) / STEP_SIM_DEG)))
            seen, blocked = [], None
            for i in range(1, n + 1):
                q[-1] = math.radians(cur + (target - cur) * i / n)
                agent.set_target_qpos(q.clone())
                time.sleep(1 / HZ)
                present = robot.bus.sync_read("Present_Position", num_retry=2)["gripper"]
                seen = (seen + [present])[-(STALL_STEPS + 1):]
                lead = servo_target(agent, math.degrees(float(q[-1]))) - present
                if (len(seen) > STALL_STEPS and abs(seen[-1] - seen[0]) < STALL_MOVE_DEG
                        and abs(lead) > STALL_LEAD_DEG):
                    q[-1] = math.radians(sim_from_servo(agent, present))   # stop pushing: target = where the jaw is
                    agent.set_target_qpos(q.clone())
                    blocked = {"present_servo_deg": round(present, 2), "lead_servo_deg": round(lead, 2),
                               "at_command": i, "of": n}
                    print(f"  blocked at servo {present:+.2f} (target led it by {lead:+.2f}): stopped pushing")
                    break
            time.sleep(SETTLE_S)
            got = read_gripper(robot, agent)
            got["target_sim_deg"], got["target_servo_deg"] = target, round(servo_target(agent, target), 2)
            got["error_servo_deg"] = round(got["servo_deg"] - got["target_servo_deg"], 2)
            if blocked is None and abs(got["error_servo_deg"]) > STALL_LEAD_DEG:
                # settled short of the target (a stop near the end of the sweep): stop pushing there too
                q[-1] = math.radians(sim_from_servo(agent, got["servo_deg"]))
                agent.set_target_qpos(q.clone())
                blocked = {"present_servo_deg": got["servo_deg"], "lead_servo_deg": -got["error_servo_deg"],
                           "after": "settle"}
                print(f"  settled at servo {got['servo_deg']:+.2f}, {-got['error_servo_deg']:+.2f} short of the "
                      "target: stopped pushing")
            if blocked:
                got["blocked"] = blocked
            result["ends"][label] = got
            print(f"{label}:", got, "  <- photo now" if label != "back_60" else "")
            if blocked and label != "open_120":
                print("\nSTOP: blocked before an end that should be free. Nothing more will move until the fold. "
                      "Tell me what you saw.")
                result["stopped_at"] = label
                break
            if not blocked and abs(got["error_servo_deg"]) > REACHED_DEG:
                print(f"\nSTOP: the gripper is {got['error_servo_deg']:+.1f} servo deg from its target. "
                      "Nothing more will move until the fold. Tell me what you saw.")
                result["stopped_at"] = label
                break
        ask("\nEnter: fold to the exit pose and release torque ")
        agent.reset(torch.tensor(exit_qpos(), dtype=torch.float32))
        folded = True
    finally:
        if folded:
            robot.disconnect()                          # torque off at the folded pose
            print("folded, torque off")
        else:
            print("\nInterrupted before the fold. Torque is still ON and holding. Support the arm, then run "
                  "deploy_utils/hold_test.py style release or cut its power.")
            try:
                ask("Support the arm, then press Enter to fold it to the exit pose ")
                agent.reset(torch.tensor(exit_qpos(), dtype=torch.float32))
                robot.disconnect()
                print("folded, torque off")
            except Exception as e:                      # keep torque on: releasing would drop a raised arm
                print(f"fold FAILED ({e}). Torque is still ON: support the arm, then cut its power.")
    if fake:
        result["fake_commands"] = len(robot.log)
        result["fake_largest_gripper_step_servo_deg"] = round(max(abs(c["step"].get("gripper", 0.0)) for c in robot.log), 3)
        result["fake_largest_arm_step_deg"] = round(max(abs(v) for c in robot.log for j, v in c["step"].items() if j != "gripper"), 3)
        result["fake_events"] = robot.events
    save(result)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--read", action="store_true")
    g.add_argument("--drive", action="store_true")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--fake-open-stop", type=float, default=None, help="with --fake: the jaw stops at this servo angle")
    ap.add_argument("--out", default=str(ROOT / "runs" / "pr9_gripper_check"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    def save(result):
        result["time"] = datetime.now().isoformat(timespec="seconds")
        path = out / f"{'drive' if a.drive else 'read'}{'_fake' if a.fake else ''}_{datetime.now():%Y%m%d_%H%M%S}.json"
        path.write_text(json.dumps(result, indent=1))
        print(f"\nsaved {path}")

    if a.read:
        read_mode(save)
    else:
        drive_mode(save, a.fake, a.fake_open_stop)


if __name__ == "__main__":
    main()
