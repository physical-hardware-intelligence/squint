"""
Phi 2026-10-13: does the arm hold still when deploy.py's connect turns torque on? Run with a person watching.

Connects exactly the way deploy.py does (create_real_robot().connect(): LeRobot configure, then torque on, with
robot_config.py's goal = present patch), reads every joint for 3 s, then disconnects. Disconnect turns torque OFF,
so the arm goes limp: start with it resting (folded on its stops) or hold it.
Pass: no joint moves more than 1 degree. Sends no goal of its own.
"""

import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deploy_utils.robot_config import create_real_robot

robot = create_real_robot()


def raw(bus):
    return bus.sync_read("Present_Position", normalize=False, num_retry=2)


input("Arm resting or held, someone watching, hand near the power? Enter to turn torque on (Ctrl-C to abort) ")
robot.connect(calibrate=False)
try:
    bus = robot.bus
    start = raw(bus)
    goal = bus.sync_read("Goal_Position", normalize=False, num_retry=2)
    torque = bus.sync_read("Torque_Enable", normalize=False, num_retry=2)
    print("torque:", torque)
    print("goal - present at torque-on (ticks):", {j: int(goal[j]) - int(start[j]) for j in start})
    worst = {}
    t0 = time.time()
    while time.time() - t0 < 3.0:
        now = raw(bus)
        for j in now:
            worst[j] = max(worst.get(j, 0), abs(int(now[j]) - int(start[j])))
        time.sleep(0.05)
    deg = {j: round(v * 360 / 4096, 2) for j, v in worst.items()}
    print("largest movement over 3 s (deg):", deg)
    ok = max(deg.values()) <= 1.0
    print("HOLD TEST", "PASS" if ok else "FAIL")
finally:
    robot.disconnect()   # torque off (disable_torque_on_disconnect=True): the arm goes limp
    print("disconnected, torque off")
sys.exit(0 if ok else 1)   # non-zero stops a chained `hold_test.py && deploy.py`
