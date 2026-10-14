"""
Read-only motor health check before a run: temperature, load, voltage and error status of every servo.
Uses the no-motion tuner's ReadOnlyArm (no writes, no torque). Exits 1 if any servo is above --max-temp or reports a
status error, so it can gate a launch:  python deploy_utils/motor_check.py && python deploy.py ...
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from deploy_utils.tune_camera_nomotion import JOINTS, ReadOnlyArm


def main():
    ap = argparse.ArgumentParser(description="read-only servo health check")
    ap.add_argument("--max-temp", type=int, default=50, help="refuse above this temperature, deg C (servo protection ~70)")
    a = ap.parse_args()
    arm = ReadOnlyArm()
    try:
        t, l, v, s = (arm.read_raw(r) for r in ("Present_Temperature", "Present_Load", "Present_Voltage", "Status"))
    finally:
        report = arm.close()
    print(f"{'joint':14s} {'temp C':>6s} {'load':>6s} {'volt':>5s} {'status':>6s}")
    for j in JOINTS:
        print(f"{j:14s} {int(t[j]):6d} {int(l[j]):6d} {int(v[j]) / 10:5.1f} {int(s[j]):6d}")
    hot = {j: int(t[j]) for j in JOINTS if int(t[j]) > a.max_temp}
    err = {j: int(s[j]) for j in JOINTS if int(s[j]) != 0}
    if hot or err:
        print(f"REFUSED: too hot {hot or '-'}; status errors {err or '-'}")
        sys.exit(1)
    print("motors OK | arm untouched:", report["goal_unchanged"] and not any(report["position_drift_ticks"].values()))


if __name__ == "__main__":
    main()
