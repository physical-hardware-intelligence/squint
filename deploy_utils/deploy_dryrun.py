"""
Phi 2026-10-13: run deploy.py's real main() with a FAKE arm, so the whole pipeline can be tested with no hardware.

The fake arm is a perfect servo: every goal LeRobotRealAgent sends becomes the next position read. It records every
command, so we can see exactly what the real arm would have been told: the start-pose move, each policy step, the
exit fold. The camera returns a fixed real wrist frame. Keys are scripted: Enter to start one episode, then 'q'.

  --mode original   ManiSkill's own Sim2RealEnv (expected: the RecursionError seen on the arm, then the Phi guard)
  --mode fixed      deploy_utils.phi_safety.Sim2RealEnv
Usage (repo root): source env.sh && MPLBACKEND=Agg python deploy_utils/deploy_dryrun.py --checkpoint runs/<run>/ckpt.pt
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import atexit
import builtins
import json
import time
import types

import cv2
import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# the arm's resting pose read 2026-10-13 after the crash (servo degrees; gripper servo degrees)
START_DEG = {"shoulder_pan": -9.5, "shoulder_lift": -104.7, "elbow_flex": 96.7, "wrist_flex": 60.5,
             "wrist_roll": -7.7, "gripper": -66.9}


class FakeBus:
    def __init__(self):
        self.motors = {j: types.SimpleNamespace(norm_mode=None) for j in JOINTS}
        self.pos = dict(START_DEG)

    def sync_read(self, name, *args, **kwargs):
        assert name == "Present_Position", name
        return dict(self.pos)


class FakeCamera:
    """A fixed wrist image (any size): centre-cropped square, shown 480x480 inside a 640x480 frame like SimCamera."""

    def __init__(self, path):
        img = cv2.imread(path)
        if img is None:
            raise SystemExit(f"--frame {path}: not an image")
        h, w = img.shape[:2]; c = min(h, w)
        img = cv2.resize(img[(h - c) // 2:(h + c) // 2, (w - c) // 2:(w + c) // 2], (480, 480))
        self.rgb = np.pad(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), ((0, 0), (80, 80), (0, 0)))

    def async_read(self):
        return self.rgb.copy()

    def connect(self):
        pass

    def disconnect(self):
        pass


class SimCamera:
    """Closed loop: renders Squint's own sim wrist camera at the fake arm's current pose (sim-as-real). The frame is
    padded to 640x480 so deploy's centre crop gets the square render back. Also scores the reach like the sim does."""

    def __init__(self, bus, seed, env_id="SO101ReachCube-v1"):
        import gymnasium as gym, torch, envs  # noqa: F401
        self.torch, self.bus = torch, bus
        self.env = gym.make(env_id, obs_mode="rgb+segmentation", render_mode="sensors", num_envs=1,
                            domain_randomization=False, sensor_configs={"width": 480, "height": 480})
        self.env.reset(seed=seed)
        self.u = self.env.unwrapped
        self.lim = self.u.agent.robot.get_qlimits()[0].cpu()
        self.trace = []

    def _sync(self):
        d = dict(self.bus.pos)
        d["gripper"] = (d["gripper"] + 62.5) / 127.12 * 130 - 10          # manipulator.py's servo -> sim map
        q = self.torch.deg2rad(self.torch.tensor([[d[j] for j in JOINTS]], dtype=self.torch.float32))
        q = self.torch.max(self.torch.min(q, self.lim[:, 1]), self.lim[:, 0])
        self.u.agent.robot.set_qpos(q)
        self.u._update_wrist_camera_pose()

    def async_read(self):
        self._sync()
        dist = float(self.torch.linalg.norm(self.u.goal_site.pose.p - self.u.agent.tcp_pose.p, axis=1)[0])
        self.trace.append(dist)
        rgb = self.u.get_obs()["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy()
        return np.pad(rgb, ((0, 0), (80, 80), (0, 0)))

    def connect(self):
        pass

    def disconnect(self):
        pass


class FakeRobot:
    name = "so_follower"          # what lerobot 0.4.3 reports

    def __init__(self, camera="sim", seed=0, frame=None, env_id="SO101ReachCube-v1"):
        self.bus = FakeBus()
        self.cameras = {"base_camera": FakeCamera(frame) if camera == "frame" else SimCamera(self.bus, seed, env_id)}
        self.log, self.events = [], []

    def connect(self, calibrate=True):
        self.events.append("connect (torque on)")

    def disconnect(self):
        self.events.append(f"disconnect (torque off) after {len(self.log)} commands")

    def send_action(self, action):
        goal = {k.removesuffix(".pos"): float(v) for k, v in action.items()}
        prev = dict(self.bus.pos)
        self.bus.pos.update(goal)                      # perfect servo
        self.log.append({"phase": PHASE["now"], "goal": goal, "step": {j: goal[j] - prev[j] for j in goal},
                         "t": time.perf_counter()})
        return action


PHASE = {"now": "connect"}


class ScriptedKeys:
    """Enter for episode 0, nothing during its steps, then 'q'."""

    def __init__(self, steps):
        self.calls, self.steps = 0, steps

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def check_key(self):
        self.calls += 1
        if self.calls == 1:
            PHASE["now"] = "policy"
            return "\n"
        if self.calls <= self.steps + 1:
            return None
        PHASE["now"] = "after episode"
        return "q"


def summarize(robot):
    phases = {}
    for e in robot.log:
        phases.setdefault(e["phase"], []).append(e)
    out = {"events": robot.events, "commands_per_phase": {k: len(v) for k, v in phases.items()}}
    for k, v in phases.items():
        out[k] = {"first_goal": {j: round(x, 1) for j, x in v[0]["goal"].items()},
                  "last_goal": {j: round(x, 1) for j, x in v[-1]["goal"].items()},
                  "max_step_deg": {j: round(max(abs(e["step"][j]) for e in v), 2) for j in JOINTS}}
    pol = [e["t"] for e in robot.log if e["phase"] == "policy"]
    if len(pol) > 2:
        out["policy_command_rate_hz"] = round((len(pol) - 1) / (pol[-1] - pol[0]), 2)
    cam = robot.cameras["base_camera"]
    if isinstance(cam, SimCamera) and cam.trace:
        tr = cam.trace
        out["sim_in_the_loop"] = {"tcp_to_goal_m_start": round(tr[0], 3), "min": round(min(tr), 3),
                                  "at_min_frame": tr.index(min(tr)), "end": round(tr[-1], 3),
                                  "reached_<=0.02m": min(tr) <= 0.02, "frames": len(tr)}
    print("\n=== DRY RUN: what the real arm would have been told (servo degrees) ===")
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["original", "fixed"], default="fixed")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--camera", choices=["frame", "sim"], default="sim", help="sim: closed loop (default); frame: a fixed image")
    ap.add_argument("--frame", default=None, help="wrist image for --camera frame")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record-dir", default=None, help="pass --record_dir to deploy.py (flight recorder test)")
    ap.add_argument("--hold-at-end", action="store_true", help="pass --hold_at_end to deploy.py")
    ap.add_argument("--env-id", default="SO101ReachCube-v1",
                    help="task; the sim camera teleports the robot, so objects never move (pipeline test, not a grasp test)")
    a = ap.parse_args()

    import tyro
    import deploy
    from mani_skill.envs.sim2real_env import Sim2RealEnv as ManiSkillSim2RealEnv

    if a.camera == "frame" and not a.frame:
        raise SystemExit("--camera frame needs --frame <image>")
    robot = FakeRobot(a.camera, a.seed, a.frame, a.env_id)
    atexit.register(summarize, robot)                 # registered first, so it runs last
    deploy.create_real_robot = lambda: robot
    deploy.KeyboardController = lambda: ScriptedKeys(a.steps)
    builtins.input = lambda *args: ""                 # --no-continuous_eval's per-step Enter
    if a.mode == "original":
        deploy.Sim2RealEnv = ManiSkillSim2RealEnv
    print(f"dry run, mode={a.mode}, Sim2RealEnv = {deploy.Sim2RealEnv.__module__}.{deploy.Sim2RealEnv.__name__}")

    orig_reset = deploy.LeRobotRealAgent.reset

    def reset_tagged(self, qpos):
        if PHASE["now"] in ("connect",):
            PHASE["now"] = "start-pose move"
        elif PHASE["now"] == "after episode":
            PHASE["now"] = "exit fold"
        elif PHASE["now"] == "policy":
            PHASE["now"] = "episode reset"
        elif PHASE["now"] == "start-pose move":
            PHASE["now"] = "second reset (deploy main / guard)"
        orig_reset(self, qpos)
    deploy.LeRobotRealAgent.reset = reset_tagged

    args = tyro.cli(deploy.Args, args=[
        f"--checkpoint={a.checkpoint}", f"--env_id={a.env_id}",
        "--no-continuous_eval", "--control_freq=15", "--debug", f"--max_episode_steps={a.steps}"]
        + (["--hold_at_end"] if a.hold_at_end else [])
        + ([f"--record_dir={a.record_dir}"] if a.record_dir else []))
    deploy.main(args)


if __name__ == "__main__":
    main()
