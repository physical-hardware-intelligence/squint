"""
Match the real wrist camera to the sim at the same pose, and lock the camera's image settings (Phi, 2026-10-14).

The policy was trained on the sim's image at each joint angle, so this measures the real image against exactly
that, region by region, with the sim's own segmentation (each region shrunk 2 px at 128x128 so small pose errors
do not mix them):
  robot        gripper, jaws, wrist: mean R, G, B real vs sim; "bright" = the parts the sim renders above 100
  background   everything the greenscreen replaces (sim: black; checkpoint B trained on solid 0-50)
  clipped      % of real pixels at 250+ in any channel, per single frame
  policy diff  mean |real - sim| on the 16x16 the network sees

No arm motion. --pose real (default) reads the joints through tune_camera_nomotion.ReadOnlyArm (no writes, no
torque) and checks at exit that nothing changed. --pose start renders the task's start keyframe and never opens
the arm's port (use it while deploy.py holds the arm there).

  (no flag)      measure with the camera's current settings
  --apply FILE   apply saved settings (manual modes first), then measure
  --sweep        exposure x gain, then white balance, then fine gain: apply the closest match and save it
  --check FILE   exit 1 unless the camera's settings equal FILE (a pre-flight before deploy.py)

The camera's settings are read and written with uvcc (https://github.com/joelpurra/uvcc, not bundled):
SQUINT_UVCC = the command, SQUINT_UVC_VENDOR / SQUINT_UVC_PRODUCT = the camera's USB ids (rig.local.sh).
Run in a Terminal with Camera permission (repo root): source env.sh && python deploy_utils/camera_match.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import gymnasium as gym
import envs  # noqa: F401
from mani_skill.utils.structs import Pose

ROOT = Path(__file__).resolve().parent.parent
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
UVCC = (os.environ.get("SQUINT_UVCC", "uvcc").split()
        + ["--vendor", os.environ.get("SQUINT_UVC_VENDOR", ""), "--product", os.environ.get("SQUINT_UVC_PRODUCT", "")])
EXPOSURES = [333, 313, 250, 200, 156, 125, 100]     # 0.1 ms units; above 333 the camera drops below 30 fps
GAINS = [0, 8, 16, 22, 28, 34]                      # the black floor lifts past ~32 (2026-10-13 sweep)
WBS = [2800, 3400, 4000, 4600, 5200, 5800, 6400]    # kelvin
BG_OK = 10.0                                        # background level the score tolerates (A trained on ~0)


def uvcc(*args):
    r = subprocess.run(UVCC + [str(a) for a in args], capture_output=True, text=True, timeout=15)
    if r.returncode != 0:
        raise RuntimeError(f"uvcc {' '.join(map(str, args))}: {(r.stderr or r.stdout).strip()}")
    return r.stdout.strip()


def read_settings():
    return json.loads(uvcc("export"))


def apply_settings(s):
    """Manual modes first: the camera refuses exposure writes while auto exposure is on."""
    uvcc("set", "auto_exposure_mode", 1)
    uvcc("set", "auto_white_balance_temperature", 0)
    for k, v in s.items():
        if k not in ("auto_exposure_mode", "auto_white_balance_temperature"):
            uvcc("set", k, v)


class Camera:
    """deploy.py's image path: 640x480, optional 180 deg rotation, centre crop, resize to 128x128 (RGB)."""

    def __init__(self, index, rotate_180):
        self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            raise SystemExit("could not open the camera: check SQUINT_CAMERA and this Terminal's Camera permission")
        self.rotate_180 = rotate_180

    def _to128(self, bgr):
        if self.rotate_180:
            bgr = cv2.rotate(bgr, cv2.ROTATE_180)
        h, w = bgr.shape[:2]
        c = min(h, w)
        y0, x0 = (h - c) // 2, (w - c) // 2
        return cv2.cvtColor(cv2.resize(bgr[y0:y0 + c, x0:x0 + c], (128, 128)), cv2.COLOR_BGR2RGB)

    def grab(self, settle=0.8, n=5):
        """Mean of n frames after `settle` s of discarded frames (a setting takes a few frames to apply)."""
        t = time.time()
        while time.time() - t < settle:
            self.cap.read()
        frames = []
        while len(frames) < n:
            ok, bgr = self.cap.read()
            if ok:
                frames.append(self._to128(bgr))
        clipped = float(np.mean([(f >= 250).any(2).mean() for f in frames]) * 100)
        return np.mean(frames, 0).astype(np.float32), clipped

    def close(self):
        self.cap.release()


def sim_view(qpos_rad, cube_xy_pan=None):
    """Sim image + region masks at these joint angles. cube_xy_pan: cube centre in metres from the pan axis."""
    env = gym.make("SO101ReachCube-v1", obs_mode="rgb+segmentation", render_mode="sensors", num_envs=1,
                   domain_randomization=False, sensor_configs={"width": 128, "height": 128})
    env.reset(seed=0)
    u = env.unwrapped
    half = float(u.item_half_sizes[0])
    pan_x = float(u.agent.robot.links_map["shoulder_link"].pose.p[0, 0])
    p = [[5.0, 5.0, half]] if cube_xy_pan is None else [[cube_xy_pan[0] + pan_x, cube_xy_pan[1], half]]
    u.item.set_pose(Pose.create_from_pq(p=p))
    robot = u.agent.robot
    lim = robot.get_qlimits()[0].cpu()
    q = torch.tensor(qpos_rad, dtype=torch.float32)
    qc = torch.max(torch.min(q, lim[:, 1]), lim[:, 0])
    clamped = {j: round(float(np.degrees(q[i] - qc[i])), 1) for i, j in enumerate(JOINTS) if abs(float(q[i] - qc[i])) > 1e-6}
    robot.set_qpos(qc[None])
    u._update_wrist_camera_pose()
    cam = u.get_obs()["sensor_data"]["base_camera"]
    rgb = cam["rgb"][0].cpu().numpy().astype(np.float32)
    seg = cam["segmentation"][0, ..., 0].cpu().numpy()
    keep = np.isin(seg, u._segmentation_ids_to_keep.cpu().numpy())
    cube = np.isin(seg, u.item.per_scene_id.cpu().numpy())
    k = np.ones((5, 5), np.uint8)
    shrink = lambda m: cv2.erode(m.astype(np.uint8), k).astype(bool)
    robot_m = shrink(keep & ~cube)
    masks = {"robot": robot_m, "bright": robot_m & (rgb.mean(2) > 100), "cube": shrink(cube), "background": shrink(~keep)}
    env.close()
    return rgb, masks, clamped


def region(img, m):
    if m.sum() < 20:
        return None
    px = img[m]
    mean = px.mean(0)
    return {"pixels": int(m.sum()), "rgb": [round(float(v), 1) for v in mean], "level": round(float(mean.mean()), 1),
            "cast": round(float(((px[:, 0] - px[:, 2]) / (px.sum(1) + 1e-6)).mean()), 3)}


def compare(real, clipped, sim, masks):
    r = {name: region(real, m) for name, m in masks.items()}
    s = {name: region(sim, m) for name, m in masks.items()}
    to16 = lambda x: cv2.resize(x, (16, 16), interpolation=cv2.INTER_AREA)
    out = {"real": r, "sim": s, "clipped_pct": round(clipped, 2),
           "policy_diff_16": round(float(np.abs(to16(real) - to16(sim)).mean()), 1)}
    b_real, b_sim = r["bright"] or r["robot"], s["bright"] or s["robot"]
    out["bright_ratio"] = round(b_real["level"] / max(b_sim["level"], 1e-6), 3)
    out["bright_cast_diff"] = round(b_real["cast"] - b_sim["cast"], 3)
    out["background"] = r["background"]["level"] if r["background"] else None
    out["score"] = round(100 * abs(np.log(max(out["bright_ratio"], 1e-6))) + 100 * abs(out["bright_cast_diff"])
                         + 20 * clipped + 2 * max(0.0, (out["background"] or 0) - BG_OK), 1)
    return out


def line(tag, c):
    return (f"{tag:34s} bright {c['real']['bright']['level'] if c['real']['bright'] else '-':>6} vs sim "
            f"{c['sim']['bright']['level'] if c['sim']['bright'] else '-':>6} (x{c['bright_ratio']:.2f}) | cast diff "
            f"{c['bright_cast_diff']:+.3f} | background {c['background']} | clipped {c['clipped_pct']:.1f}% | "
            f"policy diff {c['policy_diff_16']} | score {c['score']}")


def label(kv):
    short = {"absolute_exposure_time": "exp", "gain": "gain", "white_balance_temperature": "wb"}
    return " ".join(f"{short.get(k, k)} {v}" for k, v in kv.items())


def picture(path, real, sim):
    up = lambda x, n=256: cv2.resize(np.clip(x, 0, 255).astype(np.uint8), (n, n), interpolation=cv2.INTER_NEAREST)
    to16 = lambda x: cv2.resize(x, (16, 16), interpolation=cv2.INTER_AREA)
    diff = np.abs(real - sim)
    rows = np.vstack([np.hstack([up(real), up(sim), up(diff)]), np.hstack([up(to16(real)), up(to16(sim)), up(to16(diff))])])
    cv2.imwrite(str(path), cv2.cvtColor(rows, cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser(description="match the real wrist camera to the sim at the same pose")
    ap.add_argument("--pose", choices=["real", "start"], default="real")
    ap.add_argument("--cube-xy", nargs=2, type=float, default=None, metavar=("X_CM", "Y_CM"),
                    help="cube centre in cm from the pan axis, if the cube is in view")
    ap.add_argument("--apply", default=None)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--check", default=None)
    ap.add_argument("--out", default=str(ROOT / "tune_snapshots"))
    a = ap.parse_args()

    if a.check:
        want, now = json.load(open(a.check)), read_settings()
        diff = {k: (now.get(k), v) for k, v in want.items() if now.get(k) != v}
        print("camera settings match the lock" if not diff else f"camera settings differ from the lock (now, lock): {diff}")
        sys.exit(1 if diff else 0)

    found = read_settings()
    print("camera settings as found:", found)
    arm, report = None, None
    if a.pose == "real":
        from deploy_utils.tune_camera_nomotion import ReadOnlyArm
        arm = ReadOnlyArm()
        deg, _ = arm.qpos_deg()
        qpos = [np.radians(deg[j]) for j in JOINTS]
        print("arm (read-only, deg; gripper in sim deg):", {j: round(deg[j], 1) for j in JOINTS})
    else:
        from envs.robot.so101 import SO101
        qpos = SO101.keyframes["start"].qpos.tolist()
    try:
        cube = None if a.cube_xy is None else (a.cube_xy[0] / 100, a.cube_xy[1] / 100)
        sim, masks, clamped = sim_view(qpos, cube)
        print("sim at the same pose; clamped to sim limits (deg):", clamped or "none",
              "| region pixels:", {k: int(m.sum()) for k, m in masks.items()})
        cam_env = os.environ.get("SQUINT_CAMERA", "0")
        cam = Camera(int(cam_env) if cam_env.isdigit() else cam_env, os.environ.get("SQUINT_CAMERA_ROTATION", "0") == "180")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        results = {"pose": a.pose, "qpos_deg": [round(float(np.degrees(v)), 2) for v in qpos], "found": found}

        real, clipped = cam.grab()
        c = compare(real, clipped, sim, masks)
        print(line("as found", c))
        results["as_found"] = c
        picture(out / f"{stamp}_camera_match_found.png", real, sim)

        if a.apply:
            apply_settings(json.load(open(a.apply)))
            real, clipped = cam.grab()
            c = compare(real, clipped, sim, masks)
            print(line(f"applied {Path(a.apply).name}", c))
            results["applied"] = {"file": a.apply, **c}

        if a.sweep:
            apply_settings({"power_line_frequency": 2, "white_balance_temperature": found.get("white_balance_temperature", 3400)})
            trials = []

            def trial(**kv):
                apply_settings(kv)
                img, cl = cam.grab()
                cc = compare(img, cl, sim, masks)
                trials.append({**kv, **{k: cc[k] for k in ("bright_ratio", "bright_cast_diff", "background", "clipped_pct", "policy_diff_16", "score")}})
                print(line(label(kv), cc))
                return cc, img

            wb = found.get("white_balance_temperature", 3400)
            best = None
            for e in EXPOSURES:
                for g in GAINS:
                    cc, img = trial(absolute_exposure_time=e, gain=g, white_balance_temperature=wb)
                    if best is None or cc["score"] < best[0]["score"]:
                        best = (cc, img, dict(absolute_exposure_time=e, gain=g, white_balance_temperature=wb))
            for w in WBS:
                kv = dict(best[2], white_balance_temperature=w)
                cc, img = trial(**kv)
                if cc["score"] < best[0]["score"]:
                    best = (cc, img, kv)
            for dg in (-4, -2, 2, 4):
                kv = dict(best[2], gain=int(np.clip(best[2]["gain"] + dg, 0, 63)))
                cc, img = trial(**kv)
                if cc["score"] < best[0]["score"]:
                    best = (cc, img, kv)
            apply_settings(best[2])
            lock = read_settings()
            lock_path = out / f"{stamp}_camera_lock.json"
            lock_path.write_text(json.dumps(lock, indent=2))
            real, clipped = cam.grab()
            c = compare(real, clipped, sim, masks)
            print(line("CHOSEN " + label(best[2]), c))
            print(f"lock saved: {lock_path}")
            results.update(sweep=trials, chosen=best[2], after=c, lock=str(lock_path))
            picture(out / f"{stamp}_camera_match_locked.png", real, sim)
        cam.close()
        (out / f"{stamp}_camera_match.json").write_text(json.dumps(results, indent=1))
        print(f"report: {out / f'{stamp}_camera_match.json'} | pictures: {out}/{stamp}_camera_match_*.png "
              "(rows: 128x128, 16x16; columns: real, sim, |difference|)")
    finally:
        if arm is not None:
            report = arm.close()
            print("arm untouched:", report)


if __name__ == "__main__":
    main()
