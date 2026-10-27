"""Preflight for two-arm training runs: nothing launches unless every check passes.

    python experiments/bimanual/preflight.py --run ENV_ID "TRAIN ARGS" [--run ENV_ID "TRAIN ARGS" ...]
For each run, each step in its own process (one GPU simulator per process):
  1. the env's own checks: check_carry_starts.py for carry envs, check_reset_frames.py otherwise;
  2. a short training run with the run's exact arguments, only smaller (64 envs, 20k steps, evaluation at 10k): exit 0,
     a checkpoint, and the trainer's own printout must show the expected image channels, state size, gamma and critic
     support (v_min, v_max);
  3. the stage counter loads that checkpoint on the same env and runs 16 episodes.
Prints one line per check and exits 1 if any fails. Run from the repository root on the GPU machine.
"""

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
CHANNELS = {"FirstFrame": 12, "TopCam": 9}  # anything else: 6 (left, right wrist)


def expected(env_id):
    ch = next((c for k, c in CHANNELS.items() if k in env_id), 6)
    state = 28 if "BinInState" in env_id else 24
    return ch, state


def arg_value(args, name, default):
    m = re.search(rf"--{name}[= ](\S+)", args)
    return float(m.group(1)) if m else default


def sh(cmd, log):
    with open(log, "w") as f:
        return subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=2, action="append", metavar=("ENV_ID", "TRAIN_ARGS"), required=True)
    a = ap.parse_args()
    out = ROOT / "runs" / "preflight"
    out.mkdir(parents=True, exist_ok=True)
    results = []

    def record(name, ok, detail=""):
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)

    for i, (env_id, train_args) in enumerate(a.run, start=1):
        tag = f"{i}_{env_id.replace('-', '_')}"  # numbered: two runs may share an env (B008's binstate and binstate_g090)
        ch, state = expected(env_id)
        # 1. env checks
        check = "check_carry_starts.py" if "Carry" in env_id else "check_reset_frames.py"
        rc = sh([PY, f"experiments/bimanual/{check}", env_id], out / f"{tag}_envcheck.log")
        record(f"{env_id}: {check}", rc == 0, f"log runs/preflight/{tag}_envcheck.log")
        # 2. short training with the run's own arguments
        small = re.sub(r"--(num_envs|total_timesteps|eval_freq|exp_name)[= ]\S+", "", train_args)
        name = f"preflight_{tag}"
        cmd = [PY, "train_squint.py", f"--env_id={env_id}", f"--exp_name={name}", "--num_envs=64",
               "--total_timesteps=20000", "--eval_freq=10000", *shlex.split(small)]
        log = out / f"{tag}_train.log"
        rc = sh(cmd, log)
        text = log.read_text(errors="replace")
        obs = re.search(r"Observations: \((\d+), (\d+), (\d+)\), State: (\d+)", text)
        # Explorer's A100 nodes segfault (139) at interpreter exit after the trainer's own cleanup; that one case passes
        after_cleanup = rc in (139, -11) and text.rstrip().endswith("Cleanup complete. Exiting.")
        record(f"{env_id}: short training exits 0", rc == 0 or after_cleanup,
               f"log runs/preflight/{tag}_train.log" + (f" (exit {rc} after cleanup)" if after_cleanup else ""))
        record(f"{env_id}: images 16x16x{ch}, state {state}",
               bool(obs) and (int(obs.group(3)), int(obs.group(4))) == (ch, state),
               f"trainer printed {obs.groups() if obs else 'nothing'}")
        for key, default in (("gamma", 0.9), ("v_min", -20.0), ("v_max", 20.0)):
            want = arg_value(train_args, key, default)
            got = re.search(rf"\b{key}=([-\d.e]+)", text)
            record(f"{env_id}: {key} = {want}", bool(got) and abs(float(got.group(1)) - want) < 1e-9,
                   f"trainer printed {got.group(1) if got else 'nothing'}")
        m = re.search(r"--train_env_kwargs[= ](\S+)", train_args)
        if m:  # settings for the training env only must reach the trainer as given
            want = m.group(1).strip("'\"")
            got = re.search(r"train_env_kwargs=('[^']*'|None)", text)
            record(f"{env_id}: train_env_kwargs = {want}", f"train_env_kwargs='{want}'" in text,
                   f"trainer printed {got.group(1) if got else 'nothing'}")
        ckpt = ROOT / "runs" / name / "ckpt.pt"
        record(f"{env_id}: checkpoint saved", ckpt.exists())
        # 3. the counter loads it on the same env
        if ckpt.exists():
            rc = sh([PY, "experiments/bimanual/stage_counts.py", "--env_id", env_id, "--checkpoint", str(ckpt),
                     "--num_envs", "16"], out / f"{tag}_count.log")
            record(f"{env_id}: stage counter runs the checkpoint", rc == 0, f"log runs/preflight/{tag}_count.log")
    print(f"preflight: {sum(results)} of {len(results)} passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
