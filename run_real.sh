#!/usr/bin/env bash
# Run our trained Squint models on the real SO-101 with every pre-flight check (Phi, 2026-10-19).
#
#   ./run_real.sh camera          once per session, no run active: re-apply the camera lock (the camera forgets it
#                                 on every power cycle). No arm motion. Opens a real-vs-sim picture
#   ./run_real.sh P               servo check, camera-lock check, torque-on hold test, then model P (PlaceCube).
#                                 A and B: ReachCube. Recorded to runs/<run>/real_eval_<date>/
#   ./run_real.sh camera-start    in a second window while a run waits at the start pose: is the exposure right?
#   ./run_real.sh camera-sweep    same place, if it is not: find new settings and save a new lock
#   ./run_real.sh dry P           the whole run on a fake arm, no hardware
#   ./run_real.sh check           servo health and camera lock only, no motion
#   ./run_real.sh verdict [FILE]  judge a start-pose camera report (default: the newest)
#
# Use a terminal with Camera permission (Terminal.app). During a run: Enter once starts an episode; at its end the
# arm holds its last pose and Enter returns it to the start pose; s skips; q or Ctrl-C folds the arm, torque off.
# Setup, and the manual commands behind each step: docs/so101-phi.md.
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
PY=.venv/bin/python
LOCK=${SQUINT_CAMERA_LOCK:-tune_snapshots/20261014_camera_lock_startpose.json}

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print } /^set -euo/ { exit }' "$0"; exit "${1:-0}"; }

model() {
  case "${1:-}" in
    A) RUN=reach_camtuned_2026-10-13;        TASK=SO101ReachCube-v1; STEPS=60 ;;
    B) RUN=reach_camtuned_bgrand_2026-10-14; TASK=SO101ReachCube-v1; STEPS=60 ;;
    P) RUN=place_camtuned_bgrand_2026-10-14; TASK=SO101PlaceCube-v1; STEPS=250 ;;
    *) echo "which model? A, B or P"; usage 2 ;;
  esac
  CKPT=runs/$RUN/ckpt.pt
  [ -f "$CKPT" ] || { echo "missing $CKPT: copy it from the training machine (docs/so101-phi.md)"; exit 1; }
}

need_uvcc() {   # camera_match.py reads and writes the camera's settings through uvcc
  if ${SQUINT_UVCC:-uvcc} --version >/dev/null 2>&1; then return; fi
  local node=node alt="$HOME/tools/uvcc/node_modules/.bin/uvcc"
  [ -x /opt/homebrew/opt/node@24/bin/node ] && node=/opt/homebrew/opt/node@24/bin/node
  if [ -e "$alt" ] && "$node" "$alt" --version >/dev/null 2>&1; then export SQUINT_UVCC="$node $alt"; return; fi
  echo "uvcc does not run (SQUINT_UVCC=${SQUINT_UVCC:-uvcc}). Install it once (Node.js), then run this again:"
  echo "  mkdir -p ~/tools/uvcc && cd ~/tools/uvcc && npm install uvcc@8.0.0"
  exit 1
}

port_free() {
  if [ ! -e "${SQUINT_PORT:?set SQUINT_PORT in rig.local.sh}" ]; then
    echo "no arm at $SQUINT_PORT: plug in its USB and power. Ports now: $(ls /dev/tty.usbmodem* 2>/dev/null || echo none)"
    exit 1
  fi
  if lsof "$SQUINT_PORT" >/dev/null 2>&1; then
    echo "the arm's port is open in another program (is a run still going?): $SQUINT_PORT"; exit 1
  fi
}

check() {
  need_uvcc; port_free
  $PY deploy_utils/motor_check.py
  $PY deploy_utils/camera_match.py --check "$LOCK" \
    || { echo "the camera is not locked (it forgets after every power cycle): ./run_real.sh camera"; exit 1; }
}

verdict() {   # a start-pose report against what the policy was trained on
  $PY - "${1:-}" <<'EOF'
import glob, json, os, sys
f = sys.argv[1] or max(glob.glob("tune_snapshots/*_camera_match.json"), key=os.path.getmtime)
d = json.load(open(f))
c = d.get("after") or d.get("applied") or d["as_found"]   # after a sweep or --apply
jaws, bg, clip = (c["real"]["robot"] or {}).get("level"), c["background"], c["clipped_pct"]
# Start pose. Training's light gives jaws 140-196 (envs/base_random_env.py: ambient 0.2-0.5 per channel), widened by
# a +-30% brightness jitter (utils.py ColorJitterWrapper); our 14 Oct lock read 127 and placed 5/5. Background 0-50.
print(f"\n{os.path.basename(f)}: jaws {jaws} (want 120-200) | background {bg} (want <= 50) | clipped {clip}% (want <= 1)")
if d["pose"] != "start":
    print("  -> not judged: measured with the arm folded, where the sim hides the jaws. Use camera-start during a run")
elif jaws is not None and 120 <= jaws <= 200 and bg <= 50 and clip <= 1:
    print("  -> EXPOSURE OK")
else:
    print("  -> NOT OK: ./run_real.sh camera-sweep in this window (the run keeps holding the start pose)")
EOF
}

case "${1:-}" in
  camera)
    need_uvcc; port_free
    $PY deploy_utils/camera_match.py --apply "$LOCK"
    open "$(ls -t tune_snapshots/*_camera_match_found.png | head -1)" 2>/dev/null || true
    echo "The picture should show the jaws at the bottom (columns: real, sim, difference). With the arm folded the"
    echo "brightness numbers mean little; check them at the start pose: ./run_real.sh camera-start during the first run." ;;
  camera-start)
    need_uvcc
    $PY deploy_utils/camera_match.py --pose start
    verdict ;;
  camera-sweep)
    need_uvcc
    $PY deploy_utils/camera_match.py --pose start --sweep
    verdict
    new=$(ls -t tune_snapshots/*_camera_lock.json | head -1) || true
    echo "new lock: $new. To check later runs against it, add to rig.local.sh:  export SQUINT_CAMERA_LOCK=$new" ;;
  check) check ;;
  verdict) verdict "${2:-}" ;;
  dry)
    model "${2:-}"
    MPLBACKEND=Agg $PY deploy_utils/deploy_dryrun.py --checkpoint "$CKPT" --env-id "$TASK" --steps "$STEPS" --hold-at-end ;;
  A|B|P)
    model "$1"
    check
    $PY deploy_utils/hold_test.py
    OUT=runs/$RUN/real_eval_$(date +%F)
    echo "recording to $OUT. Hands out of the camera's view until the arm stops."
    echo "First run this session? While the arm waits at the start pose: ./run_real.sh camera-start in a second window."
    $PY deploy.py --checkpoint "$CKPT" --env_id "$TASK" --control_freq 15 --max_episode_steps "$STEPS" \
      --hold_at_end --record_dir "$OUT" ;;
  ""|-h|--help|help) usage ;;
  *) echo "unknown: $1"; usage 2 ;;
esac
