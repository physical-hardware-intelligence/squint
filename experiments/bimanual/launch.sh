#!/usr/bin/env bash
# Start one training run with a log that records what ran: time, commit, GPU, command, exit code.
#   experiments/bimanual/launch.sh NAME [train_squint.py args...]
# Example (Task 1, B003's settings):
#   experiments/bimanual/launch.sh handover_b003 --env_id=SO101HandoverBar-v1 --num_envs=512 \
#       --total_timesteps=3000000 --eval_freq=250000 --env_background_level 0 50 --env_background_tint 6
# Runs in the foreground: start it inside tmux. Writes runs/NAME_train.log; the run itself goes to runs/NAME/.
set -u
cd "$(dirname "$0")/../.."
NAME=$1
shift
LOG=runs/${NAME}_train.log
mkdir -p runs
COMMIT=$(git rev-parse HEAD 2>/dev/null || cat COMMIT 2>/dev/null || echo unknown)
CHANGED=$(git status --porcelain --untracked-files=no 2>/dev/null | wc -l | tr -d ' ')
{
  echo "# $(date '+%Y-%m-%dT%H:%M:%S%z') commit=$COMMIT uncommitted_files=$CHANGED"
  nvidia-smi --query-gpu=name,memory.used --format=csv,noheader 2>/dev/null
  echo "CMD: python train_squint.py --exp_name=$NAME $*"
} > "$LOG"
python train_squint.py --exp_name="$NAME" "$@" >> "$LOG" 2>&1
echo "EXIT $?" >> "$LOG"
