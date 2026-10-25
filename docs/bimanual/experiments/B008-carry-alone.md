# B008: Can the carry be learned on its own, and what does it need?

**Status:** protocol written 2026-10-25, before the runs (revised before launch: reward v2, the review's fixes).

## Question
In every run that hands over, the receiver then holds the bar and barely moves it (28-30 cm from the bin → 23-29 cm). The reward has measured dips after the handover, γ = 0.9 hides the bin's payoff, and the bin is in the receiver's camera only 2% of the carry ([reward-design](../reward-design.md)). Trained on the carry alone: does it work, and which of the three matters?

## Intuition
Start every episode at the moment the receiver has just taken the bar. The handover cannot get in the way, and every episode is practice at the part that never worked. Change one thing per comparison.

## Design
Episodes start from 3,000+ carry starts saved from B003's two seeds at 3M (`harvest_carry_starts.py`, 512 envs × 10 episodes each): the receiver holds the bar alone, the handover latched, each start with its episode's background colour, gripper drive and opening wrist frames. Episode 60 steps (6 s). From scratch.

| Condition | Env | Inputs | Reward | γ | Critic support |
|---|---|---|---|---|---|
| wrist | `SO101HandoverCarry-v1` | wrist cameras (task 1's) | v2 | 0.97 | −25 … 35 |
| memory | `SO101HandoverCarryFirstFrame-v1` | + both wrist images from the original episode's first step (12 channels) | v2 | 0.97 | −25 … 35 |
| top | `SO101HandoverCarryTopCam-v1` | + a fixed camera 90 cm above the table (9 channels) | v2 | 0.97 | −25 … 35 |
| binstate | `SO101HandoverCarryBinInState-v1` | + the bin's x, y, cos yaw, sin yaw (sim only: a ceiling) | v2 | 0.97 | −25 … 35 |
| binstate_g090 | `SO101HandoverCarryBinInState-v1` | as binstate | v2 | 0.9 | −20 … 20 |
| binstate_rv1 | `SO101HandoverCarryBinInStateRewardV1-v1` | as binstate | v1 | 0.97 | −25 … 35 |

Comparisons: reward = binstate vs binstate_rv1; horizon = binstate vs binstate_g090; information = wrist, memory, top vs binstate.

Settings for every run (`train_squint.py` defaults unless listed): 512 envs, 1M steps, evaluation every 250k on 16 envs, `--eval_reconfiguration_freq 0` (a rebuilt scene would not fit the saved starts; the env refuses), backgrounds 0-50 ±6, colour jitter on (the same jitter for a camera's current and kept frames), images 16 × 16, batch 512, 256 updates per iteration, learning starts 5,000, buffer 1M, lr 3e-4, τ 0.01, 2 critics, 101 atoms, entropy auto-tuned. Seeds 1 and 2; all seed-1 runs first. Stage counts at 500k and 1M (seeds 7 and 8, 256 episodes), with the bar-to-bin distance during the carry. Box queue `runs/b008_queue.sh`, counts to `runs/b008_counts.txt`.

## Preflight (must pass before any run starts)
`experiments/bimanual/preflight.py`, per condition:
- the restore check (`check_carry_starts.py`): bar pose restored exactly, gripper drive and background restored, carries after one still step, no slip while still, reset frame = fresh render, the condition's inputs;
- a 20k-step training run with the condition's exact arguments: exit 0; channels, state size, γ and support as printed by the trainer;
- the counter running that checkpoint.

## Predictions
1. binstate: bar in the bin in at least 64 of 256 (25%) at 1M. If not, the carry fails for a reason other than reward dips, horizon or information.
2. Reward matters: binstate_rv1 below half of binstate's in-bin count.
3. Horizon matters: binstate_g090 below half of binstate's.
4. Seeing the bin adds little: wrist at least half of binstate's (blind placement lands in 62% of layouts).

## Decision rule
- 1 fails: look at the carry itself in videos (grip, reach, lowering); the reward and γ are not enough.
- 1 holds: train the full task with whichever of v2 and γ 0.97 mattered (2, 3).
- 4 holds: the real rig needs no new camera for the carry. 4 fails, and memory or top reaches at least half of binstate: that input is the real-rig candidate; next, how to add it without breaking the handover.

## Results
_Not run._
