# B008: Can the carry be learned on its own, and what does it need?

**Status:** done 2026-10-25 16:32. Reward v2 is the main fix (in bin 87% vs v1's 17%); the top camera matches the sim-only bin position (90% vs 87%); γ 0.9 vs 0.97 shows no consistent effect; memory hurts.

## Question
In every run that hands over, the receiver then holds the bar and barely moves it (28-30 cm from the bin → 23-29 cm). The reward has measured dips after the handover, γ = 0.9 hides the bin's payoff, and the bin is in the receiver's camera only 2% of the carry ([reward-design](../reward-design.md)). Trained on the carry alone: does it work, and which of the three matters?

## Intuition
Start every episode just after the receiver has taken the bar. The handover cannot get in the way, and every episode is practice at the part that never worked. Change one thing per comparison.

## Design
Episodes start from carry starts saved from B003's two seeds at 3M (`harvest_carry_starts.py`, 512 envs × 10 episodes each): saved 5 steps into a carry and kept only if the carry was unbroken 10 steps later (see Pre-launch findings), the receiver holding the bar alone, the handover latched, each start with its episode's background colour, gripper drive and opening wrist frames. Episode 60 steps (6 s). From scratch.

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

## Pre-launch findings (2026-10-25, before any B008 training)
- **Preflight 1 failed the restore check:** with every env restored, only 51 of 64 restored carries still held the bar after 10 still steps. The earlier smoke check had passed because it counted envs without a saved start as passing (fixed: rates are now over restored envs only).
- **Restores were less stable than the originals:** under B003's own policy, 94% of original carries (188/200) were still held 10 steps after they began, against 83% (377/452) after a restore. Poses, gripper drive, background and object sizes are restored exactly; PhysX's contact cache is not saved by any state save (ManiSkill's own reset-to-state included), and the starts were saved at the instant of release, when the grip is barely held.
- **Change:** starts are now saved 5 steps into a carry and kept only if the original carry was unbroken 10 steps later. The release itself is the full task's job, not B008's.
- **Settled bank (2,764 starts, all 512 env indices covered, median 5 each), checked:** restore check 8 of 8 (holding still: 61 of 64 keep the carry). Under each harvest policy (`check_carry_fidelity.py`), restored carries are still held 11 steps later in 99% (seed 1's policy on its own starts, 498/505) and 97% (seed 2's on its own, 254/263); in fresh original episodes, 98% (257/262) of carries unbroken 5 steps in are unbroken 10 steps later. A first test that drove all starts with seed 1's policy gave 82%: each policy drops the other's carries (66% and 12%), a sign of how narrow each learned carry is, not of the restore.

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
Preflight 48 of 48 (`experiments/bimanual/out/b008_preflight.log`); all 12 runs exit 0 (commit `c1e5c41`). Stage counts at 1M, 256 episodes per run (counting seeds 7 + 8): bar ever in the bin / ever a full success. Raw: `experiments/bimanual/out/stages_b008_*.json`, `b008_counts.txt`.

| Condition | Seed 1 | Seed 2 | In bin, both seeds | Success, both seeds |
|---|---|---|---|---|
| top camera (90 cm) | 232 / 182 | 229 / 165 | 90% | 68% |
| binstate (v2, γ 0.97) | 223 / 148 | 220 / 179 | 87% | 64% |
| binstate, γ 0.9 | 135 / 118 | 218 / 168 | 69% | 56% |
| wrist | 119 / 93 | 83 / 56 | 39% | 29% |
| binstate, reward v1 | 0 / 0 | 87 / 39 | 17% | 8% |
| memory | 23 / 2 | 44 / 7 | 13% | 2% |

At 500k (in bin, seed 1 / seed 2): top 88 / 130, binstate 89 / 30, binstate γ 0.9 114 / 201, wrist 96 / 11, reward v1 1 / 0, memory 1 / 19. Every condition carries in nearly all episodes (≥ 254 of 256); the bar starts 26-27 cm from the bin.

**Predictions:**
1. Held: binstate 223 and 220 in the bin (≥ 64 each).
2. Held on both seeds: reward v1 0 and 87, below half of binstate's (111, 110). v1 is not always 0: seed 2 learned part of the carry. Over both seeds v2 puts 5× as many bars in the bin and 8× as many successes.
3. Failed on both seeds: γ 0.9 reached 135 and 218, not below half of 0.97's. Seed 1's gap (135 vs 223) is within the seed-to-seed spread (seed 2: 218 vs 220). γ 0.97 was never worse, so it stays.
4. Mixed: wrist 119 is at least half of binstate's 223 on seed 1; 83 is not on seed 2. Overall 39% vs 87%.

**Reading:**
- Reward v2 is the main fix: same inputs and γ, only the reward differs, 17% → 87% in the bin.
- The top camera (an input the real rig can have) matches the sim-only bin position, on both seeds; it is also the least seed-dependent.
- Wrist cameras alone carry to the bin in 39%, about half the ceiling, and vary most between seeds (119 vs 83).
- First-frame memory hurts here too (13%): the extra channels hurt, even with no handover to break.
- A single seed misleads: binstate seed 2 was at 30 at 500k and 220 at 1M; γ 0.9 differed by 83 between seeds.

**Decision (rule: 1 holds → the full task with what mattered; a deployable input at least half of binstate → the real-rig candidate):** the full task gets reward v2 and γ 0.97; the top camera is the real-rig candidate, wrist-only the no-hardware baseline (B010). The top camera here was 90 cm overhead; Ilia's rig has it 56-65 cm up, in front of the arms and tilted (LeHome's scene file): B009 checks that placement on the carry first.

