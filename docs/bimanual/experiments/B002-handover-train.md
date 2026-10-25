# B002: Does Squint learn the handover?

**Status:** stopped at 1.5M steps, 2026-10-24 20:47 (plateau; amendment A2). Stall found at the release.

## Question
Does Squint, unchanged, learn `SO101HandoverBar-v1` from scratch in simulation, and where does it stall if it does not?

## Method
- `train_squint.py` on the 4080 box, from a copy of this branch (the box's own checkout is not touched).
- Settings: PlaceCube's recipe (domain randomization on, backgrounds 0-50 ±6, colour jitter), 512 envs (two cameras per env), 10M steps, evaluation every 250k steps on 16 envs, seed 1. Episode 120 steps.
- First a smoke test: 64 envs, 50k steps, to check it runs on the GPU simulator.
- After training: run the final checkpoint on 64 evaluation episodes and count each stage (picker grasps, both hold, handed over, bar in bin, success).

## Predictions
Pure RL on a long, three-stage task, with no curriculum:
1. The picker learns to grasp and lift (in at least 80% of episodes) by 3M steps.
2. Mid-air handovers happen in fewer than 20% of episodes at 10M steps.
3. Success stays below 10% at 10M steps.
4. Throughput 800-1,200 steps/s, so about 2.5-3.5 h [estimate; PlaceCube ran about 2,200 steps/s with one arm and one camera].

## Decision rule
- Success at least 50%: evaluate at deploy settings, then plan the real run.
- 10-50%: continue to 30M steps.
- Below 10%: add a curriculum or reward change aimed at the stage where the counts stall.

## Amendments
- **A1, 2026-10-24, before the run.** Pre-flight added: `experiments/bimanual/check_reward.py` checks the reward stages and the success flag on constructed states, since B001 never got past the grasp. Result: 16 of 16 passed (bands 0.5, 4.2, 6.5, 7.7, 12.2, 15 in order; normalized maximum 1; the handover latch; the bin test lying, standing, outside, and on a rotated bin).

- **A2, 2026-10-24 20:47, during the run.** Stopped at 1.5M of 10M steps: evaluation return flat for 750k steps (40.7, 41.8, 41.8, 38.8) with 0% success. Continuing two more hours on a plateau was judged less informative than counting the stages. A later breakthrough cannot be excluded. Checkpoint kept (`ckpt_1.5M.pt`).

## Results
**Smoke test** (2026-10-24, commit `3d7dec9`): 64 envs, 50k steps in 6 min, exit 0, checkpoint and 3 evaluation videos saved; evaluation return −53.6 at step 0, +2.7 at 50k. The first launch (commit `57284e2`) crashed at the action size: the trainer read the two-arm Dict action space from the unwrapped env. Fixed in `train_squint.py` (`single_action_box`); single-arm training unchanged.

**Main run** (commit `3d7dec9`, `runs/handover_b002_2026-10-24` on the box): 512 envs at about 1,140 steps/s, 8.6 GB of GPU memory; about 2.4 h for 10M steps.

| Evaluation at | 0 | 250k | 500k | 750k | 1M | 1.25M | 1.5M |
|---|---|---|---|---|---|---|---|
| Return | −53.6 | 25.4 | 32.1 | 40.7 | 41.8 | 41.8 | 38.8 |
| Success | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**Stage counts** (`experiments/bimanual/stage_counts.py`, 1.5M checkpoint, 128 episodes, seed 7):

| Stage | Episodes |
|---|---|
| picker grasps | 127 |
| bar at the handover point, receiver within 3 cm of it | 124 |
| both hold the lifted bar (mid-air handover) | 17 |
| picker lets go, receiver carries | 0 |
| bar in bin, success | 0, 0 |

**Predictions:** 1 held (127/128 grasp by 1.5M). 2 consistent (13% handovers at 1.5M). 3 held (0%). 4 held (about 1,050-1,140 steps/s).

**Diagnosis:** the stall is the release. In the reward, letting go raised the reward from about 6.9 to 7.3 (stage 4 tops out at 7; stage 5 started at 7 plus a small distance term), while a drop falls back to about 2. Keeping both grippers on the bar was the safe optimum.

**Decision (rule: below 10%, change the stage that stalls):** B003, a clean release pays a full point more.

