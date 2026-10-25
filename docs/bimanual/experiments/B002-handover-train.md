# B002: Does Squint learn the handover?

**Status:** protocol written 2026-10-24, before the run.

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

## Results
_Not run._
