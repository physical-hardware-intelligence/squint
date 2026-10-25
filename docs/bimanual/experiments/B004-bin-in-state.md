# B004: Is the stall missing information?

**Status:** protocol written 2026-10-24, before the run.

## Question
After B003 the receiver takes the bar (39 of 128) but never reaches the bin, and the bin is out of its camera for 88% of the carry. If the policy is simply told where the bin is, does the bar reach the bin?

## Intuition
The policy sees only the current wrist images and its joint angles. No memory. Once the bin leaves the picture, it has no way to know where the bin is. B004 hands it that missing fact directly. If the bar then lands in the bin, missing information was the stall.

## Change (one, from B003)
Env `SO101HandoverBarBinInState-v1`: 4 numbers added to the state input (bin x, y, cos yaw, sin yaw). Simulation only: a diagnostic, not a deployable policy. Reward, layout, settings and seed as B003.

## Method
3M steps, seed 1, evaluation every 250k; stage counts at 1.5M and 3M (128 episodes, seed 7, counter with `--env_id`).

## Predictions
1. At 3M, the bar in the bin in at least 10 of 128 episodes.
2. At 3M, success at least 5%.
3. Carries stay at least 30 of 128 (B003: 39).

## Decision rule
- Bin ≥ 10 of 128: missing information is the stall. Next (B005): a fix that works on the real rig, chosen among memory, a top camera, or a layout that keeps the bin in view.
- Carries ≥ 30 but bin < 10: information is not enough; look at the carry itself (lowering, opening).
- Carries < 10: the extra input disturbed learning; rerun with seed 2 before concluding.

## Results
_Not run._
