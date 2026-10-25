# B004: Is the stall missing information?

**Status:** done 2026-10-24 23:31, inconclusive: neither seed released (0 carries in both), so the bin's position was never tested. Next: a warm start from B003 (proposed, not run).

## Question
After B003 the receiver takes the bar (39 of 128) but never reaches the bin, and the bin is out of its camera for 88% of the carry. If the policy is simply told where the bin is, does the bar reach the bin?

_(Protocol text as written. B003 A2 later refined the 88%: with occlusion, the receiver's camera sees the bin in 2% of carry steps, the picker's in 40%. The question stands.)_

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
**Run 1** (commit `33a2ad9`, seed 1, `runs/handover_b004_2026-10-24`): exit 0, about 1,050 steps/s. Evaluation returns 250k-3M: 18.3, 27.9, 29.5, 36.9, 38.5, 34.1, 38.6, 37.8, 36.1, 38.9, 31.1, 38.9; success 0.

| Stage (of 128) | B003 1.5M | B004 1.5M | B003 3M | B004 3M |
|---|---|---|---|---|
| picker grasps | 128 | 127 | 128 | 128 |
| both hold (mid-air handover) | 33 | 6 | 48 | 21 |
| receiver carries | 1 | 0 | 39 | **0** |
| bar in bin | 0 | 0 | 0 | 0 |

**Reading:** the bin's position can only matter after the release, and run 1 never released. So this run says nothing about the question. It does show that two runs differing by one input can diverge before the handover (48 vs 21 handovers): with one seed per arm, run-to-run spread is unknown and may be large.

**Decision (rule: carries < 10, rerun with seed 2 before concluding):** run 2, seed 2, otherwise identical.

**Run 2** (commit `4870946`, seed 2, `runs/handover_b004s2_2026-10-24`): exit 0. Evaluation returns 250k-3M: 11.5, 28.4, 32.4, 33.3, 33.9, 37.1, 38.4, 33.2, 35.7, 38.9, 33.8, 39.5; success 0.

| Stage (of 128) | B003 3M (seed 1) | B004 run 1 3M (seed 1) | B004 run 2 3M (seed 2) |
|---|---|---|---|
| mid-air handover | 48 | 21 | 35 |
| both-hold steps per handover | about 2 | about 7 | about 20 |
| receiver carries | 39 | 0 | **0** |
| bar in bin | 0 | 0 | 0 |

**Reading:** run 2 behaves like B002: both arms keep holding (about 20 steps per handover) and the picker never lets go. Neither B004 seed released, so the question (does knowing the bin's position get the bar there?) is still untested. Two explanations fit and these runs cannot separate them: B003's release was a lucky seed, or the 4 extra inputs slow the release. Both say the same thing for the method: **from scratch, whether the release is learned at all varies more between runs than the effect we want to measure.**

**Decision (rule followed: seed 2 run, still < 10 carries):** stop testing from scratch. Proposed next, as its own protocol: start from B003's 3M checkpoint, which already carries in 39 of 128, and compare "B003 continued" with "B003 + bin position" (the 4 new inputs start at zero weight, so both begin as the same policy). Separately, B003 with seed 2 measures whether its release reproduces.
