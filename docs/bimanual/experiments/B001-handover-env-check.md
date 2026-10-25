# B001: Is the handover doable in simulation?

**Status:** protocol written 2026-10-24, before the counted run.

## Question
Can a scripted controller (inverse kinematics, top-down grasps) complete `SO101HandoverBar-v1`, and do the reward stages and the success flag fire in the intended order?

## Why
A training run costs about an hour of GPU. If the task is impossible (reach, collisions, gripper geometry) or the success flag is wrong, Squint fails for reasons we cannot see.

## Method
- `experiments/bimanual/handover_oracle.py`, CPU simulation, one env, domain randomization off, seeds 1-10 (both picker sides occur).
- The picker grasps its end of the bar from above, lifts, and brings it to the handover point (17 cm in front of the pan axes, centred, 12 cm up) with the free end toward the receiver. The receiver grasps the free end. The picker opens and returns to its start pose. The receiver carries the bar over the bin along its long side, lowers it, opens, and returns.

## Predictions
1. At least 8 of 10 layouts succeed.
2. In every success the flags first fire in this order: `picker_grasp`, `both_hold`, `handed_over`, `bar_in_bin`, `success`.
3. No arm-arm or table contact in a successful run.

## Decision rule
- 8 or more of 10: the task is feasible as specified; go to B002 (training).
- Fewer: classify each failure (reach, collision, grasp, flag bug) and fix the layout or the env before training.

## Amendments
- **A1, 2026-10-24, before the counted run.** A first try (scratch script, seeds 1 and 3) never grasped: it assumed the jaws close along the gripper's y axis. Measured in sim, they close along x (fingertip to fingertip (0.28, −0.96, 0.10) against gripper x (0.29, −0.96, 0)). Fixed in the oracle; the env is unchanged. The TCP now stops 6 mm above the bar's centre: the URDF fingertips sit 4.6-12 mm short of the jaws' real ends (real2sim2real gap ledger G18), and seed 3 touched the table.

## Results
_Not run._
