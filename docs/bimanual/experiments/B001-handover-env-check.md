# B001: Is the handover doable in simulation?

**Status:** done 2026-10-24: 0 of 10 in both runs. The layout checks out; my scripted grasp does not (see Conclusion).

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

- **A2, 2026-10-24, after run 1, before run 2.** Run 1's strategy cannot work (see Results): the oracle now presents the bar upright at the handover point (picker's fingers horizontal, free end up), the receiver grasps the top end from above, and the receiver lowers the hanging bar into the bin until its bottom is 4 mm above the floor before letting go. The env, the seeds and the predictions are unchanged.

## Results
**Run 1** (2026-10-24, commit `b9bad97`, `experiments/bimanual/out/B001_20261024-195423.json`): **0 of 10.** Prediction 1 fails; decision rule: diagnose before training.

| Failure | Runs | Cause |
|---|---|---|
| arms collide at the handover | 10 | both grippers came from above, their centres 3.6 cm apart along the bar; the gripper bodies are wider than that |
| a gripper touches the bin | 10 | lowering the bar flat brings the fingers to the walls |
| the picker never grasps | 1 (seed 2) | not yet diagnosed |

The env is not at fault: these are the oracle's choices. Hence A2.

**Run 2** (A2, commit `3de31bd`, `experiments/bimanual/out/B001_20261024-195633.json`): **0 of 10.** The picker never holds the bar through the lift; an arm touches the table in all 10.

**Diagnosis** (exploratory, scratch scripts, 2026-10-24):
- Joint solutions are exact (error ≤ 5 mm) and the jaws do straddle the bar's width when closing. But a wide opening (45°) makes the moving jaw push the bar 15-25 mm before clamping, and pushed near its end the bar turns out of the jaws. Narrow openings (8-18°) land a jaw on the bar's top. 0 of 7 grasps held in every one of 24 opening, offset and height combinations.
- The trained PlaceCube policy grasps the same 25 mm section from straight above in 8 of 8 sim runs, so the simulator can grasp it: what fails is my scripted grasp, not the env.

## Conclusion
- **Checked:** both arms reach the handover point, and only the receiver reaches the bin (reach ≤ 43 cm, sampled). Two top-down grippers on a 60 mm bar collide. Presenting the bar upright solves the joints (error ≤ 3 mm).
- **Not established:** that a full handover works in this simulator. Scripted grasping of the bar does not work, and fixing it means modelling the jaws' real contact geometry.
- **Decision:** test feasibility with learning instead (B002). A curriculum from scripted mid-task states waits for a working scripted grasp.

