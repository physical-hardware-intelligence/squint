# B013: Chain the two learned skills: does the bar reach the bin?

**Status:** done 2026-10-25 22:39. **The first bars in the bin from the whole task:** with the top-camera carry policy, 71 of 256 episodes (28%) end with the bar in the bin; success at the end up to 38 (15%). Wrist-only carry: 0 and 21.

## Question
Each part has been learned, but never all of them in one run. B011's policy hands over and releases (193 and 101 of 256); B008's and B009's carry policies, trained from saved carry starts, put the bar in the bin (wrist 36-44%, top camera 93-95%). Run one after the other: how often does the whole task end with the bar in the bin?

## Intuition
Twelve experiments tried to learn the whole task in one run from scratch; every reward fix for one stage opened a shortcut at another (B002, B010, B012). Chaining uses only policies that already work, needs no training, and on the real rig needs only a switch rule.

## Method
`experiments/bimanual/chain_eval.py`, normal starts, no training:
1. The handover policy (B011 seed 1, 3M: wrist cameras, reward v2, γ 0.9) acts from the start.
2. Once the receiver has carried the bar alone for 6 steps in a row (handover latched, receiver gripping, picker's jaws off the bar: the point at which B008's saved starts were taken), the carry policy acts until the end. No switching back.

| Condition | Env | Carry policy | Its rate from saved starts (in bin, 256 episodes) |
|---|---|---|---|
| wrist | `SO101HandoverBarV2-v1` | B008 wrist s1, s2 (1M) | 112, 91 |
| top | `SO101HandoverBarV2TopCamLeHome-v1` (B009's camera) | B009 s1, s2 (1M) | 239, 242 |

The handover policy is given only the wrist images (checked against the raw cameras at the first step). 256 episodes per pair (seeds 7 + 8), outcomes at step 120 (the task's limit) and 180 (a handover late in the episode leaves the carry policy less than the 60 steps it was trained for). The switch uses the simulator's flags; on the real rig it would read the gripper joints (not tested here).

**Control:** the handover policy alone, same script and seeds: must reproduce B011's 3M counts (handovers 99 and 94, carries 51 and 50) within count noise, which checks the script.

## Predictions
1. wrist: the bar in the bin in at least 26 of 256 (10%) for at least one carry seed, by step 180 [estimate: 40% carries × 40% = 16%].
2. top: in the bin in at least 64 of 256 (25%) for at least one carry seed [estimate: 40% × 94% = 37%].
3. Of the episodes that switch, the carry policy gets the bar in at least half as often as from its saved starts (wrist ≥ 20%, top ≥ 47%). Below that, B011's handovers end in states unlike the saved ones (taken from B003's policies).

## Decision rule
- 2 holds: the chain is the route to the real rig. Next: calibrate the top camera (Monday), retrain the carry policy with it, and a switch rule from the gripper joints.
- 3 fails: harvest new carry starts from B011's handovers and retrain the carry policy on them (about 25 min), then repeat.
- Few switches (under 30% of episodes): the handover policy is the bottleneck; improve it alone.

## Results
Run 2026-10-25 22:29-22:39, commit `304f7e0`, `experiments/bimanual/chain_eval.py` (command in its docstring; `--carry` and `--env_id` per condition). Raw: `experiments/bimanual/out/b013_chains.txt`, `chain_*.json`.

**Control:** the handover policy alone gave 107 and 94 handovers, 55 and 50 carries (B011's own counts: 99 and 94, 51 and 50): the script reproduces it.

| 256 episodes (seeds 7 + 8), outcomes by step 120 | wrist: B008 s1 | wrist: B008 s2 | top: B009 s1 | top: B009 s2 | handover policy alone |
|---|---|---|---|---|---|
| handovers | 205 | 216 | 202 | 200 | 201 |
| switched to the carry policy | 102 | 114 | 100 | 106 | (105 carries) |
| **bar in the bin** | 0 | 21 | **71** | 55 | 0 |
| in the bin at step 120 | 0 | 18 | 65 | 52 | 0 |
| success, ever / at step 120 | 0 / 0 | 12 / 8 | 20 / 10 | 43 / **38** | 0 / 0 |
| of the switched, in the bin | 0% | 18% | **71%** | 52% | |
| same carry policy from its saved starts | 44% | 36% | 93% | 95% | |

The switch comes early (median step 26-28; at most 3 of 256 per pair after step 100), so 180 steps change almost nothing (in the bin 22, 71, 56; success at the end 7, 7, 39).

**Predictions.** 1 fails narrowly: wrist at most 22 of 256 by step 180 (26 needed). 2 holds: top 71 (64 needed). 3 holds for the top camera (71%, 52%; 47% needed), fails for wrist (0%, 18%; 20% needed).

**Reading.**
- The whole task works end to end in simulation for the first time, when the carry policy has the top camera: 21-28% of all episodes put the bar in the bin.
- Two losses, in order: (1) only about 40% of episodes reach a carry (the handover policy's release, as in B011); (2) the carry policies do worse on B011's handovers than on the saved starts they were trained on (from B003's policies): 95% → 52%, 93% → 71%, and one wrist policy from 44% to 0. They were trained on another policy's handovers.

**Decision (rule "2 holds").** The chain is the route to the real rig: calibrate the top camera (Monday), then retrain the carry policy with it, from carry starts harvested from the handover policy it will follow (rule "3 fails" for wrist points the same way). Then a switch rule from the gripper joints, tested in simulation against the flag-based switch used here. The handover policy's release (40% of episodes) is the next limit.
