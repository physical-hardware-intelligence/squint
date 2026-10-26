# B013: Chain the two learned skills: does the bar reach the bin?

**Status:** protocol written 2026-10-25, before the runs.

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
_Not run._
