# B023: Why the receiver hovers, and reward v7

**Status:** step 0 done 2026-10-27 11:44: the hover is in the wrong place (a scripted close from it grasps 8% of the time). Reward v7 pays the position; its design data, checks and runs follow as amendments.

## Why
v5's open share stopped the receiver pressing a shut gripper on the bar: that failure fell from 106 of 256 to 0 on every seed of B020 and B021. In 3 of 4 B021 seeds, though, the receiver now hovers open next to the bar instead (81-95 of 256).

Hypothesis, from reading the reward: the open share also charges a failed close at the bar. A gripper that closes and misses is a shut gripper near the bar, which costs about 1 point per step until it reopens. A real grip gains only 0.61 over a hover, so trying looks worse than waiting.

## Step 0: is hovering a choice? (no learning)
`probe_scripted_close.py` runs on B021 seeds 1, 3 and 4 (the hovering ones) and seed 2 (control), on episode seeds 7 + 8. The first time an episode hovers for 5 steps in a row, the script takes over its receiver for 10 steps: arm held still, gripper closing at the full rate. Hovering means the picker holds the bar at the handover point, and the receiver's grip point is within 3 cm of its end of the bar with its gripper at least 15% open, with no both-hold yet.

Reading:
- **A real grasp in at least half of the takeovers on the hovering seeds:** the hover is in place, and the policy chooses not to close. v7 then makes the try free near the bar. The open share charges only on the way in, fading out between 4 and 2 cm from the bar's end, so a failed close at the bar costs no more than under v2. v6's pay for the grip stays.
- **Fewer than half:** the hover is not in a position to grasp, and closing cannot help. v7 must then pay the position (jaws around the bar), not the attempt.

## Step 0 results
Run 2026-10-27 11:40-11:44 on the RTX box (tree `db051bc`). Raw: `experiments/bimanual/out/b023_step0.txt`.

| B021 at 6M, episode seeds 7 + 8 | hovers taken over | real grasp | shut on air | shut beside the bar |
|---|---|---|---|---|
| seeds 1, 3, 4 (hovering) | 250 | **21 (8%)** | 182 (73%) | 47 (19%) |
| seed 2 (control) | 27 | 11 (41%) | 6 | 10 |

**Reading (fewer than half).** The hover is not in a position to grasp: closing from it mostly closes on air. The receiver stops near the bar's end without its jaws around the bar, so making a try free would not help. v7 must pay the position, jaws around the bar, and not the attempt. The hypothesis written under Why (a failed close costs more than a grip gains) is not the main problem: closing from where it hovers would fail anyway. Seed 2 hovers far less (27 against 250 across the other three), and its hovers are closer to grippable.

## Results
_Not run._
