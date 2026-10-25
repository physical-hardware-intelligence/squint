# B005: From a policy that already lets go, does knowing the bin's position get the bar in?

**Status:** protocol written 2026-10-24, before the run.

## Question
B004 asked this from scratch and could not answer: neither seed learned to let go. Here both arms start from B003's 3M checkpoint, which already carries in 39 of 128 episodes, so the comparison begins after the release.

## Intuition
Two copies of the same policy keep training. One is also told where the bin is; the other is not. If only the told one starts landing the bar in the bin, missing information was the stall.

## Design
| Run | Env | Start | Change |
|---|---|---|---|
| control | `SO101HandoverBar-v1` | B003 `ckpt_3M.pt` | none (B003 continued) |
| bin | `SO101HandoverBarBinInState-v1` | B003 `ckpt_3M.pt` widened by `experiments/bimanual/widen_state_input.py` (4 new inputs, zero weights) | the bin's x, y, cos yaw, sin yaw in the state |

Both: 1.5M more steps, seed 1, B003's settings, evaluation every 250k; stage counts at 750k and 1.5M (128 episodes, seed 7). Each starts with a fresh replay buffer and optimizer and 5,000 steps of random actions (`learning_starts`), the same for both. Run one after the other (two at once do not fit in 16 GB).

## Pre-flight
The widened checkpoint, before training, counted in the bin env: carries and handovers within noise of B003's 3M count (39, 48). This checks that the widening changes nothing.

## Amendments
- **A1, 2026-10-24 23:50, after the pre-flight, before training.** The pre-flight count gave 55 handovers / 48 carries for the widened checkpoint. An exact check shows the widening changes nothing (52 tensors identical, new columns zero, identical outputs on 1,000 random states). The gap is counting noise: the *unwidened* B003 checkpoint, recounted now, gives 60 / 52 (seed 7; B003's own count was 48 / 39) and 68 / 51 (seed 8). Counts of 128 episodes vary by about 10. So every count in B005 uses seeds 7 and 8 (256 episodes), and the decision reads bin counts per 256.

## Predictions
If missing information is the stall:
1. bin run at 1.5M: bar in the bin in at least 20 of 256 (seeds 7 + 8).
2. control at 1.5M: bar in the bin in fewer than 10 of 256.

## Decision rule
- Prediction 1 and 2 hold: information is the stall. Next: a real-rig fix (memory, top camera or layout), each tested against this bin run.
- Both runs get the bar in the bin: more training was enough; information is not the main stall.
- Neither does: look at the carry itself (where the bar goes, lowering, opening), with `who_sees_what.py` and the eval videos.
- Either run loses the release (carries below 10): note it; a warm start can forget, and the comparison is then read from the earlier count (750k).

## Results
_Not run._
