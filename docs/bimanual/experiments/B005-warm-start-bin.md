# B005: From a policy that already lets go, does knowing the bin's position get the bar in?

**Status:** done 2026-10-25 00:35. The bin input broke the handover (7 of 256, 0 carries); the control relearned the release (87 of 256 carry) but no bar reached the bin. The carry barely moves toward the bin, where its reward is almost flat.

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
Both runs exit 0 (commit `20f4637`). Evaluation returns 0-1.5M, control: 46.2, 26.4, 33.6, 37.6, 38.6, 35.8, 42.2; bin: 46.2, 30.9, 38.2, 32.5, 39.6, 36.0, 39.6. Success 0 throughout.

| Stage (of 256, seeds 7 + 8) | control 750k | control 1.5M | bin 750k | bin 1.5M |
|---|---|---|---|---|
| mid-air handover | 103 | 120 | 1 | 7 |
| receiver carries | 1 | 87 | 0 | 0 |
| bar in bin | 0 | 0 | 0 | 0 |

**Predictions:** 1 failed (0 in the bin). 2 held (0 in the bin), but trivially.

**Reading:**
- Restarting training alone made the control forget the release for a while (1 carry at 750k), then relearn it (87 at 1.5M).
- Adding the bin's position broke the handover and it did not recover. With B004, all three runs given the bin's position failed to release; both runs without it released. The mechanism is unknown. Giving the policy privileged state is therefore not a clean probe of "missing information" here.

**What the carry does** (control 1.5M; `stage_counts.py` now reports the horizontal bar-to-bin distance during the carry; 89 carrying episodes): median 28.5 cm at the start of the carry, closest 23 cm; none within 10 cm. The receiver holds the bar and barely moves it toward the bin.

**Why it might not move:** stage 5 pays `1 − tanh(5 d)` for the distance d to the bin (`handover.py`, `place`). That term is 0.09 at 30 cm, and 10 cm closer adds 0.09 of 16 points (computed): almost flat where the carry starts, so moving toward the bin barely changes the reward.

**Decision (rule: neither gets the bar in, look at the carry):** two explanations remain, vision (the bin is mostly out of the receiver's camera) and a flat reward at the carry's distance. The reward one is a one-line change and cheaper to test, so it goes first (B006, proposed).

