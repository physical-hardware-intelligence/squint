# B007: Does remembering the first frame get the bar to the bin?

**Status:** done 2026-10-25 04:46. Memory never got tested: neither B007 seed learned the handover (6 and 0 of 256), while B003 seed 2 did (88). Every run with an extra input has now failed at the handover.

## Question
B003-B006 ruled out the release payoff, a flat carry reward and plain extra training: the receiver carries the bar but stops about 21-23 cm from the bin. The receiver's camera sees the bin at step 0 in 99% of episodes and in 2% of carry steps (B003 A2). If the policy keeps the first frame, does the bar reach the bin?

## Intuition
The policy has no memory: once the bin leaves the picture, nothing tells it where the bin is. Keeping the opening snapshot, when the bin is in view, gives it that fact for the whole episode. The receiver starts from the same pose each time, so "bin at this pixel in the snapshot" maps almost directly to "bin at this spot on the table".

## Change (one, from B003)
Env `SO101HandoverBarFirstFrame-v1`: both wrist images from the episode's first step are given at every step beside the current ones (left now, right now, left first, right first: 12 channels instead of 6). Both, not only the receiver's: roles change every episode, and the first frames also tell the policy which arm is which. Reward, layout, settings as B003 (v1 reward, `1 − tanh(5 d)` carry).

Checked on the box (2026-10-25): 12 channels; the kept frame equals the current one at step 0, stays fixed through the episode, is carried into the episode's final observation, and is replaced at the next episode. Smoke test (64 envs, 30k steps) exit 0.

## Method
From scratch, 3M steps, evaluation every 250k, run on the box overnight in this order:
1. B007 seed 1
2. B003 seed 2 (v1, no memory): the missing baseline. Does B003's release reproduce?
3. B007 seed 2

Stage counts at 1.5M and 3M for each (seeds 7 + 8, 256 episodes), with the carry's bar-to-bin distance. Queue: `runs/overnight_2026-10-25.sh` on the box, counts in `runs/overnight_counts.txt`.

## Amendments
- **A1, 2026-10-25 02:10, during run 1.** Run 1's 1.5M count disagreed with its training returns (picker grasps 43 of 128 and return 3 in the counter, against return 34 in training). Cause: `BimanualWristCameraEnv.reset` rendered its observation before moving the wrist cameras to the new arm pose and returned that frame. A memoryless policy loses one frame per episode; B007 kept it all episode. The kept "first frame" showed the previous episode's last gripper view in training, and a different view in the counter. My first check compared the kept frame with the step-0 frame, and both were stale, so it passed. Fix: reset observes again after the camera sync (`bimanual_base.py`), and the first-frame env captures that synced frame. New check `experiments/bimanual/check_reset_frames.py` compares reset frames with a fresh render: 6 of 9 on the old code, 9 of 9 after; the plain env 3 of 3. Run 1 (2.0M steps) is kept but not used. All three runs restart on the fixed code; B003 seed 2 now also gets a correct step-0 frame (one frame per episode; B003 seed 1 had the stale one). The single-arm `WristCameraEnv.reset` has the same stale-frame pattern (not changed here).

## Predictions
If the missing view of the bin is the stall:
1. At 3M, in at least one B007 seed: median closest bar-to-bin during the carry below 15 cm (B005 control: 23 cm; B006: 21 cm).
2. At 3M, in at least one B007 seed: bar in the bin in at least 10 of 256.
3. B003 seed 2 at 3M: 0 in the bin.

## Decision rule
- 2 holds and 3 holds: memory unblocks the carry. Next: keep training toward success; plan the first frame on the real rig.
- B007 carries but closest stays above 20 cm: memory is not enough; next, a top camera or look at the carry in the videos.
- Neither B007 seed releases while B003 seed 2 does: the extra input hinders the release, as the bin input did in B004/B005; note it.
- B003 seed 2 does not release: B003's release was seed luck; the release itself needs fixing first (mid-task starts, then SGS).

## Results
All on commit `2f94d84` (the reset fix), exit 0. Counts: seeds 7 + 8, 256 episodes.

| At 3M | B003 seed 2 (no memory) | B007 seed 1 | B007 seed 2 |
|---|---|---|---|
| picker grasps | 253 | 205 | 222 |
| receiver within 3 cm of the bar | 179 | 48 | 37 |
| mid-air handover | 88 | 6 | 0 |
| receiver carries | 66 | 0 | 0 |
| carry: bar-to-bin at start → closest (median) | 30 → 29 cm | — | — |
| bar in bin, success | 0, 0 | 0, 0 | 0, 0 |

At 1.5M: B003 seed 2 32 handovers, 1 carry; B007 seed 1 6, 0; seed 2 3, 0.

Evaluation returns 250k-3M, B003 seed 2: 7.8, 29.2, 36.0, 35.5, 38.6, 36.5, 38.5, 37.9, 37.8, 36.6, 35.2, 40.2. B007 seed 1: 8.4, 19.5, 22.9, 26.5, 31.8, 17.4, 28.9, 31.6, 23.9, 28.7, 25.2, 16.4. Seed 2: 6.3, 17.5, 20.9, 28.9, 21.0, 20.5, 22.1, 25.9, 26.5, 22.9, 24.0, 25.8. Success 0 throughout.

**Predictions:** 1 and 2 failed: no B007 carry at all, so no distance to measure. 3 held (B003 seed 2: 0 in the bin).

**Reading:**
- B003's release reproduces: seed 2 hands over (88) and carries (66) like seed 1. Its carry also stalls (30 → 29 cm).
- B007 fails earlier than the carry: the receiver rarely gets near the bar (37-48 of 256 within 3 cm, against 179). Memory never got the chance to help.
- Pattern across experiments: every run with an extra input failed at the handover (B004 ×2 and B005 with the bin position, B007 ×2 with the first frame); every run without one handed over and carried (B003 ×2, B005 control, B006). 5 against 4. Both added inputs are constant within an episode and change between episodes; why that would hurt the handover is not known.

**Decision (rule: neither B007 seed releases while B003 seed 2 does):** the extra input hinders learning before the carry, as the bin input did. The question "does seeing the bin unblock the carry?" is still open, and adding inputs to the whole task from scratch cannot answer it. Next: test the carry on its own (episodes that start with the receiver already holding the bar), where the handover cannot get in the way.

