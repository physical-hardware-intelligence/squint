# B007: Does remembering the first frame get the bar to the bin?

**Status:** run 1 stopped at 2.0M (invalid: the kept frame was stale, A1); fixed and restarted 2026-10-25 02:15.

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
_Not run._
