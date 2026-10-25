# B003: Does paying for the release get past the stall?

**Status:** done 2026-10-24 21:45. The release is learned (39 of 128 carry); nothing reaches the bin. During the carry the bin is almost never in the receiver's own camera (2%), but is in the picker's camera 40% of the time (A2).

## Question
B002 stalled with both grippers on the bar (17 of 128 episodes) and no releases. If a clean release pays a full point more, does the policy let go and carry on to the bin?

## Change (one, from B002)
Reward stage 5 (the receiver alone carries the bar after a mid-air handover) starts at 8 instead of 7; stage 6 at 10 instead of 9; success 16 instead of 15 (normalized by 16). Checked on constructed states: 17 of 17 (`experiments/bimanual/check_reward.py`).

## Method
B002's settings and seed, 3M steps (about 50 min), evaluation every 250k; stage counts at 1.5M and 3M with B002's counter (128 episodes, seed 7).

## Predictions
1. At 1.5M, as many or more mid-air handovers than B002 (≥ 17 of 128).
2. At 1.5M, releases followed by the receiver carrying in at least 10 of 128.
3. At 3M, the bar in the bin in at least 5 of 128; success at least 3%.

## Decision rule
- Releases still 0 at 1.5M: the payoff was not the cause; next, a curriculum starting episodes from both-hold states.
- Releases but nothing reaches the bin by 3M: look at what follows the release (drops, carrying).
- Success at least 10% at 3M: continue training and evaluate at deploy settings.

## Amendments
- **A1, 2026-10-24 21:20, after the 1.5M counts and before any 3M data.** The rule leaves a gap: releases above 0 but far below prediction 2. Added: at 3M, fewer than 10 carries in 128 counts as "releases not learned", and the curriculum follows. The counter now counts the carry from stage 5's own condition (`handed_over & receiver_grasp & ~picker_touch_bar`, commit `ce3f407`) rather than from reward bands, which penalties can shift.

- **A2, 2026-10-24 22:05, correction after the run.** The table below counted only the receiver's camera, but the policy gets both images. Exact count (`experiments/bimanual/who_sees_what.py`, segmentation pixels with occlusion, at least 64 pixels = one policy pixel; 3M checkpoint, 128 episodes, seed 7):

  | Phase (steps) | Picker cam: bar, bin | Receiver cam: bar, bin | Bin in either |
  |---|---|---|---|
  | start (128) | 96%, 0% | 0%, 99% | 99% |
  | picker holds (7,964) | 100%, 14% | 58%, 1% | 15% |
  | both hold (96) | 100%, 17% | 98%, 0% | 17% |
  | receiver carries (4,582) | 16%, 40% | 100%, 2% | 42% |

  At the start each camera sees one object. During the carry the bin is mostly visible through the *other* arm's camera, so the policy must map the picker camera's view to the receiver's motion. B004 still tests whether the bin's position, given directly, unblocks the carry. Carry steps here (4,582) differ from the stage count (3,746) on the same checkpoint and seed: GPU simulation is not reproducible, so read percentages, not counts.

## Results
**Main run** (commit `455fafa`, `runs/handover_b003_2026-10-24` on the box): 512 envs at about 1,050-1,140 steps/s.

| Evaluation at | 0 | 250k | 500k | 750k | 1M | 1.25M | 1.5M |
|---|---|---|---|---|---|---|---|
| Return | −50.2 | 18.4 | 36.4 | 33.2 | 35.3 | 36.8 | 37.0 |
| Success | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Returns are not comparable with B002's: the reward scale changed.

**Stage counts at 1.5M** (128 episodes, seed 7; counter `ce3f407`):

| Stage | B002 | B003 |
|---|---|---|
| picker grasps | 127 | 128 |
| receiver within 3 cm of the bar | 124 | 119 |
| both hold the lifted bar (mid-air handover) | 17 | 33 |
| picker lets go, receiver carries | 0 | 1 |
| bar in bin, success | 0, 0 | 0, 0 |

Two counts of the same checkpoint and seed gave 127/31 and 128/33 (grasp/handover): the GPU simulator is not bit-reproducible, so differences of a few episodes are noise.

**Predictions at 1.5M:** 1 held (33 ≥ 17 handovers, about twice B002). 2 failed (1 carry, predicted ≥ 10).

**Decision at 1.5M (rule: curriculum only if releases are still 0):** 1 release, so the run continued to 3M.

**3M** (training exit 0; evaluation returns 2.25M-3M: 37.1, 49.2, 40.0, 41.7; success 0 throughout). Stage counts, 128 episodes, seed 7, counter `54ba49f`:

| Stage | 1.5M | 3M |
|---|---|---|
| picker grasps | 128 | 128 |
| both hold (mid-air handover) | 33 | 48 |
| picker lets go, receiver carries | 1 | 39 |
| bar in bin, success | 0, 0 | 0, 0 |
| arms touching / robot touching table | 8 / 10 | 24 / 22 |

Both-hold lasts 115 steps in total over 48 episodes (about 2 steps each): the picker now lets go almost at once. Carrying lasts 3,746 steps over 39 episodes (about 96 each): the receiver holds the bar until the episode ends.

**Bin in the receiver's wrist view, first measure** (bin centre or a rim corner projected inside the 128 × 128 image; geometry only, occlusion ignored, any size, so an upper bound; agrees with the segmentation image on 98.2% of 3,840 camera-steps). A2's exact count (occlusion included, at least one policy pixel) gives 2% for the carry:

| Phase | Bin in view |
|---|---|
| first step | 125 of 128 episodes |
| both hold | 0 of 115 steps |
| receiver carries | 437 of 3,746 steps (12%) |

The policy has no memory and no bin position in its input. Yash noted the receiver losing the bin in the 2.5M video before the count. (Corrected by A2: the policy gets both images, and the picker's camera often sees the bin.)

**Predictions:** 1 held. 2 failed at 1.5M (1 carry) but was met by 3M (39). 3 failed (0 in bin, 0% success).

**Decision (rule: releases but nothing in the bin by 3M, look at what follows the release):** what follows is a carry with the bin out of the receiver's own camera (A2: 2% of carry steps; the picker's camera sees it 40%). This is a correlation; the causal test is B004: give the policy the bin's position (simulation only, a diagnostic, not deployable) and change nothing else. If carries then reach the bin, missing information is the stall, and the deployable fix is memory, a top camera, or a layout that keeps the bin in view. The mid-air start curriculum is not triggered (39 ≥ 10).
