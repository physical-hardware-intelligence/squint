# B003: Does paying for the release get past the stall?

**Status:** running (launched 2026-10-24, commit `455fafa`); 1.5M counts in, 3M due about 21:45.

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

**Decision at 1.5M (rule: curriculum only if releases are still 0):** 1 release, so the run continues to 3M as planned. 3M results go below.
