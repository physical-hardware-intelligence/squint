# B003: Does paying for the release get past the stall?

**Status:** protocol written 2026-10-24, before the run.

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

## Results
_Not run._
