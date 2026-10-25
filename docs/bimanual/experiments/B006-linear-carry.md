# B006: Does a carry reward that pulls at every distance get the bar to the bin?

**Status:** done 2026-10-25 01:20. The steeper carry reward barely moved the bar closer (closest 21 cm vs 23 cm); 0 in the bin. The reward is not the main stall.

## Question
In B005's control, the receiver carries the bar but moves it only from 28.5 to 23 cm from the bin. If the carry reward pulls as hard far away as up close, does the bar reach the bin?

## Intuition
The carry is scored by its distance d to the goal. The current score, `1 − tanh(5 d)`, rises steeply only in the last ~15 cm and is almost flat where the carry starts (about 30 cm away). Like a game of hot-and-cold where the guide stays silent until you are a hand's width away, the receiver hears no "warmer" and wanders.

## Change (one, from B005's control)
Env `SO101HandoverBarLinearCarry-v1`: the carry's distance term is `1 − d / 40 cm` instead of `1 − tanh(5 d)`. Everything else is v1: stages, bands (stage 5 stays 8-10), success, penalties. The same term also sits inside stage 6, as in v1.

| Gain in reward (of 16) for 10 cm closer, measured on constructed states (`check_reward.py`) | v1 (tanh) | B006 (linear) |
|---|---|---|
| 40 → 30 cm | 0.06 [computed] | 0.25 |
| 30 → 20 cm | 0.14 | 0.25 |
| 20 → 10 cm | 0.30 | 0.25 |

The handover point is 31-37 cm from the goal (300 sampled layouts), inside the 40 cm range.

## Method
- Start from B003's `ckpt_3M.pt` (already lets go), 1.5M steps, seed 1, B003's settings, evaluation every 250k. Same start, steps and seed as B005's control, which is the comparison (not rerun).
- Stage counts at 750k and 1.5M, seeds 7 and 8 (256 episodes), with the carry's bar-to-bin distance.
- Pre-flight: `check_reward.py` 18/18 on both envs (bands, order, success, penalties, the carry reward rising toward the goal).
- Caveat: changing the reward makes the warm-started critic's values slightly wrong at first; B005's control also restarted (fresh replay buffer, random first 5,000 steps), so both pay a restart cost, not identical ones.

## Predictions
If the flat carry reward is the stall:
1. At 1.5M, median closest bar-to-bin distance during the carry below 15 cm (control: 23 cm).
2. At 1.5M, bar in the bin in at least 10 of 256 (control: 0).

## Decision rule
- Prediction 2 holds: the reward was a stall. Next: train on and measure success; vision fixes only if success stays low.
- Prediction 1 holds, 2 fails: the bar gets closer but not in. Look at lowering and opening (the height term, the release over the bin).
- Neither (closest still above 20 cm): the reward is not the main stall; vision is. Next: B007, first-frame memory.
- Carries below 20 of 256 at 1.5M: the change disrupted the policy; read the 750k count and note it.

## Results
Commit `9d3e767`, `runs/handover_b006_2026-10-25`, exit 0. Evaluation returns 0-1.5M: 46.9, 31.0, 24.7, 37.4, 39.1, 38.8, 32.7; success 0.

| At 1.5M (of 256, seeds 7 + 8) | B005 control (tanh) | B006 (linear) |
|---|---|---|
| mid-air handover | 120 | 107 |
| receiver carries | 87 | 97 |
| carry: bar-to-bin at start → closest (median) | 28.5 → 23 cm | 29.8 → 21.4 cm |
| closest within 10 cm | 0 | 1 |
| bar in bin, success | 0, 0 | 0, 0 |

At 750k: B006 82 carries (control 1), closest 21.7 cm.

**Predictions:** 1 failed (21.4 cm, predicted < 15). 2 failed (0 in the bin).

**Reading:** pulling 1.8-4× harder at the carry's distance moved the closest approach by about 1.5 cm. The flat reward is not what stops the carry. Side effect: B006 kept the release through the warm-start restart (82 carries at 750k vs the control's 1); a carry reward that pays throughout keeps the carry worth doing.

**Decision (rule: closest still above 20 cm, the reward is not the main stall; vision is):** next is B007, first-frame memory.

