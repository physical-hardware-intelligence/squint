# B011: Is γ 0.97 what stalls the handover in the whole task?

**Status:** protocol written 2026-10-25, before the runs.

## Question
B010's wrist run (reward v2, γ 0.97) made 3 handovers in 256 episodes at 1.5M and 29 at 3M; B003 (reward v1, γ 0.9) made 66 [from 33 of 128] and 128. Before the handover the two rewards are identical (8,000 states checked, largest difference 0), so γ is the one difference there. With γ 0.9, does the handover come back?

## Intuition
The handover step is barely rewarded: closing the grip pays about +0.1 over hovering beside the bar, an arm touch costs 2, a nudge past 5 cm costs 2.5 ([reward-design](../reward-design.md) §7). Whether a run learns such a step can tip on the horizon. If γ 0.9 brings the handover back, the carry after it has reward v2 and may reach the bin.

## Change (one, from B010's wrist condition)
γ 0.9, critic support −20 … 20 (B003's). Env `SO101HandoverBarV2-v1`, everything else as B010: 512 envs, 3M steps, evaluation every 250k (scene rebuilt per episode), backgrounds 0-50 ±6, colour jitter on.

## Method
Seeds 1 and 2. Stage counts at 1.5M and 3M (seeds 7 + 8, 256 episodes): handovers, carries, in the bin and success (at any step and at the last step), closest bar-to-bin. Preflight first. Queue order (after B010's top seed 1): B011 s1, B010 wrist s2, B011 s2, B010 top s2.

## Predictions
If γ 0.97 stalls the handover:
1. Handovers at 3M at least 70 of 256 on each seed (B003: 128 and 88; B010 wrist seed 1: 29).
2. Handovers above B010 wrist's on the same seed.
3. The bar in the bin at least 10 of 256 on at least one seed (reward v2 after the handover).

## Decision rule
- 1 and 2 hold: γ is the handover's problem. Next: keep γ 0.97's better finish (B008) without losing the handover, by making the handover step clearly rewarded (v3) and testing it at γ 0.97.
- 1 fails: γ is not the cause (or not alone); next, the v3 handover reward at both γ values.
- 3 holds: the first bars in the bin from the whole task; measure success at the end and look at the videos.

## Results
_Not run._
