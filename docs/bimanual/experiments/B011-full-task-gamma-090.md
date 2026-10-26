# B011: Is γ 0.97 what stalls the handover in the whole task?

**Status:** done 2026-10-25 23:25. Seed 1 learned the handover at γ 0.9 (193 of 256), seed 2 did not (13): γ 0.9 is not a reliable fix. Seed 1's carries stall (0 in the bin).

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
Seed 1 run 2026-10-25 19:46-20:38, commit `06193d4`, preflight 8 of 8 (`experiments/bimanual/out/b011_preflight.log`): `python train_squint.py --exp_name=b011_wrist_s1 --env_id=SO101HandoverBarV2-v1 --seed=1 --num_envs=512 --total_timesteps=3000000 --eval_freq=250000 --env_background_level 0 50 --env_background_tint 6 --gamma=0.9 --v_min=-20 --v_max=20`. Raw: `experiments/bimanual/out/b009_b010_counts.txt` (B011's lines follow the 19:41 reorder note), `stages_b011_*.json`.

| 256 episodes (counting seeds 7 + 8) | B011 s1 (γ 0.9) | B010 wrist s1 (γ 0.97) |
|---|---|---|
| handovers at 1.5M | 109 | 3 |
| handovers at 3M | **193** | 29 |
| carries at 3M | 101 | 2 |
| in the bin, success | 0, 0 | 0, 0 |
| carried bar to the bin, median start → closest | 32.1 → 30.8 cm (seed 7), 32.8 → 31.6 cm (seed 8) | |

**Seed 2** (2026-10-25 22:30-23:25, same command with `--seed=2`): at 1.5M 1 handover, at 3M **13** (7 and 6), 0 carries. The picker brings the bar to the handover point (233 of 256) and the receiver comes within 3 cm (141), but it does not grip: B010's pattern, at γ 0.9.

**Predictions.** 1 holds on seed 1 (193), fails on seed 2 (13). 2 holds on seed 1 (193 vs 29), fails on seed 2 (13 vs B010's 53). 3 fails (0 in the bin).

**Reading.** On seed 1, only γ changed and the handover came back (29 → 193) with the release (101 carries). Seed 2 did not: at γ 0.9 one seed of two learned the handover, at γ 0.97 none of four runs did (29, 53, top camera 14, B012 8). γ 0.9 helps, but from scratch the grip is found on some seeds only, as expected of a step paid +0.16 over hovering ([reward-design](../reward-design.md) §7). The rest of this paragraph is seed 1's. The carries then move the bar about 1 cm toward the bin, as in B003-B006. Carried alone (B008, bin position given), the bar reached the bin at both γ (87%, 69%) and γ 0.97 held the finish better (success at the end 46% vs 27%). The handover wants 0.9, the carry 0.97. Rule "1 and 2 hold" → [B012](B012-v3-carry-practice.md): a clearly paid handover step (reward v3) at γ 0.97, plus carry practice.
