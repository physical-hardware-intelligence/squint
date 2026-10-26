# I025: does carrying the mouth-up term into stages 3 and 5 turn the socket up?

**Status:** run 2026-10-26. Predictions 1 and 3 missed at their deadlines; the effect came later and is large: I025 retains 165 of 256 at 8.5M, I024c 0 of 256 through 7M. Results below.

## Question
I024 (from scratch, approach fix 26e9702) inserts well (3M: in hole 197 of 256, seated 15, max depth p50 22.7 mm, p90 41.5 mm) but with the socket mouth down (median pitch while both up −37 to −46° from 1M to 2.75M). The probe's retention physics says a peg released 10° mouth-down falls out at friction 0.3. Under 26e9702 the mouth-up term (0-1 for 0-20° up) is paid in stage 2 only: stage 3 starts at a flat 10, so once the tip is in, mouth-down pays the same as mouth-up. Does paying the term through stages 3 and 5 turn the mouth up without losing the insertion?

## Change (`exp/stage3-tilt`, 6159342)
Stage 3 = 9 + tilt + depth terms (was 10 + depth terms); stage 5 base 15 + tilt (was 16), so letting go mouth-level still pays no more than holding mouth-level. Every state on the probe's intended path is mouth-up 20°, so its values are unchanged. Probe 34 of 34 (`experiments/bimanual/out/I025_insert_probe.txt`); the two new checks fail on 26e9702.

## Conditions
| | Start | Reward | Seed |
|---|---|---|---|
| I024c | I024's latest checkpoint at job start | 26e9702 (unchanged) | 2 |
| I025 | the same checkpoint | 6159342 | 2 |

Both: `SO101InsertPegRetain-v1`, `--actor_update_starts=250000`, 512 envs, γ 0.97, support −25 … 35, eval every 250k on 16 envs, backgrounds 0-50 ±6. Diag (`insert_diag.py`, 256 episodes, seed 8) every 250k.

## Predictions
1. By 1M, I025's median socket pitch while both up is above 0°; I024c's stays below −20°.
2. By 1M, I025 keeps the insertion: in hole ≥ 150 of 256.
3. By 2M, I025 retains at the end ≥ 5 of 256; I024c ≤ 2.

## Decision rule
- 1 and 3 hold: the change goes into PR #18.
- 1 holds, 3 fails: the mouth turns but the release does not follow; check `depth_at_release_mm`, `pitch_at_release_deg` and `retreat_at_end_cm`.
- 1 fails: 1 point is too weak against whatever makes mouth-down easier; look at the arm poses before raising the weight.
- 2 fails: the warm start lost the insertion again; rerun I025 from scratch.

## Results
Diag (`insert_diag.py`, 256 episodes, seed 8; JSONs `out/ret_i025_*.json`, `out/ret_i024c_*.json` on Explorer). Same start (I024 `ckpt_4750336`), same seed (2); only the stage 3/5 tilt term differs. Single seed per arm.

| Step | I025 in hole / seated / released seated / fell / **retained** | I024c same |
|---|---|---|
| 250k (actor held) | 198 / 47 / 1 / 1 / **0** | 176 / 33 / 0 / 0 / **0** |
| 1M | 154 / 30 / 0 / 0 / **0** | 166 / 25 / 0 / 0 / **0** |
| 2M | 176 / 91 / 6 / 0 / **0** | 190 / 48 / 5 / 0 / **0** |
| 4M | 177 / 130 / 14 / 0 / **0** | 176 / 63 / 4 / 2 / **0** |
| 5M | 201 / 155 / 48 / 4 / **0** | 165 / 97 / 4 / 1 / **0** |
| 5.5M | 189 / 135 / 131 / 3 / **0** | 204 / 108 / 8 / 1 / **0** |
| 6M | 185 / 132 / 131 / 1 / **128** | 137 / 81 / 4 / 1 / **0** |
| 7M | 190 / 136 / 134 / 1 / **132** | 182 / 118 / 3 / 2 / **0** |
| 8.5M | 210 / 167 / 166 / 2 / **165** | |

Median peg arm distance from the peg at the end, in released episodes: I025 0.6-1.4 cm to 5M, 2.3 cm at 5.5M, 6.2 cm at 6M, 9.9 cm at 8.5M; I024c 0.5-0.7 cm throughout. Median mouth pitch at release: I025 +26 to +37° from 3.25M on; I024c −17 to +10°.

Predictions:
1. Missed. I025's median pitch while both up was −24.4° at 1M (I024c −30.8°). It is not the lever: I025 still carries the socket mouth-down (median −14 to −16° from 6M) and turns it up before letting go (+26 to +31° at release).
2. Held: 154 in hole at 1M.
3. Missed at 2M (0 retained). From 6M, 128-165 of 256.

By the decision rule (1 and 3 by their deadlines) the change does not qualify; the rule set deadlines I guessed, not the effect. What the A/B shows: with the tilt term in stages 3 and 5, the policy seats more (167 vs 118 at 7-8.5M), lets go mouth-up, and from 5.5M moves away; without it, it lets go of 3-8 per 256 near level, more of those fall out, and the arm stays at the peg through 7M. Eval (16 envs) success at the end: I025 0.56 at 9.25M, 0.50 at 9.5M; I024c 0 through 7.5M.

Videos (wide camera, `ops/showcase_wide.py`): `experiments/bimanual/out/video_i025_8500k/wide_best_zoom.mp4`, `wide_grid.mp4`: 36 of 64 episodes succeed at the last step in that render.

What this does not show: a second seed; and why the move away started between 5.5M and 6M with an unchanged 0.2 per cm pay (I026, the larger pay without the tilt term, did not move away by 3M).
