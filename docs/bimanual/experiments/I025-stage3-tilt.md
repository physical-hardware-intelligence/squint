# I025: does carrying the mouth-up term into stages 3 and 5 turn the socket up?

**Status:** protocol written 2026-10-26, before any result.

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
_Not run._
