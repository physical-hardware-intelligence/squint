# I021: Does a warm start from the held insertion learn to leave the peg in?

**Status:** running. Protocol written 2026-10-25 ~22:45 PDT, before any result; both runs queued at 22:35. `insert_probe.py` 31 of 31 on the tree that runs (`experiments/bimanual/out/I021_insert_probe.txt`).

## Question
I001 inserts the peg while holding it (27% success at the end on `SO101InsertPeg-v1`, 7M) but never leaves it in: retained 0 of 256 on `SO101InsertPegRetain-v1`. Both earlier release rewards (v8, v9), warm-started from I001 or I014, lost the insertion itself within 500k ([I001-I020](I001-I020-insertion-dev.md)). Does the retain reward, with the critic relearned before the actor moves, keep the insertion and add the release?

## The env, `SO101InsertPegRetain-v1` (`envs/insertion.py`, `dense_reward`)
| Stage | Condition | Points |
|---|---|---|
| 1 | per arm: reach its grip point 1 − tanh(5d); held: 2 + lift (0-1 over 8 cm); arms add | 0-6 |
| 2 | both held and up: 1.75 × near + 1.25 × align × near + 1 × mouth tilt (0-20° up) | 6-10 |
| 3 | both held, leading face in the hole: 10 + 4 × depth/25 mm × align + 2 × (depth − 25)/(48 − 25) mm × align | 10-16 |
| 4 | seated (≥ 45 mm) and socket held up: + 0.5 × jaw opened × mouth-up (full at 10°) | ≤ 16.5 |
| 5 | peg arm off the peg (no contact from jaws, wrist or forearm), peg seated, socket up: 16 + mouth-up × (0.5 + moving away 0-5 cm from the peg's centre line) | 16.5-17.5 |
| success | seated, released, TCP ≥ 4 cm from the peg's centre line, socket up, every step | 18 |
| penalties | robot on the table −2, arms touching −1 | |

`near` = 1 − tanh(4 × √(gap² + (2 × lateral)²)), gap outside the mouth along the axis, lateral off it. No meeting point: the arms choose where. Why each threshold: the retention physics in `insert_probe.py` (seated pegs stay at both frictions with the mouth ≥ 10° up; released at 25 mm they fall out). A peg that falls out after the release scores stage 1 (≤ 4); a seated peg still held scores 16, 2 below success. Probe: the intended path never decreases (10 at the mouth, 14 at 25 mm, 16 at 48 mm, 16.5 open, 17.5 at 5 cm, 18), and opening too shallow, mouth level, mouth down, or with the socket dropped adds nothing; letting go mouth-level or mouth-down pays no more than holding.

## Change for the warm start
`train_squint.py --actor_update_starts=N` (new, default 0 = unchanged): the actor and α are not trained before step N; the critic and the encoder are. α is held because the frozen actor's entropy is fixed, so α would drift one way. The encoder is shared with the actor, so the actor's features still move during those steps. Checked on Explorer, 20k steps from I001 7M, 64 envs: with the actor held for the whole run, the saved actor is bit-identical to the source and log α unchanged (−5.847), while critic and encoder changed; with the default 0, actor and log α changed (−5.847 → −4.912). Both runs, like earlier smoke runs on the base trainer, exit with a segfault after "Cleanup complete" (exit 139); the checkpoints are written before it.

## Conditions
| | Start | Critic | Seed |
|---|---|---|---|
| I021 | I001 `ckpt_7000064` | I001's | 10 |
| I022 | I001 `ckpt_7000064` | fresh (`fresh_critic.py`) | 11 |

Both: `SO101InsertPegRetain-v1`, `--actor_update_starts=250000`, 512 envs, γ 0.97, support −25 … 35, eval every 250k on 16 envs, backgrounds 0-50 ±6, up to 7 h 55 min. Diag (`insert_diag.py`, 256 episodes, seed 8) at 500k, 1M, 2M.

## Baseline (I001 7M on this env, 256 episodes)
both up 240, in hole 197, seated 11, released while seated 1, fell out 1, retained 0; max depth 9.8 / 25.9 / 40.8 mm (p10/50/90); socket pitch while both up −4.5 / 9.1 / 19.0°.

## Predictions
1. At 500k, both runs keep the insertion: in hole ≥ 150 of 256 (I019, I020 under v9: 84, 2).
2. By 1M, seated ≥ 40 of 256 on at least one run (baseline 11).
3. By 2M, retained at the end ≥ 10 of 256 on at least one run.

## Decision rule
- 1 fails on both: the frozen actor did not protect the insertion. Next: also freeze the encoder for the first 250k, or train the retain env from scratch.
- 1 holds, 2 fails: depth stalls at ~25 mm as in v1. Look at depth over time in videos; if the peg jams on the rim, the per-axis fit test should show it.
- 2 holds, 3 fails: the release is the stall. Check `depth_at_release_mm` and `pitch_at_release_deg` against the probe's retention table before changing anything.
- 3 holds: the retained-peg checkpoint goes to the real-arm test.

## Results
_Not run._
