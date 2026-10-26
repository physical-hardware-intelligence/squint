# I021: Does a warm start from the held insertion learn to leave the peg in?

**Status:** I021 results to 2.75M below (predictions 1-3 missed); I022 and I023 queued. Protocol written 2026-10-25 ~22:45 PDT, before any result; both runs queued at 22:35. `insert_probe.py` 31 of 31 on the tree that runs (`experiments/bimanual/out/I021_insert_probe.txt`).

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

`near` = 1 − tanh(4 × √(axial² + (2 × lateral)²)), axial the tip's distance from the mouth plane along the axis (either side; see the I023 bug below), lateral off it. No meeting point: the arms choose where. Why each threshold: the retention physics in `insert_probe.py` (seated pegs stay at both frictions with the mouth ≥ 10° up; released at 25 mm they fall out). A peg that falls out after the release scores stage 1 (≤ 4); a seated peg still held scores 16, 2 below success. Probe: the intended path never decreases (10 at the mouth, 14 at 25 mm, 16 at 48 mm, 16.5 open, 17.5 at 5 cm, 18), and opening too shallow, mouth level, mouth down, or with the socket dropped adds nothing; letting go mouth-level or mouth-down pays no more than holding.

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

### I021 (warm, I001's critic), diag every 250k, 256 episodes, seed 8 (`experiments/bimanual/out/I021_ret_*.json`)
| Step | In hole | Seated | Retained at end | Max depth p10/50/90 mm |
|---|---|---|---|---|
| 250k (actor still held) | 193 | 14 | 0 | 10.9 / 25.3 / 41.6 |
| 500k | 128 | 5 | 0 | 5.8 / 21.2 / 34.9 |
| 750k | 144 | 1 | 0 | 3.3 / 16.8 / 29.9 |
| 1M | 150 | 4 | 0 | 8.3 / 22.7 / 35.0 |
| 1.5M | 183 | 1 | 0 | 6.0 / 17.0 / 29.5 |
| 2M | 183 | 0 | 0 | 6.0 / 18.4 / 28.1 |
| 2.5M | 182 | 1 | 0 | 4.1 / 17.4 / 29.5 |
| 2.75M | 180 | 5 | 0 | 9.1 / 22.2 / 34.4 |

Eval (16 envs) success 0 at every step to 2.75M. Prediction 1 missed (128 < 150 at 500k), 2 missed (seated ≤ 5 after 250k), 3 missed (0 retained). α stayed small (exp log α 0.0029 at the start, 0.0021 at 2.5M), so the drop is not added exploration noise.

### Why the peg stops (`insert_stall.py`, `experiments/bimanual/out/I021_stall.txt`)
At each episode's deepest step, both checkpoints show the same picture: the peg rides one side wall (lateral 4-5 mm of the 5 mm clearance, fit ≈ 0), contact with the socket is light (median 0.5-1.6 N) and sideways (share along the socket axis ≤ 0.03), and over the 10 steps before it the peg arm's TCP and the peg move in by the same amount (2.5M: 1.1-1.7 mm and 1.7-1.9 mm, median) with no slip along the peg (median ≤ 0.11 mm). Only pegs at 45-60 mm push on a face (share 0.91 at 250k: the floor). In-plane geometry agrees: a 22 mm peg in the 32 mm hole jams only where 22 cos θ + d sin θ > 32 mm, which allows any tilt below 20 mm depth and 32.6° at 25 mm; the stalled pegs are tilted 10-18° (median) at 10-30 mm.

So the peg is not wedged and not slipping: the policy stops advancing it. Under the decision rule this is the "2 fails" branch, but not the rim jam it anticipated. The depth fell after the actor began learning (250k → 750k: median 25.3 → 16.8 mm).

Inference, not tested: the retain reward's per-mm slope below 25 mm matches v1's (normalized 0.009 vs 0.010 per mm), but v1 also paid its full success (16 of 16) from 25 mm, a step the retain ladder does not have (14 of 18 at 25 mm). I022 (fresh critic) and I023 (from scratch) separate a warm-critic effect from the reward before any reward change.

### I023 (from scratch): a reward bug, fixed in I024
I023 grasps and lifts both parts by 500k (both up 240-253 of 256 from 500k to 1.75M) and never enters the hole (0 of 256 through 1.75M). At 2.5M the peg tip sits on the socket's axis line 34-43 cm past the mouth, behind the closed end, with the mouth pointing away from the peg arm (4 episodes dumped at steps 100 and 149: lateral 4-47 mm, near 0.64-0.97, reward 8.9-9.9 of stage 2's 10). The cause is mine: `approach` used only the part of the axial distance outside the mouth, `clamp(-axial, 0)`, so any tip past the mouth plane on the axis read as at the mouth. Both envs had it; I001, I021 and I022 found the real mouth anyway, and their depth stall is inside the hole (stage 3), which this term does not score. Fix: `approach` uses |axial|. New probe check: a tip on the axis 40 cm behind the closed end gets the same `near` as one 40 cm in front (0.078); the old code gave 1.000 (`experiments/bimanual/out/I024_insert_probe.txt`, 32 of 32 on the fix). I023 is stopped; I024 reruns it on the fixed env (seed 1, same arguments).

