# I001-I020: building Task 2 (mid-air peg insertion), 2026-10-25

**Status:** done; superseded by [I021](I021-retain.md). These runs used nine reward versions (`SO101InsertPeg-v1` … `-v9`) on a working branch on Explorer (`s2r/v2`, commits `5c5d4b7` … `08a0857`, not in this repo). This PR keeps what they taught as two envs in `envs/insertion.py`; the version flags are gone. Raw diagnostics: `experiments/bimanual/out/I0*.json`.

## The task
Left arm picks the socket, right arm the peg (ALOHA's Bimanual Insertion, Zhao et al. 2023, arXiv 2304.13705). Parts in `envs/insertion_parts.py`: peg 22 × 22 × 90 mm; socket 40 × 40 × 80 mm, hole 32 × 32 mm, 50 mm deep, 5 mm clearance a side (GUIDE §6: a policy pixel is 0.9-1.8 cm). Rig, cameras, inputs and actions as Task 1. Domain randomization: sizes ±3%, friction 0.3-0.7, density 250-450 kg/m³.

**What counts as success.** ALOHA's simulated insertion counts success when the peg touches the socket's pins while both arms hold (no release test). Parv's definition is stricter: the peg must stay in after the peg arm lets go and moves away. ALOHA's closest test is Put On Shoe: "successful if the shoe clings to the foot after both arms releases" (paper, full text read). Both tests are now envs:

| Env | Success | Steps |
|---|---|---|
| `SO101InsertPeg-v1` | peg 25 mm in, within 20°, both parts held and lifted | 120 |
| `SO101InsertPegRetain-v1` | peg seated (within 5 mm of the hole floor), no part of the peg arm touching it and its TCP ≥ 4 cm from the peg's centre line, socket still held up; read every step | 150 |

## Runs
All: 512 envs, γ 0.97, critic support −25 … 35, eval every 250k on 16 envs, backgrounds 0-50 ±6, A100. "Diag" = 128 or 256 episodes, DR on.

| Run | Reward | Start | Last step | Result |
|---|---|---|---|---|
| I001 | v1 | scratch, seed 1 | 7.5M | grasps both by 2M; diag 3.5M: success 89/256 (35%); eval 7.5M: success at end 6/16 |
| I002, I003 | v1 | scratch, seeds 2, 3 | 1.5M, 1M | peg never grasped (I002 1M: socket 126/128, peg 0); cancelled |
| I004 | v2 pre-grasp | scratch | 750k | cancelled once I001 grasped without it |
| I006, I007 | v3 presentation | scratch | 1.25M, 250k | I006 1M: peg grasp 0/128 (hovers open, never closes); cancelled |
| I008 | v4 standoff + corridor | warm, I001 2.5M | 2M | diag 1M: in hole 79/128, success 10/128; eval 2M: 2/16 |
| I009-I013 | v4-v6 | scratch or warm | ≤ 250k | cancelled for the next version before any result |
| I014 | v7 tip/floor/seat | warm, I001 3.5M | 3.5M | eval dips to 0 at 250k, 6/16 success once at 750k, 1/16 at end at 3M |
| I015 | v7c | warm, I001 3.5M | 750k | 0/16; cancelled |
| I016, I017 | v8b, v8 release | warm, I001 4.5M | 1M, 750k | I016 500k: in hole 228 → 10 of 256; 0 retained; cancelled |
| I019 | v9 | warm, I014 2.0M | 1M | 500k: in hole 84/256, peg ≥ 35 mm 0; cancelled |
| I020 | v9 | warm, I001 6.0M | 1.25M | 500k: in hole 2/256; cancelled |

## What each change was for, and what happened
| Version | Observed (diag) | Change | Outcome |
|---|---|---|---|
| v2 | I001 750k: peg arm parks closed fingertips on the peg's top (tips 14.9 mm apart at jaw 0, peg 22 mm), reach already pays 0.95; peg grasp 7/128 | pay an open jaw straddling the part | moot: I001 left the parking spot by 2M (peg 128/128) |
| v3 | I001 2M: both up, peg ~38° off the axis, last 5 cm not closed | socket presented at a fixed meeting point; alignment rescaled 30° → 0 | I006 never grasped the peg (v2's term made hovering open pay as much as closing) |
| v4 | I001 2.25M: peg pressed on the rim 2 cm off the axis, tilted, still (0.2 cm/s); rim contact in 57/125 | standoff 3 cm out, inward progress only inside an 8 mm corridor, −0.5 for rim contact | I008 (I001 2.5M + 1M of v4): 10/128 success (seed 7); I001 itself at 3.5M (v1 throughout): 89/256 (seed 8). No sign v4 helped |
| v5 | v2's pre-grasp paid nothing for closing | pay the closing motion | cancelled before a result |
| v6 | in-hole = tip centre within 16 mm of the axis: a 22 mm peg 12 mm off, face on the rim, counted as in | in-hole needs lateral < 6 mm | cancelled; the 6 mm test also rejects a peg resting in a corner (6.9 mm), fixed below |
| v7 | I001 3.5M (128 episodes): 25 of 35 successes flicker at 22-25 mm (success flat from 25 mm); 4 lost at > 45 mm (bug below) | tip = leading end; hole floor; depth pays to 40 mm | I014: 1/16 at the end at 3M, below I001 |
| v8 | Parv's retention test; I001 4.5M under it: peg ≥ 35 mm 51/256, retained at the end 0 | depth to 45 mm, release and retreat stages, mouth-up term | I016 backed out of the hole (228 → 10) |
| v9 | v8 cut the held-25 mm reward from 16 to 12.2 | keep v1's 10-14 over 0-25 mm | I019, I020: in hole 84 and 2 at 500k |

## Errors (mine), found and fixed
1. **Tip flip.** The tip was the peg end nearer the mouth. Past 45 mm (half the peg) the outer end is nearer, so a peg 46 mm deep read as 44 mm outside, pointing away, and the reward fell to stage 1-2 (4 of 35 I001 successes at 3.5M). Fixed: the leading end along the socket axis.
2. **No hole floor.** A tip behind the closed end, on the axis, counted as inside. Not seen exploited. Fixed: axial < depth + 5 mm.
3. **In-hole test.** v1 counted the tip centre within 16 mm of the axis (face on the rim counts); v6 used a Euclidean 6 mm (rejects a resting corner at 4.9/4.9 mm; ignores the peg's roll, under which a 22 mm peg is up to 31.1 mm wide). Fixed in this PR: the leading face's extent per socket axis, roll and tilt included, must fit the hole within 1 mm (`geometry()["fit"]`).
4. **"v1 cannot get past the parked jaw"** (I-A1): wrong for seed 1, which grasped both by 2M. Seeds 2 and 3 never grasped the peg, so it holds for them.
5. **Videos shown as success** (v1, held insertion) score 0 under the retention test: released at 25 mm the peg falls out in every case measured.

## What carries into this PR
1. **v1's reward learns the held insertion with no grasp shaping**, on 1 of 3 seeds (seeds 2 and 3 had not grasped the peg at 1M; cancelled at 1.5M and 1M). I001 at 7M on this PR's `SO101InsertPeg-v1` (stricter in-hole test), 256 episodes: success 103 at some step, 70 at the last step (27%); max depth p10/50/90 9.8 / 25.2 / 44.8 mm. On `SO101InsertPegRetain-v1`: 0 retained, 11 seated.
2. **Retention needs depth and a mouth not pointing down** (`insert_probe.py`, socket held rigidly, 2 s): released at 25 mm the peg falls out at every pitch (−10° to +20°) and friction (0.3, 0.7). Mouth up 10-20°, from 42 mm it slides to the floor (50.5 mm) at both frictions; level, from 46 mm (0.3) or 42 mm (0.7); 10° mouth-down it falls out from any depth tried at friction 0.3 and stays from 46 mm at 0.7.
3. **Policies already tilt the mouth up**: I001 4.25M socket pitch over the last 30 steps 4.0 / 11.7 / 22.3° (p10/50/90); I001 7M while both are up 9.1° median.
4. **Warm starts under a changed reward dip, and the release rewards (v8, v9) never recovered insertion within 1-1.25M**: I016 in hole 228 → 10, I020 → 2, I019 84 of 256 at 500k. v4 and v7 warm starts dipped and recovered (I008 1M 10/128; I014 750k 6/16 once). I001's held-25 mm state pays 16 of 16 under v1 and 14 of 18 under Retain; I021 tests whether the dip is the critic.
5. **Dropped, not measured as harmful**: the fixed meeting point (v3), corridor and rim penalty (v4), pre-grasp terms (v2, v5). Each made the reward depend on a designer-chosen pose; v1 learned without them, and none was compared with v1 at equal steps on equal seeds.
