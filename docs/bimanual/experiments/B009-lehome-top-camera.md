# B009: Does the top camera placed as on Ilia's rig work as well on the carry?

**Status:** done 2026-10-25 17:45. The LeHome placement works: in the bin 94% (ever) and 83% (at the end) over both seeds, against 88% and 79% for the 90 cm camera. B010 keeps it.

## Question
B008's top camera was 90 cm above the table, looking straight down: in bin 90% on the carry. Ilia's rig (LeHome) has it lower, in front of the arms and tilted. Placed that way, is the carry as good?

## Intuition
The real rig will have the camera where Ilia's is, so the policy should train with that view. Closer and tilted, the camera sees the bar and the bin bigger, but from a slant.

## Change (one, from B008's top condition)
Env `SO101HandoverCarryTopCamLeHome-v1` (`LeHomeTopCamera` in `envs/bimanual_base.py`), from LeHome's scene file (`lehome-challenge/.../garment_bi_cfg_v2.py`): centred between the arms, 44 cm in front of their bases, 56 cm above the table, looking down and back toward the robots at 19° from vertical; lens 67.2° × 52.9° (4:3, focal 28.7 mm, aperture 38.11 mm), squeezed into the 128 × 128 image as a real 4:3 frame resized to square would be. LeHome's README gives ~65 cm for the real rig: to measure on ours.

Pre-check (4 restored carries, 128 × 128): bin 323-390 pixels (90 cm camera: 172-215), bar 44-64 (0-22).

## Method
As B008 (reward v2, γ 0.97, support −25 … 35, 512 envs, 1M steps, evaluation every 250k, `--eval_reconfiguration_freq 0`, the same 2,764 carry starts), seeds 1 and 2. Stage counts at 500k and 1M (seeds 7 + 8, 256 episodes), now also at the episode's last step. Preflight first.

## Prediction
In the bin at least 80% over both seeds (B008's 90 cm camera: 90%).

## Decision rule
- ≥ 80%: B010's top-camera condition uses this placement (it is already queued with it).
- < 80%: read B010's top condition against this; rerun B010's top condition with the 90 cm camera, and look at what the slanted view hides (videos, `who_sees_what.py`).

## Amendment
- **A1, 2026-10-25 19:10, camera roll.** Checked against a real frame from the organisers' recordings (`observation.images.right_front`): the real image has the arms at the bottom edge; this run's camera renders them at the top (LeHome's sim camera is rolled the same way, and LeHome's code rotates its image 180°). Placement and lens are LeHome's; only the roll differs, a fixed rotation of every image, so the run's sim results stand. A policy from it would see the real camera upside down: real-rig runs use `...TopCamLeHome-v2` (rolled to match: arm bases at pixels (31, 109) and (97, 109) of 128, the workspace centre at (64, 63)). Still unmeasured on the real rig: the lens (the real camera is 16:9; LeHome's sim lens is 4:3) and the height (56 cm in LeHome's sim, ~65 cm in its README).

## Results
Preflight 24 of 24 (with B010's two conditions; `experiments/bimanual/out/b009_b010_preflight.log`). Both runs exit 0, commit `06193d4`. Stage counts, 256 episodes per run (seeds 7 + 8); raw `experiments/bimanual/out/stages_b009_*.json`, `b009_counts.txt`.

| At 1M | In bin: ever / at end | Success: ever / at end |
|---|---|---|
| seed 1 | 239 / 207 | 116 / 50 |
| seed 2 | 242 / 218 | 198 / 128 |
| both seeds | 94% / 83% | 61% / 35% |
| B008, 90 cm camera (recount), both seeds | 88% / 79% | 67% / 40% |

At 500k: seed 1 in bin 233 (at end 211), success 169 (at end 96); seed 2 in bin 219 (191), success 114 (68).

**Prediction:** held, 94% in the bin (83% at the end).

**Reading:** the camera placed as on Ilia's rig puts more bars in the bin than the 90 cm one; clean finishes are similar and seed-dependent. Seed 1 finished less often at 1M than at 500k (success at the end 96 → 50): the finish is not stable, as B008's recount found.

**Decision (rule: ≥ 80% → B010's top condition uses this placement):** kept; B010 is running with it.

