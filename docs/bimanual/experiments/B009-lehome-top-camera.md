# B009: Does the top camera placed as on Ilia's rig work as well on the carry?

**Status:** protocol written 2026-10-25, before the runs.

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

## Results
_Not run._
