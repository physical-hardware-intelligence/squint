# B010: Does the whole task learn with reward v2, from wrist cameras or with a top camera?

**Status:** done 2026-10-25 21:30. The handover stalls at γ 0.97 on both seeds and with either camera (3M: 29, 53 and 14 of 256; B003: 128 and 88), so the carry is barely reached. Top seed 2 not run.

## Question
B008 learned the carry alone with reward v2. Trained from scratch on the whole task (grasp, hand over, carry, place), does the bar reach the bin, with the wrist cameras alone or with the top camera placed as on Ilia's rig?

## Intuition
B003 learned the handover but never the carry; B008 showed the carry needed reward v2, not new inputs. Put the two together. The open risk: every earlier run that added an input (bin position, first frame) lost the handover.

## Changes from B003 (two, on purpose)
Reward v2 and γ 0.97 (critic support −25 … 35): B008 showed v2 is the fix and γ 0.97 never worse. This is the setting to take forward, not a test of each change on its own.

| Condition | Env | Inputs |
|---|---|---|
| wrist | `SO101HandoverBarV2-v1` | the two wrist cameras (as B003) |
| top | `SO101HandoverBarV2TopCamLeHome-v1` | + top camera placed as LeHome's ([B009](B009-lehome-top-camera.md)) |

## Method
As B003 otherwise: 512 envs, 3M steps, evaluation every 250k on 16 envs (scene rebuilt per episode, as B003), backgrounds 0-50 ±6, colour jitter on. Seeds 1 and 2, order wrist s1, top s1, wrist s2, top s2. Stage counts at 1.5M and 3M (seeds 7 + 8, 256 episodes): handovers, carries, in the bin and success (at any step and at the last step), closest bar-to-bin. Preflight first. Reference: B003 at 3M (v1, γ 0.9, wrist), 256 episodes each: seed 1 handovers 128, carries 103 (B005 A1's recount); seed 2 handovers 88, carries 66 (B007); in the bin 0 on both.

## Predictions
1. wrist: the bar in the bin in at least 26 of 256 (10%) on at least one seed at 3M; success above 0.
2. wrist: handovers at least 70 of 256 on each seed (B003: 128 and 88).
3. top: handovers at least half of wrist's on the same seed. Below that, the extra-input pattern holds even with v2.
4. top: in the bin at least wrist's, on each seed where 3 holds.

## Decision rule
- 1 holds: the whole task works in simulation; next, longer training for success, then the real rig.
- 3 fails: an added input costs the handover again; find out why (an input of pure noise as a control) before the real rig uses a top camera.
- 1 fails with carries present: the whole task does not reach the carry enough to learn it; next, mix saved carry starts into whole-task training (curriculum, then SGS).

## Amendment
- **A1, 2026-10-25 19:10, camera roll.** Checked against a real frame from the organisers' recordings (`observation.images.right_front`): the real image has the arms at the bottom edge; this run's camera renders them at the top (LeHome's sim camera is rolled the same way, and LeHome's code rotates its image 180°). Placement and lens are LeHome's; only the roll differs, a fixed rotation of every image, so the run's sim results stand. A policy from it would see the real camera upside down: real-rig runs use `...TopCamLeHome-v2` (rolled to match: arm bases at pixels (31, 109) and (97, 109) of 128, the workspace centre at (64, 63)). Still unmeasured on the real rig: the lens (the real camera is 16:9; LeHome's sim lens is 4:3) and the height (56 cm in LeHome's sim, ~65 cm in its README).

## Results
Run 2026-10-25 17:48-21:30, commit `06193d4`, preflight 24 of 24. Command (wrist seed 1): `python train_squint.py --exp_name=b010_wrist_s1 --env_id=SO101HandoverBarV2-v1 --seed=1 --num_envs=512 --total_timesteps=3000000 --eval_freq=250000 --env_background_level 0 50 --env_background_tint 6 --gamma=0.97 --v_min=-25 --v_max=35`; top: `--env_id=SO101HandoverBarV2TopCamLeHome-v1`; seed 2: `--seed=2`. Raw: `experiments/bimanual/out/b009_b010_counts.txt`, `stages_b010_*.json`.

| 256 episodes (counting seeds 7 + 8) | wrist s1 | top s1 | wrist s2 | B003 at 3M (s1 / s2) |
|---|---|---|---|---|
| handovers at 1.5M | 3 | 13 | 5 | |
| handovers at 3M | **29** | **14** | **53** | 128 / 88 |
| carries at 3M | 2 | 0 | 0 | 103 / 66 |
| in the bin, success | 0, 0 | 0, 0 | 0, 0 | 0, 0 |

Top seed 2 was not run: the queue was reordered at 19:41 (B011 first) and at 21:30 (B012), since both conditions stalled at the handover, before the camera could matter; its `-v1` camera is also rolled upside down (A1).

**Predictions.** 1 fails (0 in the bin). 2 fails (29 and 53, below 70). 3: top 14 against the 14.5 needed on seed 1, a miss by one handover at counts too low to separate the conditions (at 1.5M top led, 13 vs 3). 4 not reached.

**Decision.** Rule 3 formally triggers, but its premise (wrist hands over, top does not) is absent: both stalled. Not acted on; the top camera is retested once the handover works, with the calibrated camera. Rule "1 fails with carries present" does not apply: 2 carries in 768 episodes. The stall is the handover itself. Before the handover v1 and v2 are identical (8,000 states), so γ is the one difference from B003 there: tested in [B011](B011-full-task-gamma-090.md).
