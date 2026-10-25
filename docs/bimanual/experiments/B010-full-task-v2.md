# B010: Does the whole task learn with reward v2, from wrist cameras or with a top camera?

**Status:** protocol written 2026-10-25, before the runs.

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

## Results
_Not run._
