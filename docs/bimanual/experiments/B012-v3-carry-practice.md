# B012: Does the whole task reach the bin with reward v3 and carry practice?

**Status:** running. Protocol written 2026-10-25 before the runs; preflight 17 of 17 (21:31, tree at `7ff1f57`); A s1 started 21:38.

## Question
B010/B011: at γ 0.97 the handover stalls (29 handovers); at γ 0.9 it works (193) but the carry stalls (31 cm from the bin, 0 in it). With a handover step that pays a full point (v3) and a quarter of training episodes starting mid-carry, does the bar reach the bin at γ 0.97?

## The two changes, step by step
| Step | Today (v2) | B012 |
|---|---|---|
| receiver grips | +0.16 over hovering (picker's jaws 11.5% open, measured) | +1.10 (reward v3) |
| jaws touch while both hold | −2 | no penalty (wrist/forearm contact still −2) |
| receiver's approach nudges the bar 7 cm | drops to 3.33 (5 cm cliff) | stays 5.46 (within 10 cm) |
| practice at carrying | only after a handover, late in an episode | also 25% of training episodes start mid-carry |

Reward probes (`reward_probe_handover.py`): v3 along the handover 4.36 → 5.46 → 6.56 → 7.71 → 8.71, never down. `check_reward.py` 18/18 on v1, v2, v3; v1 and v2 unchanged by the edit (both probes give the earlier numbers; pre-handover v1 = v2 on 8,000 states). Mixed starts (GPU, 128 envs, 4 resets): 23-31% restored, all restored ones carrying after one still step, the rest normal; with the share at 0 no bank is loaded, even with a scene rebuild every reset.

## Conditions
| | Env | γ, support | Training env |
|---|---|---|---|
| A | `SO101HandoverBarV3-v1` | 0.97, −25 … 35 | `--train_env_kwargs={"carry_start_fraction":0.25}` |
| B | `SO101HandoverBarV3-v1` | 0.97, −25 … 35 | normal starts only |

Both: wrist cameras, 512 envs, 3M steps, evaluation every 250k on normal starts (scene rebuilt per episode), backgrounds 0-50 ±6, colour jitter on. Seeds 1 and 2, order A s1, B s1, A s2, B s2. Stage counts at 1.5M and 3M on normal starts only (seeds 7 + 8, 256 episodes). Preflight first. Commit `d5c7fb6` plus this protocol.

## Predictions
1. A and B: at least 70 handovers of 256 on each seed at 3M (B010 wrist, v2 γ 0.97: 29).
2. A: the bar in the bin in at least 26 of 256 (10%) on each seed at 3M; success at the end above 0.
3. A puts more bars in the bin than B, on each seed.

## Decision rule
- 1 fails: v3 did not fix the handover at γ 0.97; look at the handovers in the videos and the per-step reward.
- 1 holds, 2 fails: more carry practice (share 0.5) or longer training (6M).
- 2 holds: the corrected, calibrated top camera (after Monday), then the real rig.

## Results
_Not run._
