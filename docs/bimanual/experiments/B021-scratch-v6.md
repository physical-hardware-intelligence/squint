# B021: The handover trained from scratch with reward v6

**Status:** done 2026-10-27. Seed 2 is the best handover policy so far (168 real handovers, 152 carries of 256); the other three hover beside the bar or lost the release.

## Why
- **B020 (v5, continued from B011) fixed its target and exposed the next problem.** On every seed measured, the shut-gripper approach went from 106 of 256 to 0. The receiver then hovered open beside the bar instead (59 to 211 of 256). Gripping paid 6.1 against the best hover's 6.0, and under v5 a missed close costs.
- **Continuing B011 loses the release in most runs.** B020 seeds 2 and 3 carried nothing. B019 lost the release in 4 of 6 runs, and the old-check control's seed 2 lost it as well by +1.5M. Changing the reward under a trained policy is the common factor, so this run starts from scratch.

## Change
**Reward v6 = v5 with stage 4 (both hold) changed.** It pays 6 + 0.5 × (1 − tanh(5 × the bar's distance to the handover point)) + 0.5 × the picker's opening. v2 and v5 paid 6 + the picker's opening. Half a point now goes to where the bar is held. It is full at the handover point and fades with distance, unlike v3's flat pay anywhere. Env: `SO101HandoverBarV6Grip-v1` (v6 and the corrected grasp check). Constructed states, from `reward_probe_v6.py`:

| State | v2 | v5 | v6 |
|---|---|---|---|
| receiver 20 cm out, open / shut | 4.48 / 4.48 | 4.48 / 4.24 | 4.48 / 4.24 |
| receiver at the bar, open (hover) / shut | 5.95 / 5.95 | 5.95 / 4.98 | 5.95 / 4.98 |
| both hold, bar 0 / 7 / 12 / 25 cm from the handover point | 6.11 everywhere | 6.11 everywhere | 6.56 / 6.39 / 6.29 / 6.13 |
| both hold, picker 50% / 100% open | 6.50 / 7.00 | 6.50 / 7.00 | 6.75 / 7.00 |
| handed over, a picker jaw still on / picker clear | 7.71 / 8.71 | 7.71 / 8.71 | 7.71 / 8.71 |

**Trained from scratch** (no checkpoint): `python train_squint.py --exp_name=b021_s<seed> --env_id=SO101HandoverBarV6Grip-v1 --seed=<seed> --num_envs=512 --total_timesteps=6000000 --eval_freq=250000 --env_background_level 0 50 --env_background_tint 6 --gamma=0.9 --v_min=-20 --v_max=20`. Seeds 1-4 run on Explorer A100s. The budget is 6M steps because from scratch B011 seed 1 went from 1 carry at 1.5M to 101 at 3M, while seed 2 had 13 handovers at 3M.

## Checks before training
- **`reward_probe_v6.py`: all five pass** (raw: `experiments/bimanual/out/b021_checks.txt`).
  - S1: a grip at the handover point beats the best hover by 0.61.
  - S2: the grip pays less with distance, and 25 cm off it pays only 0.02 over v2.
  - S3: the release never pays less as it goes.
  - S4: open beats shut at the same spot.
  - S5: reach states stay at 4 or above.
- **`check_reward.py SO101HandoverBarV6Grip-v1`: 18 of 18.**
- **No rollout check (B016's).** From scratch there is no trained policy whose states could lose value.
- **The A100 preflight is a job dependency.** The four seeds start only if it exits 0.

## Method
At 3M and 6M, on seeds 7 + 8 (256 episodes):
- stage counts under the corrected check;
- the A2 probe's classes;
- carries against real handovers;
- the chain with B014's carry policies (switch on carry, env `SO101HandoverBarV2TopCamLeHome-v2`).

## Predictions
Each holds if it holds at 6M on at least 3 of the 4 seeds.
1. **The receiver grips cleanly:** shut beside the bar (class 5) at most 40 of 256, and open but never closing (class 4) at most 40.
2. **Real handovers at least 150 of 256** (B011 at 3M: 108).
3. **The release is kept:** carries are at least 90% of real handovers.
4. **Chained, the bar in the bin in at least 130 of 256** for at least one carry policy (B011 + B014: 99).

## Decision rule
- **1-4 hold:** v6 from scratch is the handover recipe. Next: a carry policy on its carries if needed (B014's recipe), then distil the chain, then the real-rig steps.
- **2 fails because seeds never learn the handover at all** (as B011 seed 2): the bottleneck is learning from scratch, not the grasp's pay. Next: more steps, or start some episodes with the bar already at the handover point.
- **1 fails:** the grasp's pay is still wrong; the probe's top class says how.
- **3 fails:** the release is lost even from scratch, so the release reward itself needs work.

## Results
Run 2026-10-27 on Explorer A100s, commit `f6d3141`, preflight 8 of 8. Seeds 1-2 ran 03:02-05:52 PDT, seed 3 03:44-06:31, seed 4 07:09-09:56. Raw: `experiments/bimanual/out/b021_counts.txt`.

| 256 episodes (seeds 7 + 8), 6M | Target | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---|---|---|---|---|
| shut beside the bar (class 5) | ≤ 40 | 0 | 0 | 0 | 0 |
| open, never closes (class 4) | ≤ 40 | 86 | **17** | 81 | 95 |
| real handovers | ≥ 150 | 103 | **168** | 93 | 110 |
| carries (share of real handovers) | ≥ 90% | 98 (95%) | **152 (90%)** | 78 (84%) | 8 (7%) |
| chained, bar in the bin (best carry) | ≥ 130 | 12 | 20 | 0 | 1 |
| at 3M: real handovers, carries | | 115, 112 | 144, 130 | 111, 3 | 78, 3 |

**Predictions.** 1 fails: the shut approach is gone on every seed, but hovering stays above 40 on three of them. 2 fails: only seed 2 reaches 150. 3 fails: it holds on 2 of 4, and seed 4 lost the release from scratch (8 carries from 110 handovers). 4 fails on all.

**Reading.**
- Seed 2 is the best handover policy so far: 168 real handovers and 152 carries of 256 (B011: 108 and 105). The same reward gave three weaker seeds.
- Hovering survives v6 on 3 of 4 seeds. v6 pays a grip 0.61 more than a hover. But v5's open share still charges about 1 point per step for a shut gripper near the bar, and that is exactly what a failed close is. A close that misses costs more than one that works gains, so hovering is the safe choice. This explanation comes from reading the reward and is not yet measured `[UNVERIFIED]`.
- The release is lost from scratch too (seed 4), so it is not only a continuation effect.
- The chain fails because B014's carry policies were trained on B011's handovers. They switch in 74-84 episodes on seed 2, yet put at most 14 in the bin per 128. The carry has to be trained on the handover policy's own handovers (B014's lesson).

**Decision.** By the rule: 1 fails, so the grasp's pay is still wrong, and the top class is again hovering. 3 fails, so the release needs work in its own right. The next steps are proposed in the README and not yet run.
