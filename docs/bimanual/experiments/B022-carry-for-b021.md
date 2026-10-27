# B022: A carry policy for B021 seed 2's handovers

**Status:** done 2026-10-27 12:42. The best chain so far: B021 seed 2 then this carry policy puts the bar in the bin in 133 of 256 episodes (52%; before: 99).

## Why
B021 seed 2 is the best handover policy so far: 168 real handovers and 152 carries of 256. Chained with B014's carry policies, though, only 20 of 256 reach the bin. Those policies were trained on B011's handovers. B014's lesson: train the carry policy on starts from the handover policy it will follow.

## Change
B014's recipe with two changes: B021 seed 2 at 6M is the policy to follow, and the grasp check is the corrected one.
1. `harvest_carry_starts.py --checkpoints runs/b021_s2/ckpt_6M.pt --env_id SO101HandoverBarV6Grip-v1 --episodes 15 --num_envs 512 --out runs/carry_starts_b021s2.pt`. As B014, each start is saved 5 steps into a carry and kept only if the carry is unbroken 10 steps later.
2. With `SQUINT_CARRY_BANK=runs/carry_starts_b021s2.pt`: the preflight (restore checks, short training, counter), then seeds 1 and 2: `python train_squint.py --exp_name=b022_top_s<seed> --env_id=SO101HandoverCarryTopCamLeHomeGrip-v2 --seed=<seed> --num_envs=512 --total_timesteps=1000000 --eval_freq=250000 --env_background_level 0 50 --env_background_tint 6 --eval_reconfiguration_freq=0 --gamma=0.97 --v_min=-25 --v_max=35`.
3. Per seed, 256 episodes (seeds 7 + 8), two measurements:
   - stage counts from the saved starts (the carry env);
   - the chain: B021 seed 2, then this policy (`chain_eval.py`, env `SO101HandoverBarV2TopCamLeHome-v2`, switch on carry).

All on Explorer A100s.

## Predictions
1. Chained, the carry policy puts the bar in the bin in at least 80% of the episodes that switch to it, on each seed (B014: 90-92%).
2. End to end, the bar is in the bin in at least 130 of 256 episodes on at least one seed (the best so far: 99, B011 + B014).

## Decision rule
- **Both hold:** B021 seed 2 + B022 is the simulation baseline to beat. The recipe also carries over to the real rig once the top camera is calibrated.
- **1 fails:** watch the failed carries in video before changing anything (B014's rule).

## Amendment A1 (2026-10-27 11:40, before the runs)
Run on the RTX box, not Explorer. The box came back idle while Explorer's queue had not started the jobs (cancelled there). B014, whose recipe this is, ran on the box, and B019 A1 ruled the machine out as a cause of different results. B021 seed 2's checkpoint was copied from Explorer, SHA-256 checked.

## Results
Run 2026-10-27 11:44-12:42 on the RTX box, tree `db051bc`. Harvest: 4,492 carry starts, every one of the 512 env indices covered (median 9). Preflight 8 of 8. Raw: `experiments/bimanual/out/b022_counts.txt`.

| 256 episodes (seeds 7 + 8) | carry s1 | carry s2 | B011 + B014 (before) |
|---|---|---|---|
| from its saved starts: in the bin / at the end | 216 / 183 | 224 / 186 | 239 / 221 |
| chained: handovers / switched | 164 / 155 | 173 / 162 | 197 / 108 |
| **chained: in the bin (all episodes)** | **123** | **133** | 99 |
| chained: of the switched, in the bin | 79% | 82% | 90-92% |
| chained: success ever / at step 120 | 87 / 63 | 73 / 37 | 57-79 / 36-51 |

**Predictions.** 1 fails on seed 1 by a hair (79%, 80% needed) and holds on seed 2 (82%). 2 holds: 133 of 256 in the bin on seed 2 (52%; the best before was 99).

**Reading.** This is the best end-to-end result so far: B021 seed 2 hands over, then this carry policy takes the bar to the bin in 133 of 256 episodes. Two gaps remain:
- The carry is weaker on these handovers than B014's was on B011's (79-82% of switched episodes, against 90-92%).
- The finish is unclean: of 133 in the bin, 114 are still there at step 120, and 37 succeed (bar still, nothing touching, arms still).

**Decision.** By the rule (1 fails on one seed), watch the failed carries in video before changing anything. Chain videos: `chain_eval.py --video`.
