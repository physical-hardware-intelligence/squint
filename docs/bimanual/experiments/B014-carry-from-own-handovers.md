# B014: Does a carry policy trained on the handover policy's own handovers carry them as well as its saved starts?

**Status:** done 2026-10-26 00:45. Trained on B011's own handovers, the carry policy puts 90-92% of the carries it is handed in the bin (B013: 52-71%). Chained: 99 of 256 in the bin on both seeds (39%).

## Question
In B013 the top-camera carry policies put the bar in the bin from 93-95% of their saved starts, but from only 52-71% of the carries B011's handover policy hands them. Their saved starts came from B003's policies. Trained on starts saved from B011's own handovers, does the carry policy keep its rate when chained?

## Change
Carry starts harvested from B011 seed 1 (3M), the policy it will follow; otherwise B009's recipe. The camera is the corrected roll (`-v2`, the real camera's way up), as the real rig will need, so B014 against B013 changes two things; the test below compares B014 with itself.

## Method
1. `harvest_carry_starts.py --checkpoints runs/b011_wrist_s1/ckpt_3M.pt --episodes 15 --num_envs 512 --out runs/carry_starts_b011.pt` (settled 5 steps into a carry, confirmed 10 steps later, as B008).
2. `SQUINT_CARRY_BANK=runs/carry_starts_b011.pt`; preflight (restore checks, short training, counter); then seeds 1 and 2: `python train_squint.py --exp_name=b014_top_s<seed> --env_id=SO101HandoverCarryTopCamLeHome-v2 --seed=<seed> --num_envs=512 --total_timesteps=1000000 --eval_freq=250000 --env_background_level 0 50 --env_background_tint 6 --eval_reconfiguration_freq=0 --gamma=0.97 --v_min=-25 --v_max=35` (B009's command, new env id and bank).
3. Per seed, 256 episodes (seeds 7 + 8): stage counts from the saved starts (`stage_counts.py`, the carry env), and the chain (`chain_eval.py`, B011 seed 1 then this policy, env `SO101HandoverBarV2TopCamLeHome-v2`).

## Predictions
1. Chained, the carry policy puts the bar in the bin in at least 80% of the episodes that switch to it, on each seed (B013: 71% and 52%).
2. That is within 10 points of its own rate from the saved starts.
3. End to end: the bar in the bin in at least 80 of 256 episodes (31%) on each seed (B013 best: 71).

## Decision rule
- 1 and 2 hold: the carry side of the chain is solved; train the real-rig carry policy the same way once the top camera is calibrated. The remaining limit is the handover policy (about 40% of episodes reach a carry).
- 2 fails: the drop is not about whose handovers the starts came from; look at the failed carries in video before changing anything.

## Results
Run 2026-10-25 23:48 - 2026-10-26 00:45, commit `be3000c`. Harvest: 2,831 carry starts, every one of the 512 env indices covered (median 5). Preflight 8 of 8 (restored starts are carries 64 of 64, still 10 steps later 64 of 64, bar slip median 0.03 cm). Raw: `experiments/bimanual/out/b014_counts.txt`.

| 256 episodes (seeds 7 + 8) | carry s1 | carry s2 |
|---|---|---|
| from its saved starts: in the bin / at the end | 239 / 221 | 239 / 215 |
| from its saved starts: success at the end | 109 | 131 |
| **chained: handovers / switched** | 197 / 108 | 205 / 110 |
| **chained: in the bin (all episodes)** | **99** | **99** |
| chained: of the switched, in the bin | **92%** | **90%** |
| chained: success, ever / at step 120 | 57 / 36 | 79 / 51 |

**Predictions.** 1 holds (92%, 90%; 80% needed). 2 holds (93% → 92%, 93% → 90%). 3 holds (99 and 99; 80 needed).

**Decision (rule "1 and 2 hold").** The carry side of the chain is solved: train the carry policy on starts from the handover policy it follows. The real-rig carry policy will be trained this way once the top camera is calibrated. The limit is now the handover policy: only 42-43% of episodes reach a carry. Second: the clean finish (success at the end 36 and 51 of the 99 in the bin).
