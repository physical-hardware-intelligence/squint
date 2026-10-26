# B018: Second policy practising the release: does it fix the chain's weak step?

**Status:** protocol written 2026-10-26 10:30, before the runs.

## Question
The chain's weak step is the release: of the episodes where both arms hold the bar, about half never become a clean carry, and 79 of B011's 99 failed releases end with the picker's opened jaw still on the bar. Switching to B014's carry policy at that moment did not help (93 and 105 in the bin against 99): it had never practised from there. A second policy trained from saved states taken at that exact moment: does it finish the release and the carry, so the chain puts most bars in the bin?

## What is new, and what is not (one change from B014)
- **New:** the saved starts are taken where the receiver grips and the picker has **stopped squeezing**, its jaws possibly still on the bar (`harvest_carry_starts.py --at holding`), not after the picker is already off the bar. Saved when that has held for 6 steps in a row: the same rule and step as the chain's switch (`chain_eval.py --switch_on holding`). No filter on what happens next, so the failures are in the starts.
- **Not new:** reward v2 (no reward change); the policy trains from scratch (nothing to unlearn); starts from the handover policy it will follow (B011 seed 1, 3M); B014's command, env (`SO101HandoverCarryTopCamLeHome-v2`, 60-step episodes, top camera) and 1M steps.

## Method
1. `harvest_carry_starts.py --checkpoints runs/b011_wrist_s1/ckpt_3M.pt --episodes 8 --num_envs 512 --at holding --settle_steps 5 --confirm_steps 0 --out runs/carry_starts_holding_b011.pt`.
2. `SQUINT_CARRY_BANK` set to it; preflight (restore checks now judge the start by its own condition: receiver gripping, picker not squeezing, handover latched, after one still step and 10 still steps; bar within 1 cm of the receiver's gripper; then the short training and the counter).
3. Seeds 1 and 2: `python train_squint.py --exp_name=b018_s<seed> --env_id=SO101HandoverCarryTopCamLeHome-v2 --seed=<seed> --num_envs=512 --total_timesteps=1000000 --eval_freq=250000 --env_background_level 0 50 --env_background_tint 6 --eval_reconfiguration_freq=0 --gamma=0.97 --v_min=-25 --v_max=35`.
4. Per seed, 256 episodes (seeds 7 + 8): stage counts from the saved starts, and the chain: B011 seed 1, then this policy, switched with `--switch_on holding` (env `SO101HandoverBarV2TopCamLeHome-v2`).

Baseline at the same switch with no practice (B014's carry): 93 and 105 in the bin; switched 186 and 199; of the switched, 50% and 53% in the bin.

## Predictions
1. Chained, at least 75% of the switched episodes end with the bar in the bin, on each seed (no practice: 50-53%).
2. End to end, the bar in the bin in at least 140 of 256, on each seed (B014: 99; no practice at this switch: 93, 105).
3. The chained rate is within 10 points of the policy's rate from its own saved starts.

## Decision rule
- 1 and 2 hold: the release is fixed by practising it; next, the same recipe one step earlier (the handover), then distil the chain into one policy.
- From its saved starts the policy places under 70%: releasing from a touching jaw is harder to learn than the carry; look at the videos before anything else.
- 3 fails (saved starts high, chained low): the starts do not match the switch; compare the switch states with the saved ones.

## Amendment
- **A1, 2026-10-26 10:40, after the harvest and before any training.** Harvest: 3,130 starts, every env index covered (median 6). The preflight's restore check failed: held still with zero actions, 59 of 64 restored starts were still holding after one step and 55 of 64 after ten (thresholds 95% and 90%), and one bar flew 129 cm. Those thresholds were set for clean carries, which are static; these starts are mid-release. Compared directly, with B011 seed 1 acting in both (`check_holding_fidelity.py`): its own episodes from the switch moment hold 98% one step later and 95% eleven steps later, with 1 bar jump over 5 cm in 573; the restored starts hold 95% and 91%, with 14 jumps in 768 (1.8%). The restore is 3-4 points less faithful here and about 2% of training episodes start with a physics jump. Accepted: it adds noise to training only, and the chained evaluation that decides the predictions restores nothing. From here, holding banks are gated by `check_holding_fidelity.py` (restored within 5 points of the own episodes at 1 and 11 steps, jumps under 3%); the zero-action checks are printed for information.

## Results
_Not run._
