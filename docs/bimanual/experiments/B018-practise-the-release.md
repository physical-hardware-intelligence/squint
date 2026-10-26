# B018: Second policy practising the release: does it fix the chain's weak step?

**Status:** done 2026-10-26 11:25. No, and the reason is not the release: about half of the "handovers" the grasp check counted were fake (a closed gripper pressed against the bar), and those cannot be released without dropping the bar. Seed 2 stopped at 58% of training once that was found. Continued in [B019](B019-real-grasps.md).

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
Seed 1 run 2026-10-26 10:45-11:10, commit `cdeed59`; fidelity gate passed (restored 95% / 91% against own 98% / 96%), preflight 8 of 8. Raw: `experiments/bimanual/out/b018_counts.txt`, `b018_probe_and_grasp_check.txt`.

| 256 episodes (seeds 7 + 8) | B014's carry, same switch (no practice) | **B018 seed 1** |
|---|---|---|
| switched to the second policy | 186 / 199 | 193 |
| **bar in the bin (chained)** | 93 / 105 | **86** |
| from its own saved starts: carry / in the bin | | 120 / 98 (47% / 38%) |

Once the picker was clear the policy finished well (91 successes of 98 in the bin), but it cleared the picker only about half the time, even from its own training starts. In the evaluation video each episode either clears in the first steps or both arms stay around the bar to the end.

**Predictions.** 1 fails (45% of switched, not 75%). 2 fails (86, not 140). 3 holds (45% chained against 38% from the starts).

**Why (measured, no learning).** A scripted release (the picker opens, then backs away; the receiver holds still) from the same saved starts keeps the bar in 50% (control, clean-carry starts: 99%). In every drop the receiver's gripper was fully closed (openness 0.00; held: 0.10-0.11) with its grip point 1.9 cm off the bar's centre line: no bar between the jaws, a closed gripper pressed against its side. `is_grasping` (0.5 N per jaw at up to 110 degrees off the closing direction) counts that as a grasp, so the reward, the handover latch and the counts all paid for it, and the picker's touch was what held the bar up. Requiring the jaws to be stopped by the bar (at least 3% open), B011 at 3M makes 108 real handovers (193 counted), 97% of them become carries, and a scripted release from real-grip starts keeps the bar in 99%.

**Decision.** The release is not the weak step; the receiver's real grasp is. Stopped seed 2 (58% trained) as uninformative and moved to [B019](B019-real-grasps.md). Earlier experiments: their carry and in-bin counts stand (a fake grip cannot carry the bar alone); their handover counts include fake grips.
