# B020: Paying the receiver to come in with an open gripper

**Status:** done 2026-10-27. The shut approach is gone on every seed (106 to 0); the receiver then hovers open beside the bar, and two of four seeds lost the release. Continued in [B021](B021-scratch-v6.md).

## Why
B019 A2: in 106 of 256 episodes the receiver came in with its gripper already shut and pressed it against the bar's side. All 106 arrived shut; most real grasps arrived open. v2 pays the reach by the grip point's distance to a point on the bar's centre line, so a shut gripper resting on the face loses only 0.125 of 2. The receiver is not missing its aim; nothing tells it to open.

## Change
Reward v5 = v2 with stage 3's reach multiplied by 1 − w + w × min(openness / 0.15, 1), w = 0.5. A shut gripper earns half the reach; one open wider than the bar earns all of it. Nothing else changes. Env `SO101HandoverBarV5Grip-v1` (v5 and the corrected grasp check).

| Receiver 2 cm from its end of the bar | v2 | v5 |
|---|---|---|
| gripper shut | 5.80 | 4.90 |
| gripper open (15% or more) | 5.80 | 5.80 |
| both hold | 6.1 | 6.1 |

- **0.15 and no geometric exemption, set from seed 9 (not a check seed; `reward_probe_v5.py --design`).** Real grips stop at 10-11% open. Real approaches run at a median 22% and stay above 15% in 95% of the steps before the grip. A first draft at 30% would have cut most real approaches (B016's mistake). A draft exemption for "jaws around the bar" was dropped: shut grippers touching the bar sometimes register within 0.8 cm of its centre line, so it would pay them in full.
- **No new shortcut (B012).** v5 never pays more than v2 in any state, and the same off the reach steps. Stage 3 stays at or above 4, stage 2's top, so bringing the bar to the handover point never pays less than not. Unchanged from v2 and worth watching: the best hover (6.0) is only 0.1 below both holding (class 4 in the probe).

## Checks before training
1. `reward_probe_v5.py`, seeds 7 and 8, on B011 seed 1 at 3M's own episodes; every check must pass on both:
   - C1: v5 never above v2; equal off the reach steps and with the gripper open.
   - C2 (B016): real handovers lose at most 0.1 per reach step (median over episodes), and on the 3 steps before the grip at most 0.1 in at least 90% of them.
   - C3: shut-beside episodes lose at least 0.5 per reach step (median).
   - C4: no step loses more than 2w = 1.
2. `check_reward.py SO101HandoverBarV5Grip-v1` (constructed states) and the preflight.

Training starts only when all pass and B019 A1 is in.

## Training
Continue B011 seed 1 from its 3M checkpoint, everything else as B019: `python train_squint.py --exp_name=b020_s<seed> --env_id=SO101HandoverBarV5Grip-v1 --seed=<seed> --num_envs=512 --total_timesteps=3000000 --eval_freq=250000 --env_background_level 0 50 --env_background_tint 6 --gamma=0.9 --v_min=-20 --v_max=20 --checkpoint=runs/b011_wrist_s1/ckpt_3M.pt`. Four seeds: two on the RTX box, two on Explorer A100s; all four on Explorer if A1 leaves the box as the suspect.

## Method
At +1.5M and +3M, seeds 7 + 8 (256 episodes): stage counts under the corrected check, the A2 probe's classes, carries against real handovers, and the chain with B014's carry policies (switch on carry, env `SO101HandoverBarV2TopCamLeHome-v2`).

## Predictions
Each holds if it holds at +3M on at least 3 of the 4 seeds (B019 showed seeds can split).
1. Shut beside the bar (class 5) at most 40 of 256 (B011: 106).
2. Real handovers at least 150 of 256 (B011: 108-111).
3. The release is kept: carries at least 90% of real handovers (B011: 97%).
4. Chained, the bar in the bin in at least 130 of 256 for at least one carry policy (B011: 99).

## Decision rule
- 1-4 hold: v5 is the receiver's reward. The chain for the real rig is this policy plus a carry policy (B014's recipe, retrained on its carries if needed), then distil.
- 1, 2 and 3 hold, 4 fails: retrain the carry policy on the new handovers.
- 1 holds, 2 fails: the receiver opens but fails another way; rerun the A2 probe on the new policy and address its top class.
- 1 fails: the signal is too weak in 3M steps; next w = 1 (no reach without an open gripper) or grasp practice from near starts.
- 3 fails on some seeds: the release loss is not caused by the reward; test plain continuation (B015's setup, v2) on several seeds.

## Amendment A1 (2026-10-26 23:45, before any training)
- **Checks so far** (`experiments/bimanual/out/b020_checks.txt`): `reward_probe_v5.py` passes C1-C4 on seeds 7 and 8. Real handovers lose 0.0 per reach step (median), shut-beside episodes 0.83, and there is no dip at the grip in 52 of 52 and 54 of 54. `check_reward.py` passes 18 of 18. The box preflight passed 5 of 7: its short training crashed on a PhysX GPU allocation with three other jobs on the 16 GB card, a resource failure, not the code.
- **Where.** B019 A1 so far: box seed 3 lost the release by +1.5M (4 real handovers, 0 carries), as seeds 1 and 2 did there; Explorer's seed 3 kept it. Explorer's seed-3 checkpoint scores the same on the box (108 real handovers, 107 carries), so the box measures correctly and the difference is in training there. Both A1 outcomes lead to Explorer (machine suspect: Explorer only; chance: several seeds anywhere), so all four seeds run on Explorer A100s and are queued now, since the queue wait is hours.
- **Gate.** The preflight runs on an A100 as its own job (`hpc/b020_preflight.sbatch`); the four training seeds start only if it exits 0 (`--dependency=afterok`).

## Amendment A2 (2026-10-27 00:35, before any training)
The A100 preflight (job at 03:24 Explorer time) passed 7 of 8. The failure was "short training exits 0": the trainer finished, saved its checkpoints, printed its own "Cleanup complete. Exiting." and then segfaulted at interpreter exit. That is the known Explorer exit crash (B019's Explorer runs: EXIT 139 after saving, results valid). The dependency then cancelled the four training seeds. `preflight.py` now passes that one case only: exit 139 (or signal 11) with the training log ending on the trainer's cleanup line. Preflight and seeds resubmitted.

## Results
Run 2026-10-27 00:48-03:37 PDT on Explorer A100s, commit `c9ece42`, preflight 8 of 8 (after A2). Raw: `experiments/bimanual/out/b020_counts.txt`.

| 256 episodes (seeds 7 + 8), +3M | Target | B011 (start) | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---|---|---|---|---|---|
| shut beside the bar (class 5) | ≤ 40 | 106 | **0** | **0** | **0** | **0** |
| open, never closes (class 4) | | 1 | 59 | 195 | 120 | 77 |
| real handovers | ≥ 150 | 108 | 107 | 55 | 78 | 106 |
| carries (share of real handovers) | ≥ 90% | 105 (97%) | 97 (91%) | 1 | 2 | 101 (95%) |
| chained, bar in the bin (best carry) | ≥ 130 | 99 | 82 | 0 | 0 | 93 |

**Predictions.** 1 holds on all four seeds. 2 fails on all four. 3 holds on 2 of 4 and so fails. 4 fails.

**Reading.** v5 removed the shut approach entirely; the receiver then hovered open next to the bar instead. Gripping paid only 0.1 more than hovering, while a missed close now costs the shut share (about 1 point per step until it reopens). Seeds 2 and 3 also lost the release, as continuing B011 did in B019.

**Decision.** By the rule (1 holds, 2 fails): the receiver opens but fails another way, and its top class is hovering (4). Addressed in [B021](B021-scratch-v6.md) (reward v6, trained from scratch).
