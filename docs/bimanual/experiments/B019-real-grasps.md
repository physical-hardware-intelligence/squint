# B019: With a grasp check that rejects fake grips, does the handover policy make more real handovers?

**Status:** protocol written 2026-10-26 11:35, before the runs.

## What B018 found (no learning involved)
- From the saved "picker stopped squeezing" starts, a scripted release (picker opens, backs away; receiver still) keeps the bar in only 257 of 512 (50%); from clean-carry starts, 506 of 512 (`probe_scripted_release.py`).
- Every dropped bar had the receiver's gripper fully closed (openness 0.00; held: 0.10-0.11) with its grip point 1.9 cm off the bar's centre line, outside the bar. The jaws were not around the bar: a closed gripper pressed against its side. `is_grasping` accepts that: it needs 0.5 N on each jaw at up to 110 degrees off the closing direction and nothing about the jaws.
- With the jaws also required to be stopped by the bar (at least 3% open: `GRASP_NEEDS_OPEN_JAWS`), B011 seed 1 at 3M makes 108 real handovers of 256 (old check: 193), and 105 carries (old: 101): 97% of real handovers release cleanly. From real-grip holding starts the scripted release keeps the bar in 309 of 312 (99%).

So the chain's weak step is not the release but the receiver's grasp: about 42% of episodes make a real one. Counts of carries and of bars in the bin in earlier experiments stand (a fake grip cannot carry the bar alone); their handover counts include fake grips.

## Change
Continue B011 seed 1 from its 3M checkpoint with the corrected grasp check, everything else as B011 and B015 (reward v2, γ 0.9, support −20 … 20, wrist cameras): `python train_squint.py --exp_name=b019_s<seed> --env_id=SO101HandoverBarV2Grip-v1 --seed=<seed> --num_envs=512 --total_timesteps=3000000 --eval_freq=250000 --env_background_level 0 50 --env_background_tint 6 --gamma=0.9 --v_min=-20 --v_max=20 --checkpoint=runs/b011_wrist_s1/ckpt_3M.pt`. Control: B015 (the same continuation, old check: carries 101 → 104 → 91).

**What the change does to the current policy's own states (B016's lesson):** its real handovers and everything after them score as before; its fake grips no longer count as a grasp, so they lose the handover latch and the release and carry stages and score as the bar held by the picker at the handover point (at most 5.5-6). No real behaviour loses value.

## Method
Preflight first. At +1.5M and +3M, under the corrected check (env `SO101HandoverBarV2Grip-v1`): stage counts (seeds 7 + 8) and the chain with B014's carry policies (switch on carry, env `SO101HandoverBarV2TopCamLeHome-v2`; a fake grip cannot reach that switch). Training seeds 1 and 2 on the RTX box; seeds 3 and 4 on Explorer A100s when they start.

## Predictions
1. Real handovers at +3M at least 150 of 256 on each seed (B011: 108).
2. Carries at +3M at least 140 of 256 on each seed (B011: 105; B015: 91).
3. Chained, the bar in the bin in at least 130 of 256 for at least one carry policy, on each seed (B014: 99).

## Decision rule
- 1-3 hold: real grasps are the lever; the chain for the real rig is this handover policy plus a carry policy trained on its carries (B014's recipe), then distil.
- 1 holds, 3 fails: the carry policy does not handle the new handovers; retrain it on them (B014's recipe).
- 1 fails: the receiver does not learn the real grasp from this signal; practise the grasp itself (saved starts with the bar at the handover point and the receiver near).

## Results
_Not run._
