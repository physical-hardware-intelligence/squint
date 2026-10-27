# B019: With a grasp check that rejects fake grips, does the handover policy make more real handovers?

**Status:** done 2026-10-26 22:45 (protocol 11:20, before the runs). No: real handovers did not rise on any seed, and two of four seeds lost the release.

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
Seeds 1-2 on the RTX box 2026-10-26 11:25-13:21 (commit `d922b5a`, preflight 8 of 8); seeds 3-4 on Explorer A100s 12:13-16:46 PDT (commit `85f1b13`, docs-only change). Raw: `experiments/bimanual/out/b019_counts.txt`, `b019_release_failures.txt`.

| 256 episodes (seeds 7 + 8), +3M | Target | B011 (start) | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---|---|---|---|---|---|
| real handovers | ≥150 | 108 | 44 | 59 | 109 | 100 |
| carries | ≥140 | 105 | 0 | 0 | 106 | 99 |
| **bar in the bin, chained** (best of B014's two carry policies) | ≥130 | 99 | **0** | **0** | **97** | **88** |

At +1.5M: real handovers 39, 0, 107, 100; carries 0, 0, 105, 99.

**Predictions.** 1, 2 and 3 fail on every seed.

**Two outcomes, not one.** Seeds 3 and 4 stayed where B011 started. Seeds 1 and 2 lost the release: after a real handover the picker keeps the bar (seed 7 episodes: seed 1, 17 of 18 handovers end with the picker alone holding; seed 2, 14 with both still holding to the end and 14 with the picker alone; B011, 2 of 55 fail at all). The box measures correctly (B011 re-measured there: 55 real handovers, 54 carries on seed 7), and the training return stayed flat on all four seeds (32-41), so the return did not show the loss. Not known: whether the split is the seed or the machine. The two bad seeds are exactly the two run on the box; B016 agreed across machines.

**Decision.** By the rule: the receiver does not learn the real grasp from the corrected check alone; practise the grasp itself. Before that, train seed 1 on Explorer and seed 3 on the box: if the release loss follows the seed, continuing B011 is unstable and grasp practice needs several seeds; if it follows the machine, results from the two machines cannot be pooled until that is found.

## Amendment A1: seed or machine? (written 2026-10-26 22:48, before the runs)
The two seeds that lost the release are the two that ran on the RTX box. Train one more seed on each machine, same command and measurements as above: seed 1 on an Explorer A100, seed 3 on the box. "Lost the release" = at +3M, carries below half of real handovers (seeds 7 + 8). So far: kept 106/109, 99/100 (Explorer); lost 0/44, 0/59 (box).

GPU physics is not bit-reproducible across machines, so "the same seed" does not replay the same run. Reading:
- Box seed 3 loses it and Explorer seed 1 keeps it: the machine stays the suspect (3 of 3 box runs against 0 of 3). Next: compare the two installs (package versions, driver) and the same checkpoint's counts on both.
- Any other outcome: the machine is ruled out as the cause; the loss is run-to-run chance when continuing B011, and B020 must run on several seeds and check the release on each.

## Amendment A2: why does the receiver miss? (written 2026-10-26 22:48, before the run)
No learning. B011 seed 1 at 3M (B020's starting policy) runs its own episodes under the corrected check (`SO101HandoverBarV2Grip-v1`, seeds 7 + 8, 128 episodes each, settings as `stage_counts.py`). `probe_receiver_grasp.py` puts each episode in the first class that fits:
1. real handover (the receiver's corrected grasp while both hold the lifted bar);
2. the bar never reaches the handover point;
3. it does, but the receiver's grip point never comes within 3 cm of its end of the bar;
4. it comes within 3 cm, but the receiver's gripper never closes below 15% open (a real grip stops at 10-11%);
5. it closes near the bar and ends shut (below 3%) touching the bar: closed beside it;
6. it closes near the bar and ends shut not touching it: closed on air;
7. anything else.

At the receiver's first close near the bar it records, by class: the grip point's distance from the bar's centre line (the bar is 1.25 cm from centre to face), its depth in from the bar's end, the angle between the jaws' closing direction and the bar (90 degrees = across it), and the closes per episode.

**Why look (from the reward, not yet measured).** Stage 3 pays the receiver 2(1 − tanh(5 × gap)), the gap from its grip point to a point on the bar's centre line 1.2 cm in from the end. A closed gripper pressed against the bar sits at least 1.25 cm off that line and loses 0.125 of 2 compared with jaws around the bar; a real grasp pays +0.1 over a perfect hover while the picker still holds. If classes 5 and 6 dominate, the reward does not tell the receiver how to grasp, and B020 changes what the receiver is paid for near the bar, not only where it starts. If 3 dominates, the receiver does not get there and practice from near starts is the fix. If 4 dominates, it hesitates.
