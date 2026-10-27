# B023: Why the receiver hovers, and reward v7

**Status:** step 0 done 2026-10-27 11:44: the hover is in the wrong place (a scripted close from it grasps 8% of the time). Amendment A1 (reward v7, checks passed) written 12:01, before its runs.

## Why
v5's open share stopped the receiver pressing a shut gripper on the bar: that failure fell from 106 of 256 to 0 on every seed of B020 and B021. In 3 of 4 B021 seeds, though, the receiver now hovers open next to the bar instead (81-95 of 256).

Hypothesis, from reading the reward: the open share also charges a failed close at the bar. A gripper that closes and misses is a shut gripper near the bar, which costs about 1 point per step until it reopens. A real grip gains only 0.61 over a hover, so trying looks worse than waiting.

## Step 0: is hovering a choice? (no learning)
`probe_scripted_close.py` runs on B021 seeds 1, 3 and 4 (the hovering ones) and seed 2 (control), on episode seeds 7 + 8. The first time an episode hovers for 5 steps in a row, the script takes over its receiver for 10 steps: arm held still, gripper closing at the full rate. Hovering means the picker holds the bar at the handover point, and the receiver's grip point is within 3 cm of its end of the bar with its gripper at least 15% open, with no both-hold yet.

Reading:
- **A real grasp in at least half of the takeovers on the hovering seeds:** the hover is in place, and the policy chooses not to close. v7 then makes the try free near the bar. The open share charges only on the way in, fading out between 4 and 2 cm from the bar's end, so a failed close at the bar costs no more than under v2. v6's pay for the grip stays.
- **Fewer than half:** the hover is not in a position to grasp, and closing cannot help. v7 must then pay the position (jaws around the bar), not the attempt.

## Step 0 results
Run 2026-10-27 11:40-11:44 on the RTX box (tree `db051bc`). Raw: `experiments/bimanual/out/b023_step0.txt`.

| B021 at 6M, episode seeds 7 + 8 | hovers taken over | real grasp | shut on air | shut beside the bar |
|---|---|---|---|---|
| seeds 1, 3, 4 (hovering) | 250 | **21 (8%)** | 182 (73%) | 47 (19%) |
| seed 2 (control) | 27 | 11 (41%) | 6 | 10 |

**Reading (fewer than half).** The hover is not in a position to grasp: closing from it mostly closes on air. The receiver stops near the bar's end without its jaws around the bar, so making a try free would not help. v7 must pay the position, jaws around the bar, and not the attempt. The hypothesis written under Why (a failed close costs more than a grip gains) is not the main problem: closing from where it hovers would fail anyway. Seed 2 hovers far less (27 against 250 across the other three), and its hovers are closer to grippable.

## Amendment A1: reward v7 (written 2026-10-27 12:01, before the runs)
**Design data** (seed 9, not a check seed; `probe_scripted_close.py` with the grip pose at each takeover; raw: `experiments/bimanual/out/b023_design_seed9.txt`). The closes that grasped started about 1 cm from the receiver's target, the point on the bar's centre line 1.2 cm in from its end: medians 1.05-1.11 cm on seeds 2 and 3. The ones that closed on air started at 1.6-2.2 cm. The other measures overlap between the two groups (distance from the centre line, depth in from the end, angle, openness). v6 paid 1 cm only 0.09 more than 2 cm: nothing pulled the receiver the last centimetre.

**Change.** Reward v7 = v6 with stage 3's reach split in two: pull(gap) + (1 − tanh(gap / 2 cm)), each worth 1, still multiplied by v5's open share (v2-v6: 2 × pull(gap)). Half the reach now pays the approach and half the last centimetres. Env `SO101HandoverBarV7Grip-v1`. Constructed states (`reward_probe_v7.py`, raw `experiments/bimanual/out/b023_checks.txt`):

| Receiver, open, this far from its target | v6 | v7 |
|---|---|---|
| 30 / 10 / 5 cm | 4.19 / 5.08 / 5.51 | 4.09 / 4.54 / 4.77 |
| 3 / 2 / 1 / 0.5 / 0 cm | 5.70 / 5.80 / 5.90 / 5.95 / 6.00 | 4.95 / 5.14 / 5.49 / 5.73 / 6.00 |
| both hold at the handover point | 6.56 | 6.56 |

The 2 cm to 1 cm step now pays 0.35 (v6: 0.09). The pull from far away halves (10 cm: 4.54 against 5.08); watch class 3, "the receiver never within 3 cm".

**Checks (passed).**
- `reward_probe_v7.py` passes S1-S7: v6's five checks; S6, 1 cm pays at least 0.3 more than 2 cm; S7, the reach rises at every step in from 30 cm.
- `check_reward.py SO101HandoverBarV7Grip-v1`: 18 of 18.
- An A100 preflight gates the training (job dependency).

**Training.** As B021 with the v7 env: from scratch, 6M steps, seeds 1-4 on Explorer A100s: `python train_squint.py --exp_name=b023_s<seed> --env_id=SO101HandoverBarV7Grip-v1 --seed=<seed> --num_envs=512 --total_timesteps=6000000 --eval_freq=250000 --env_background_level 0 50 --env_background_tint 6 --gamma=0.9 --v_min=-20 --v_max=20`. Measured as B021 at 3M and 6M: stage counts, probe classes, carries, and the chain with B014's carry. The carry does not take new handovers (B021), so the chain is not a prediction here; B022's recipe follows for the best seed.

**Predictions** (at 6M, each holds if it holds on at least 3 of 4 seeds):
1. Hovering (class 4) at most 40 of 256 (B021: 17-95, three seeds above 80).
2. Real handovers at least 150 of 256 (B021: only seed 2, 168).
3. The receiver still comes in open: shut beside the bar (class 5) at most 40, and never within 3 cm (class 3) at most 40 (B021: 18-37).

**Decision rule.**
- 1 and 2 hold: v7 is the handover reward. Train B022's carry on its best seed, and move on to the release (3 of 13 runs or more lost it) and to the real rig.
- 1 holds, 2 fails: the receiver now closes but misses or fails another way; run the A2 probe on it.
- 1 fails: the last-centimetre pay is not the lever. Next: practice from saved starts with the receiver near the bar (the plan B019 wrote down).
- 3 fails: halving the far pull lost the approach. Restore it, and pay the last centimetres from a wider band.

## Results
_Not run._
