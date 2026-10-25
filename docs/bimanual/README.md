# Bimanual Squint

One Squint policy drives two SO-101 arms from their two wrist cameras: trained in simulation with RL, no demonstrations, to run on the real pair.

**New here, or adding a task?** Read this page, then [GUIDE.md](GUIDE.md) (files, setup, recipe, lessons).

| | |
|---|---|
| Rig | two SO-101 followers side by side, facing forward, pan axes 46 cm apart; one wrist camera each |
| Policy input | left then right wrist image (16×16, 6 channels); 24 numbers (per arm: 6 noisy joint angles, 6 targets); no memory |
| Policy output | 12 joint deltas, `pd_joint_target_delta_pos` |
| Code | `envs/bimanual_base.py` (two arms, two cameras), `envs/handover.py` (task 1) |

## Tasks

| # | Task | Env | Status |
|---|---|---|---|
| 1 | Handover: one arm picks a 25 × 25 × 60 mm red bar, hands it to the other in mid-air, which puts it in the bin. Either arm may pick; neither can reach both bar and bin | `SO101HandoverBar-v1` | training: grasp, handover and release learned; nothing in the bin yet (B001-B004) |
| 2 | Insertion in mid-air: one arm picks a socket, the other a peg; insert until seated (as ALOHA's Bimanual Insertion). Clearance to decide: 5 mm is below one policy pixel ([GUIDE §6](GUIDE.md#6-task-2-insertion-decide-these-first)) | — | next (Parv); series `I001…` |
| 3 | Hold and open: one arm holds a hinged box, the other lifts the lid past 60° | — | draft |

ALOHA (Zhao et al. 2023) defines Transfer Cube (right arm hands a cube to the left gripper) and Bimanual Insertion in simulation, learned from 50 demonstrations per task with 6-joint arms. Task 1 adds a random direction and the bin; Task 2 is the same concept as Insertion. We use 5-joint arms, one wrist camera per arm, and RL without demonstrations.

## Experiments

| # | Question | Status |
|---|---|---|
| [B001](experiments/B001-handover-env-check.md) | Is the handover doable in simulation; do the stages and success fire in order? | done: layout checks out; scripted grasp fails (0/10) |
| [B002](experiments/B002-handover-train.md) | Does Squint learn task 1, and where does it stall? | stopped at 1.5M: 99% grasp, 13% handovers, 0 releases |
| [B003](experiments/B003-release-payoff.md) | Does paying for the release get past the stall? | done: release learned (39/128 carry at 3M), 0 in bin; bin in the receiver's camera 2% of carry steps, the picker's 40% |
| [B004](experiments/B004-bin-in-state.md) | Is the stall missing information (tell the policy where the bin is)? | inconclusive: neither seed released (0 carries), so the bin question was never tested; from-scratch runs vary too much at the release |
| [B005](experiments/B005-warm-start-bin.md) | From B003's checkpoint (already lets go), does knowing the bin's position get the bar in? | no: the bin input broke the handover; the control carries but moves the bar only ~5 cm toward the bin, where the carry reward is almost flat |
| [B006](experiments/B006-linear-carry.md) | Does a carry reward that pulls at every distance get the bar to the bin? | no: closest 21 cm (control 23), 0 in the bin; the reward is not the main stall |
| [B007](experiments/B007-first-frame-memory.md) | Does remembering the first frame (bin in view) get the bar to the bin? | untested: neither memory seed learned the handover (6, 0 of 256); B003 seed 2 did (88). Every run with an extra input has failed at the handover |

## Story so far
1. **B001.** The layout works; scripted grasps don't, so learning has to find the grasp.
2. **B002.** Learns to grasp and hand over, then both arms keep holding: letting go didn't pay enough.
3. **B003.** Pay more for letting go: the receiver now takes the bar, but carries it nowhere. The receiver's own camera almost never sees the bin while carrying (2%); the picker's camera does (40%), but using it means translating another arm's view. No memory.
4. **B004.** Tell it where the bin is (sim only), to test whether missing information is the stall. Neither seed let go, so it could not answer. Lesson: from scratch, whether the release is learned varies more between runs than the effect being tested. Next: start from B003's checkpoint, which already lets go.
5. **B005.** From B003's checkpoint: the bin input broke the handover (as in all B004 runs). The control carries but moves the bar only ~5 cm toward the bin (28 → 23 cm). The carry reward is almost flat at that distance (30 → 20 cm pays 0.14 of 16 points), a second suspect besides vision.
6. **B006.** Make the carry reward pull equally at every distance (0.25 per 10 cm): the bar gets only ~1.5 cm closer (21 cm), 0 in the bin. The reward is not the main stall; vision is the remaining suspect. Fixes: memory, a top camera, or a layout that keeps the bin in view.
7. **B007.** First-frame memory (after fixing a stale reset frame): neither seed learned the handover, so memory never reached the carry. B003 seed 2 reproduces B003. Pattern: every run given an extra input (5) failed at the handover; every run without (4) carried. Next: test the carry on its own.

## Rules
- Each experiment has a protocol written before it runs: question, predictions, method, decision rule. Changes after that are dated amendments. Results, including negative ones, go in the same file with the command and the commit.
- Every number comes from code (file:line), a measurement (date, data, raw output in `experiments/bimanual/out/`), or is marked [estimate].
- One change per experiment. Read percentages, not exact counts: GPU simulation is not reproducible run to run.
