# Bimanual Squint

One Squint policy drives two SO-101 arms from their two wrist cameras: trained in simulation with RL, no demonstrations, to run on the real pair.

**New here, or adding a task?** Read this page, then [GUIDE.md](GUIDE.md) (files, setup, recipe, lessons) and [reward-design.md](reward-design.md) (why the reward is shaped as it is).

| | |
|---|---|
| Rig | two SO-101 followers side by side, facing forward, pan axes 46 cm apart; one wrist camera each |
| Policy input | left then right wrist image (16×16, 6 channels); 24 numbers (per arm: 6 noisy joint angles, 6 targets); no memory |
| Policy output | 12 joint deltas, `pd_joint_target_delta_pos` |
| Code | `envs/bimanual_base.py` (two arms, two cameras), `envs/handover.py` (task 1) |

## Tasks

| # | Task | Env | Status |
|---|---|---|---|
| 1 | Handover: one arm picks a 25 × 25 × 60 mm red bar, hands it to the other in mid-air, which puts it in the bin. Either arm may pick; neither can reach both bar and bin | `SO101HandoverBar-v1` | the whole task reaches the bin by chaining two trained policies (B013: up to 28% in the bin); not yet as one policy (B010-B012) |
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
| [B008](experiments/B008-carry-alone.md) | Trained on the carry alone: do reward dips, horizon or seeing the bin decide it? | done: reward v2 is the fix (in bin 87% vs v1 17%); top camera 90% (= sim-only bin position); wrist-only 39%; γ no consistent effect; memory hurts |
| [B009](experiments/B009-lehome-top-camera.md) | Does the top camera placed as on Ilia's rig work as well on the carry? | done: yes, in bin 94% (90 cm camera: 88%); clean finishes similar (35% at the end) |
| [B010](experiments/B010-full-task-v2.md) | Does the whole task learn with reward v2, from wrist cameras or with a top camera? | done: the handover stalls at γ 0.97 (3M: wrist 29 and 53, top 14 of 256), so the carry is barely reached (2 of 768) |
| [B011](experiments/B011-full-task-gamma-090.md) | Is γ 0.97 what stalls the handover in the whole task? | partly: seed 1 learned the handover (193; γ 0.97: 29) and released (101 carries), but the carry stalls ~31 cm from the bin; seed 2 did not hand over (13) |
| [B012](experiments/B012-v3-carry-practice.md) | Does the whole task reach the bin with reward v3 and carry practice? | stopped after A seed 1: 8 handovers of 256 (v3's grip paid away from the handover point too); B runs not done |
| [B013](experiments/B013-chain-skills.md) | Chain the learned handover and carry policies: does the bar reach the bin? | done: yes with the top-camera carry, 71 of 256 in the bin (success at the end up to 38); wrist-only 0 and 21 |
| [B014](experiments/B014-carry-from-own-handovers.md) | Trained on the handover policy's own handovers, does the carry policy hold its rate when chained? | done: yes, 90-92% of carries in the bin; 99 of 256 end to end on both seeds |
| [B015](experiments/B015-handover-longer.md) | Does more training lift the handover policy's carries, and the chain with them? | running |

## Story so far
1. **B001.** The layout works; scripted grasps don't, so learning has to find the grasp.
2. **B002.** Learns to grasp and hand over, then both arms keep holding: letting go didn't pay enough.
3. **B003.** Pay more for letting go: the receiver now takes the bar, but carries it nowhere. The receiver's own camera almost never sees the bin while carrying (2%); the picker's camera does (40%), but using it means translating another arm's view. No memory.
4. **B004.** Tell it where the bin is (sim only), to test whether missing information is the stall. Neither seed let go, so it could not answer. Lesson: from scratch, whether the release is learned varies more between runs than the effect being tested. Next: start from B003's checkpoint, which already lets go.
5. **B005.** From B003's checkpoint: the bin input broke the handover (as in all B004 runs). The control carries but moves the bar only ~5 cm toward the bin (28 → 23 cm). The carry reward is almost flat at that distance (30 → 20 cm pays 0.14 of 16 points), a second suspect besides vision.
6. **B006.** Make the carry reward pull equally at every distance (0.25 per 10 cm): the bar gets only ~1.5 cm closer (21 cm), 0 in the bin. The reward is not the main stall; vision is the remaining suspect. Fixes: memory, a top camera, or a layout that keeps the bin in view.
7. **B007.** First-frame memory (after fixing a stale reset frame): neither seed learned the handover, so memory never reached the carry. B003 seed 2 reproduces B003. Pattern: every run given an extra input (5) failed at the handover; every run without (4) carried. Next: test the carry on its own.
8. **Reward review** ([reward-design](reward-design.md)): v1 has measured dips after the handover (release, rim, drop, final let-go), and γ 0.9 sees about 1 s ahead. A blind upright placement lands in the bin in 62% of layouts, so seeing the bin is not the main blocker. Reward v2 removes the dips.
9. **B008.** The carry trained alone, from saved starts: with v2 the bar goes in the bin (87% with the bin position given; v1: 17%). A top camera gets 90%, wrist cameras alone 39%. γ: no consistent effect on getting it in, but 0.97 holds the finish better (success at the end 46% vs 27%). First successes of the project.
10. **B009.** The top camera placed as on Ilia's rig works on the carry alone: in the bin 94%.
11. **B010.** The whole task from scratch, v2 at γ 0.97: the handover stalls (29 and 53 of 256 at 3M; B003: 128 and 88), with either camera, so the carry is barely reached.
12. **B011.** Only γ changed, to 0.9: on seed 1 the handover is back (193) and the release (101 carries), but the carries stall ~31 cm from the bin. Seed 2 never grips (13): from scratch, the grip is found on some seeds only. The handover needs 0.9, the carry's finish wants 0.97 (B008). B012 removes the conflict: a handover step that clearly pays (reward v3, [reward-design](reward-design.md) §7) at 0.97, plus 25% of training episodes starting mid-carry.
13. **B012.** v3 plus carry practice at γ 0.97: 8 handovers. v3 paid gripping anywhere a full point over waiting at the handover point, a shortcut the reward probe never scored. Stopped after one seed.
14. **B013.** Stop asking one run to learn everything: chain the policies that work. B011's handover, then a carry policy from B008/B009. With the top camera, 71 of 256 bars in the bin, the first from the whole task. Limits: 40% of episodes reach a carry, and the carry policies lose ground on another policy's handovers. Next: carry starts from the handover policy itself, the calibrated top camera, the real rig.
15. **B014.** Carry starts saved from B011's own handovers: the retrained carry policy puts 90-92% of what it is handed in the bin; 99 of 256 end to end (39%). The handover policy is now the limit: 42% of episodes reach a carry.

## Rules
- Each experiment has a protocol written before it runs: question, predictions, method, decision rule. Changes after that are dated amendments. Results, including negative ones, go in the same file with the command and the commit.
- Every number comes from code (file:line), a measurement (date, data, raw output in `experiments/bimanual/out/`), or is marked [estimate].
- One change per experiment. Read percentages, not exact counts: GPU simulation is not reproducible run to run.
