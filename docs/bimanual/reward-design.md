# Reward design for the handover (v1 → v6)

The history: what each reward version changed and the measurement behind it. For the current reward explained from zero, read [reward-explained.md](reward-explained.md) first.

Why task 1 stalls after the handover, measured, and what reward v2 changes. Numbers come from `experiments/bimanual/reward_probe.py` (the reward scored on hand-set states) and `check_reward.py`, 2026-10-25.

## 1. How the reward is built

Points per step (0-16), by stage:

| Stage | Situation | Points |
|---|---|---|
| 1 | picker reaches the bar | 0-2 |
| 2 | picker holds it, brings it to the handover point | 2-4 |
| 3 | bar at the handover point, receiver reaches the free end | 4-6 |
| 4 | both hold the lifted bar (picker paid for opening) | 6-7 |
| 5 | handover done, receiver carries | 8-10 (v2: 7-10) |
| 6 | bar in the bin | 10-15 |
| | success | 16 |
| | penalties: table −6, bin −3, arms touching −2, bar not yet lifted −1 | |

The code checks the stages in order and the last one whose condition holds wins. A state that matches no later stage falls back to stage 1's formula (the picker's distance to the bar, at most 2).

## 2. v1's dips

Each stage's condition reads contact sensors at that instant: is a gripper squeezing the bar (≥ 0.5 N on both jaws), is it touching it (≥ 0.01 N on either). The moments between two stages can satisfy neither condition.

| State on the intended path | v1 | v2 |
|---|---|---|
| both hold | 6.53 | 6.53 |
| **picker opening, a jaw still touching** | **0.52** | 7.71 |
| picker clear, receiver carries | 8.68 | 8.71 |
| above wall height, just outside the rim | 9.48 | 9.40 |
| **just inside the rim** | **9.34** | 9.54 |
| over the bin centre | 9.36 | 9.60 |
| **released over the bin, falling** | **0.03** | 9.60 |
| in the bin, still holding | 13.55 | 13.55 |
| **in the bin, released, gripper brushing the wall** | **11.55** | 14.55 |
| success | 16.00 | 16.00 |

- **Release dip:** stage 4 needs the picker squeezing, stage 5 needs it not touching. While its jaws open, the squeeze ends before the touch does. (In a real release the picker is at the bar, so stage 1 gives up to about 2, not 0.52.)
- **Rim dip:** v1's height target jumps from "above the walls" to "the floor" when the bar crosses the bin's radius, so moving over the bin at a safe height loses points.
- **Drop dip:** stage 5 needs the receiver holding, stage 6 needs the bar in the bin. A bar let go above the bin satisfies neither while it falls.
- **Final release:** releasing earns +1, but brushing the bin wall costs −3.

## 3. Why a dip stops learning

SAC improves by trying small changes and keeping what its critic predicts pays more. Crossing a dip costs a few steps of low reward before the higher plateau. The critic sees the plateau only if it looks far enough ahead; otherwise "start letting go" looks like a loss and is unlearned. B002 froze holding with both arms (release dip). B003 paid stage 5 more, crossed it, and froze holding the bar (drop and final-release dips).

## 4. How far the critic looks: γ

A reward k steps ahead counts γ^k:

| steps ahead (0.1 s each) | 10 | 20 | 30 |
|---|---|---|---|
| γ = 0.9 (B001-B007) | 0.35 | 0.12 | 0.04 |
| γ = 0.97 | 0.74 | 0.54 | 0.40 |

γ = 0.9 looks about 10 steps (1 s) ahead, enough for PlaceCube. The carry takes 20-30 steps, so the bin's payoff counts about 0.12 or less.

## 5. Information was not the main blocker

Relative to the receiver, the bin is always 20-28 cm forward and 6-12 cm outward (300 layouts). A bar stood upright, blind, at the patch's centre lands inside the bin in 186 of 300 layouts (62%). A receiver that never sees the bin could still score most of the time with one fixed motion; ours does not move (28.5 → 23 cm). Seeing the bin raises 62% toward 100%; it is not what stops the carry. (Why every extra input broke the handover in B004, B005 and B007 is still unexplained.)

## 6. Reward v2

Rule: on the intended path every state scores at least the one before. Milestones are latched (`handed_over`), and after the handover the score follows the bar's progress, not contact flags. Stages 1-4 and the penalties are v1's, except as listed.

| Change | v2 rule | Effect |
|---|---|---|
| release | after the handover, receiver holding and picker not squeezing: 7 + 1 once the picker is off the bar + carry progress | both hold ≤ 7 ≤ opening; clear ≥ 8 |
| carry progress (0-2) | 1.2 × horizontal pull (straight line, 0 at 40 cm, 1 over the bin centre) + 0.4 × height | 0.30 per 10 cm at any distance (v1 at 30 cm: 0.14) |
| height | outside the bin: keep above the walls (0-1); over it: 1 + how far lowered toward the floor (1-2) | equal on both sides of the rim at wall height: no rim dip |
| drop into the bin | a bar let go over the bin's footprint counts as carried | falling ≈ 9.6, then in the bin ≥ 12 |
| final release | no bin-touch penalty once the bar is in the bin; success still needs everything let go | releasing beats holding |
| horizon | γ = 0.97, critic support −25 … 35 (per-step reward in [−0.75, 1]) | the bin's payoff visible from the carry's start |

Code: `Handover._reward_after_handover_v2` (`envs/handover.py`), `REWARD_VERSION = 2`; full task `SO101HandoverBarV2-v1`. v1 is unchanged for comparison. `check_reward.py`: 18 of 18 on both.

## 7. Reward v3: the handover step (B012)

B010/B011 found the handover itself stalls at γ 0.97: closing the grip paid +0.16 over hovering (the picker's jaws are 11.5% open while it holds the bar, measured), a jaw-to-jaw touch cost −2, and a nudge past 5 cm dropped the score by 2.6. v3 = v2 plus three changes before and during the handover (`reward_probe_handover.py`):

| Step | v2 | v3 |
|---|---|---|
| receiver 20 cm away | 4.48 | 4.36 |
| receiver at the bar, hovering | 5.95 | 5.46 |
| receiver grips: both hold | 6.11 | 6.56 |
| both hold, jaws touch | 4.11 | 6.56 |
| both hold, wrists or forearms touch | 4.11 | 4.56 |
| hovering, bar nudged 7 cm | 3.33 | 5.46 |
| picker letting go, jaws touch | 5.71 | 7.71 |
| picker clear | 8.71 | 8.71 |

- Grip: stage 3 tops out at 5.5 (4 + 1.5 × reach), stage 4 starts at 6.5 (+ 0.5 × picker opening).
- No 5 cm cliff: once the bar has reached the handover point in the picker's grip (latched), stage 3 holds within 10 cm.
- Jaw contact: no penalty for gripper-to-gripper contact while the picker is still on the bar at the handover point; wrist and forearm contact always −2.

Code: `REWARD_VERSION = 3`, env `SO101HandoverBarV3-v1`. v1 and v2 are unchanged.

## 8. Reward v4: the picker's way off the bar (B016)
B011's failed releases mostly end with the picker's jaws opened but one still on the bar (79 of 99). v2 pays +1 only once the jaws are off: nothing for the centimetres before. v4 = v2 with that +1 split: 0.5 for the jaws off, 0.5 × the picker's grip point's distance from its end of the bar, full at 6 cm. Only after the handover, with the receiver gripping and the picker not squeezing; everything else is v2.

| Picker after letting go | v2 | v4 |
|---|---|---|
| jaw on the bar, 0 / 1 / 2 cm out | 7.71 / 7.71 / 7.71 | 7.71 / 7.79 / 7.88 |
| jaw off, 2 / 4 / 6 cm out | 8.71 / 8.71 / 8.71 | 8.38 / 8.54 / 8.71 |

Code: `REWARD_VERSION = 4`, `PICKER_CLEAR_DISTANCE`, env `SO101HandoverBarV4-v1`; probe `reward_probe_release.py`; `check_reward.py` 18 of 18 on v1-v4.

**Result (B016): v4 failed.** Continued from B011 seed 1 on five seeds, the clean releases collapsed (carries 101 → 3-12). B011's picker stops a few centimetres from the bar; v2 paid that release the full +1, v4 about +0.6. Changing a trained policy's reward must be checked on the states the policy actually visits, not only along a constructed path.

## 9. The grasp check (B018, B019)
Not a reward version, but it changes what every version pays. `is_grasping` (0.5 N on each jaw, the push up to 110 degrees off the opening direction) also accepts a fully closed gripper pressed against the bar's side. B018: about half of B011's counted handovers were such fake grips (108 real of 193 counted). Since B019 the envs ending in `Grip` also require the jaws to be stopped by the bar, at least 3% open (`GRASP_NEEDS_OPEN_JAWS`, `MIN_GRASP_OPENNESS`). Real grips sit at 10-11%.

## 10. Reward v5: the receiver comes in open (B020)
B019 A2 (`probe_receiver_grasp.py`): in 106 of 256 episodes the receiver arrived with its gripper already shut and pressed it on the bar's side. A shut gripper resting on the face is at least 1.25 cm from the centre line, which v2's reach paid only 0.125 of 2 less than jaws around the bar. v5 = v2 with stage 3's reach multiplied by 0.5 + 0.5 × min(receiver openness / 0.15, 1). 0.15 was set on seed 9: real approaches run at a median 22% open and stay above 15% until the grip in 95% of steps. `reward_probe_v5.py` on B011's own episodes: real handovers lose 0.0 per reach step, shut-beside episodes 0.83, no dip at the grip.

**Result (B020):** the shut approach went from 106 to 0 on every seed measured, but the receiver then hovered open beside the bar. Gripping paid 6.1 against the best hover's 6.0, and under v5 a missed close costs.

## 11. Reward v6: gripping at the handover point (B021)
v6 = v5 with stage 4 (both hold) = 6 + 0.5 × pull(bar → handover point) + 0.5 × picker openness, where v2 and v5 paid 6 + picker openness. A grip at the point beats the best hover by 0.61, not 0.1. The extra fades with distance (25 cm off: 0.02 over v2), so it does not repeat v3's flat pay anywhere (§7). The release still never pays less as it goes. `reward_probe_v6.py` checks these on constructed states.

| State | v2 | v5 | v6 |
|---|---|---|---|
| receiver at the bar, open (hover) / shut | 5.95 / 5.95 | 5.95 / 4.98 | 5.95 / 4.98 |
| both hold, bar 0 / 7 / 12 / 25 cm from the handover point | 6.11 | 6.11 | 6.56 / 6.39 / 6.29 / 6.13 |
| both hold, picker 50% / 100% open | 6.50 / 7.00 | 6.50 / 7.00 | 6.75 / 7.00 |

Trained from scratch, not continued from B011: continuing changed the reward under a trained policy and lost the release in most runs (B019, B020).

## 12. Known, not changed yet

- **Arms touching at the handover:** handled in v3 (§7).
- **The receiver gets no signal before stage 3:** it wanders (the "visit the bin first" motion seen in the videos). A small term for waiting near the handover point would fix it; left out so that v2 changes only what happens after the handover.

B008 tests v2 and γ on the carry alone ([B008](experiments/B008-carry-alone.md)).
