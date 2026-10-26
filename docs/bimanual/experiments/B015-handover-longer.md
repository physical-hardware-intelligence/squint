# B015: Does more training lift the handover policy's carries, and does the chain follow?

**Status:** done 2026-10-26 02:27. No: 3M more steps with reward v2 left the carries flat, then lower (101 → 104 at +1.5M → 91 at +3M).

## Question
In the chain (B014) the carry policy puts 90-92% of carried bars in the bin; the limit is the handover policy: only about 42% of episodes reach a carry. B011 seed 1's carries were still rising fast (1 at 1.5M, 101 at 3M). Trained 3M steps more, does it reach more carries, and does the chained in-bin count rise with them?

## Change
Continue B011 seed 1 from its 3M checkpoint, everything else as B011: `python train_squint.py --exp_name=b015_s1 --env_id=SO101HandoverBarV2-v1 --seed=1 --num_envs=512 --total_timesteps=3000000 --eval_freq=250000 --env_background_level 0 50 --env_background_tint 6 --gamma=0.9 --v_min=-20 --v_max=20 --checkpoint=runs/b011_wrist_s1/ckpt_3M.pt`. Loading restores the encoder, actor, critic (the target critic is copied from it) and the entropy weight; the optimizers and the replay buffer start empty, and the first 5,000 steps act randomly (`learning_starts`), as any run.

## Method
Preflight first. At +1.5M and +3M: stage counts (seeds 7 + 8, 256 episodes) and the chain with B014's carry policies, seeds 1 and 2 (`chain_eval.py`, env `SO101HandoverBarV2TopCamLeHome-v2`).

## Predictions
1. Carries at +3M at least 130 of 256 (B011 at 3M: 101).
2. Chained, the bar in the bin in at least 110 of 256 for each carry seed (B014: 99 and 99).
3. The carry policy keeps at least 80% of the carries it is handed. Below that, the handover policy's handovers changed away from the ones the carry starts were saved from (B013's problem again).

## Decision rule
- 1 and 2 hold: longer training is the lever for the handover; continue it, and harvest new carry starts when 3 starts to slip.
- 1 fails: the handover's release has plateaued; look at the failed releases in video before changing the reward.
- 1 holds, 3 fails: harvest carry starts from the new policy and retrain the carry (B014's recipe, 35 min).

## Results
Run 2026-10-26 01:25-02:27, commit `4797a3b`, preflight 8 of 8 (the checkpoint loaded at step 3,000,320). Raw: `experiments/bimanual/out/b015_counts.txt`, `stages_b015_*.json`, `chain_b015_*.json`.

| 256 episodes (seeds 7 + 8) | B011 at 3M | B015 at +1.5M | B015 at +3M |
|---|---|---|---|
| handovers | 193 | 199 | 166 |
| **carries** | 101 | 104 | **91** |
| chained, in the bin (carry s1 / s2) | 99 / 99 | 96 / 97 | 89 / 70 |
| of the switched, in the bin (s1 / s2) | 92% / 90% | 97% / 92% | 94% / 86% |

**Predictions.** 1 fails (91, not 130). 2 fails (89 and 70, not 110). 3 holds (86-97%).

**Decision (rule "1 fails").** The release has plateaued under v2. The failed releases were looked at first, in numbers rather than video (`release_failures.py`): 79 of 99 end with the picker's opened jaw still on the bar, and v2 pays nothing until it is off. Next: [B016](B016-release-slope.md), a slope for that last step (reward v4).
