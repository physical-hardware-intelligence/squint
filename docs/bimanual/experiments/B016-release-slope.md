# B016: Does paying the picker's way off the bar (reward v4) raise the clean releases?

**Status:** protocol written 2026-10-26 02:45, before the runs.

## Question
In the chain, the carry policy places 90-92% of what it is handed; the handover policy hands it over cleanly in only about 40% of episodes. Of B011's 99 failed releases (256 episodes), 79 end with the receiver gripping and the picker's opened jaws still touching the bar; 5 froze holding, 14 had the receiver let go, 1 dropped it (`release_failures.py`, raw `experiments/bimanual/out/release_failures_b011_wrist_s1_3M.txt`). Switching to the carry policy as soon as the picker stops squeezing did not help (in the bin 93 and 105 vs 99 and 99): it never practised moving the picker off. With a reward that pays the picker for every centimetre it moves off, does the handover policy release cleanly more often?

## Why v2 stalls here
After the handover v2 pays 7 + 1 once the picker's jaws are off the bar + the carry's progress: a single step of +1 with nothing before it. A picker whose jaw rests on the bar gains nothing for moving 1 or 2 cm.

## Change: reward v4
v2 plus a slope on that +1: 0.5 for the jaws off, plus 0.5 × the picker's grip point's distance from its end of the bar, full at 6 cm (`PICKER_CLEAR_DISTANCE`). Only in the states after the handover where the receiver grips and the picker has stopped squeezing; everything else is v2. Probe (`reward_probe_release.py`): v2 7.71 / 7.71 / 7.71 / 8.71 / 8.71 / 8.71 along jaw on at 0, 1, 2 cm, jaw off at 2, 4, 6 cm; v4 7.71 / 7.79 / 7.88 / 8.38 / 8.54 / 8.71, never down. `check_reward.py` 18 of 18 on v1, v2, v3 and v4; `reward_probe_handover.py` unchanged. Env `SO101HandoverBarV4-v1`.

## Method
Continue B011 seed 1 from its 3M checkpoint with reward v4, 3M steps, everything else as B011 (γ 0.9, support −20 … 20, wrist cameras): `python train_squint.py --exp_name=b016_s<seed> --env_id=SO101HandoverBarV4-v1 --seed=<seed> --num_envs=512 --total_timesteps=3000000 --eval_freq=250000 --env_background_level 0 50 --env_background_tint 6 --gamma=0.9 --v_min=-20 --v_max=20 --checkpoint=runs/b011_wrist_s1/ckpt_3M.pt`. Training seeds 1 and 2 (seed 2 after B017). Control: B015, the same continuation with reward v2. Preflight first. At +1.5M and +3M: stage counts, `release_failures.py`, and the chain with B014's carry policies (seeds 7 + 8, 256 episodes).

## Predictions
1. Carries at +3M at least 140 of 256 on each training seed (B011: 101; B015 at +1.5M: 104).
2. Failed releases with the picker's jaw still on the bar at most 40 (B011: 79).
3. Chained, the bar in the bin in at least 120 of 256 for at least one carry policy (B014: 99).

## Decision rule
- 1 holds: keep v4 for the handover policy; then B017 (carry starts from it).
- 1 fails, 2 holds: the picker clears but something else breaks the carry: look at the videos.
- 1 and 2 fail: the slope is not the missing piece; the second policy has to practise the release itself (starts saved where the picker stops squeezing).

## Results
_Not run._
