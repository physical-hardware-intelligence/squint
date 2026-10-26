# B017: Carry policy retrained on B016's handovers: does the chain keep the gain?

**Status:** protocol written 2026-10-26 02:45, before the runs. Runs whatever B016 seed 1 gives.

## Question
B014 showed the carry policy must practise from the handover policy it follows (B013: 52-71% of carries placed; B014: 90-92%). B016 changes the handover policy. Retrained on starts saved from B016 seed 1's handovers, does the chain place at least as large a share of its carries, so the gain in carries becomes bars in the bin?

## Method
B014's recipe with B016 seed 1 (+3M) as the handover policy: `harvest_carry_starts.py --checkpoints runs/b016_s1/ckpt_3M.pt --episodes 15 --num_envs 512 --out runs/carry_starts_b016.pt`; preflight; carry policy seeds 1 and 2 (`SO101HandoverCarryTopCamLeHome-v2`, 1M steps, B014's command); per seed, stage counts from the saved starts and the chain (B016 seed 1, then this policy; seeds 7 + 8).

## Predictions
1. Chained, at least 85% of the episodes that switch end with the bar in the bin, on each seed.
2. In the bin, end to end: at least B016's chained count with B014's carry, on each seed.

## Decision rule
- 1 and 2 hold: this pair (B016 + B017) is the chain for the real rig; next, the same recipe with the calibrated top camera.
- 1 fails: compare with the starts' own rate; if the starts' rate holds and the chained rate does not, the switch condition is wrong for B016's releases.

## Results
_Not run._
