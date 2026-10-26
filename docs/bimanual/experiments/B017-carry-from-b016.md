# B017: Carry policy retrained on B016's handovers: does the chain keep the gain?

**Status:** not a valid test. B016 seed 1 made almost no clean releases, so the harvest saved 10 carry starts (502 of 512 env indices had none) and the carry policies trained on normal starts instead.

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
Ran 2026-10-26 04:00-04:55 as queued (raw: `experiments/bimanual/out/overnight_1026_counts.txt`): harvest 10 starts; preflight 8 of 8 (its restore checks ran on the few envs that had a start); both carry seeds count 2 carries of 256 from "saved starts" (mostly normal starts) and 0 in the bin chained. No conclusion about the recipe; B014 already showed it works when the handover policy releases.
