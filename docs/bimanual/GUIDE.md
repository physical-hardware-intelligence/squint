# Guide: add and train a two-arm task

For people and coding agents. Read [README.md](README.md) first (what exists, what we learned), then this.

## 1. Map

| File | What it is | Touch it when |
|---|---|---|
| `envs/bimanual_base.py` | Two SO-101s side by side, one wrist camera each, domain randomization, sim settings (100 Hz physics, 10 Hz control, larger contact buffers) | Rarely. Shared by every two-arm task |
| `envs/handover.py` | Task 1: objects, layout, `evaluate()` flags, staged reward, `_get_obs_agent` (the 24-number state), env registration | **Your template**: copy its structure for a new task |
| `envs/__init__.py` | Imports each task module so `gym.make` finds it | Add one import per new task file |
| `train_squint.py` | Squint's trainer. Only change for two arms: `single_action_box` flattens the two-arm action space | Not for a new task |
| `utils.py` | Colour jitter, applied to each camera separately | No |
| `experiments/bimanual/check_reward.py` | Pre-flight: puts objects in chosen states, checks each reward stage and the success flag | Write one per task |
| `experiments/bimanual/stage_counts.py` | Runs a checkpoint on 128 episodes, counts how many reach each stage | Adapt the flags per task |
| `experiments/bimanual/who_sees_what.py` | Per phase: which camera sees which object (exact, from segmentation) | Adapt the objects and phases |
| `experiments/bimanual/handover_oracle.py` | B001's scripted inverse-kinematics controller | Reference for IK on the SO-101; its scripted grasps did not work |
| `experiments/bimanual/launch.sh` | Starts a training run with a log of commit, GPU, command and exit code | Use for every run |
| `experiments/bimanual/out/` | Raw outputs behind every number in the experiment files | Add yours |
| `docs/bimanual/experiments/` | One file per experiment: protocol first, results after | One per experiment |

## 2. Setup

- Install with the repository's [README](../../README.md) (conda, `environment.yaml`).
- **Training needs a CUDA GPU.** Ours: one RTX 4080 SUPER (16 GB). Task 1 at 512 envs × 2 cameras uses about 8.6 GB and runs about 1,050-1,140 steps/s, so 3M steps take about 50 min. Two runs at once do not fit in 16 GB.
- Headless Linux: if the renderer finds no GPU, set `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`.
- macOS (`source env.sh`, MoltenVK): CPU simulation only, and the two-arm CPU simulation allows **one env** (`num_envs=1`). Good for reward checks, not for training.

## 3. What the policy sees and does

| | Shape | Detail |
|---|---|---|
| Images | 2 × 128 × 128 RGB → 6 channels | left then right wrist camera; the trainer shrinks them to 16 × 16 |
| State | 24 numbers | per arm (left, then right): 6 noisy joint angles, 6 controller targets. Built in the task's `_get_obs_agent`. B004's diagnostic env adds 4 |
| Action | 12 numbers | per arm (left, then right): 5 joint deltas + gripper, `pd_joint_target_delta_pos` |
| Episode | 120 steps | 12 s at 10 Hz (`max_episode_steps` in `register_env`) |
| Memory | none | each step sees only the current images and state |

**One policy pixel is 0.9-1.8 cm** at 10-20 cm from the camera (71° field of view, 16 pixels across; computed). Anything smaller than that is not visible to the policy.

## 4. Recipe for a new task (in this order)

1. **Name the series.** One letter per task, so parallel work does not collide: Task 1 uses `B001…`; use e.g. `I001…` for insertion. Add a row to the README's task table.
2. **Write the env** (copy `handover.py`'s structure):
   - a dataclass of randomization ranges (sizes, friction, density, layout);
   - `_load_scene`: build objects once per env with sizes drawn from `self._batched_episode_rng`; `add_to_state_dict_registry` each one;
   - `_initialize_episode`: place objects and arms for the envs being reset;
   - `_get_obs_agent`: copy it from `handover.py`;
   - `evaluate()`: boolean flags only (grasped, aligned, inserted, success…); latch any milestone that later stages require;
   - `compute_dense_reward`: stages (section 5); `compute_normalized_dense_reward` divides by the maximum;
   - `@register_env("SO101<Task>-v1", max_episode_steps=…)`, then import the module in `envs/__init__.py`.
3. **Check feasibility before training.** Every object must be reachable by the arm that needs it: SO-101 reach is about 42-43.5 cm from the pan axis, which sits 3.88 cm in front of the base origin (30k sampled poses). Task 1's handover point (pan axis + 17 cm forward, centred, 12 cm up) is reachable by both arms.
4. **Pre-flight the reward** on constructed states (`check_reward.py` pattern): every stage in its band, bands rising in order, success at the maximum, penalties applied. CPU, minutes.
5. **Smoke test:** 64 envs, 50k steps (about 6 min). It catches shape and contact-buffer errors.
6. **Write the protocol before the full run** (template below). Changes after that are dated amendments.
7. **Run** with `launch.sh` inside tmux. `runs/NAME/ckpt.pt` is overwritten at every evaluation: copy it (`ckpt_1.5M.pt`) when you want to keep a step.
8. **Measure, don't eyeball:** stage counts at fixed steps (e.g. 1.5M, 3M), plus `who_sees_what.py` when a stage stalls. Save the JSON in `out/`.
9. **Record** results, predictions held or failed, and the decision in the same file; update the README's experiment row and story.

Protocol template:

```markdown
# I00N: <question in one line>
**Status:** protocol written <date>, before the run.
## Question
## Change (one, from <previous experiment>)
## Method        (steps, seed, envs, when to count)
## Predictions   (numbers, so they can fail)
## Decision rule (what each outcome leads to)
## Results
```

## 5. Lessons from Task 1 (read before designing a reward)

| Lesson | Evidence |
|---|---|
| Each stage must pay clearly more than the one before (≥ 1 point). If the next stage pays little and failing costs a lot, the policy parks at the safe stage | B002: letting go paid +0.4, 0 releases. B003: +1, then 39/128 carry ([B002](experiments/B002-handover-train.md), [B003](experiments/B003-release-payoff.md)) |
| Latch milestones and gate later stages on the latch, so a stage cannot be reached by a shortcut | `handed_over` in `handover.py` |
| Count stages from flags, not from reward bands: penalties move an episode into a lower band | B003 A1 |
| Check what each camera sees in each phase. The policy has no memory: if an object leaves both images, the policy cannot know where it is | B003 A2: during the carry, the bin is in the receiver's camera 2% of steps, the picker's 40% |
| Scripted grasps on the simulated jaws failed; the learned policy grasps fine. Do not plan a curriculum on scripted grasps without testing them | [B001](experiments/B001-handover-env-check.md) |
| GPU simulation is not reproducible: the same checkpoint and seed can differ by about 10 of 128 episodes. Count over 2+ seeds (256+ episodes) and read percentages | B003 3M: 39 then 52 carries on two counts ([B005](experiments/B005-warm-start-bin.md) A1) |
| One seed per condition cannot separate an effect from luck | [B004](experiments/B004-bin-in-state.md) run 1: 21 handovers vs B003's 48, before the change could matter |
| Check a new measurement tool against an exact source before trusting it | the geometric bin check agreed with segmentation on 98.2% of camera-steps |

## 6. Task 2 (insertion): decide these first

Reference: ALOHA's Bimanual Insertion (Zhao et al. 2023, ACT): left arm picks a socket, right arm a peg, insert in mid-air. ACT reached 32% (scripted demonstrations) and 20% (human demonstrations) in simulation. We start from RL with no demonstrations.

| Decision | Why it matters | How to settle it |
|---|---|---|
| Clearance | 5 mm is below one policy pixel (0.9-1.8 cm, section 3): the images show "peg near hole", not "aligned to 5 mm" | Start wider (e.g. 10-15 mm) or chamfer the hole; tighten in later experiments, one change each |
| Object sizes | Must fit the simulated jaws; B001 showed jaw contact is subtle | Measure the jaw opening in sim; confirm with a short training run that the arm grasps it |
| Layout | Each arm reaches its object; both reach the insertion point | Reuse Task 1's reach numbers (section 4, step 3) |
| Visibility at insertion | Each arm's camera may see only its own object (as Task 1 at the start) | Run `who_sees_what.py` on the first checkpoint that gets both grasps |
| Success flag | Must be exact and frame-independent | Measure depth along the socket's own axis (as Task 1 tests the bar in the bin's own frame) |

Suggested stages (a starting point, not a decision): both reach → both grasp → both lifted → peg tip near the hole (distance) → aligned (axis angle) → inserted (depth) → seated, still, success.
