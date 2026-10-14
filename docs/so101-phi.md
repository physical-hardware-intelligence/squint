# Squint on an SO-101: the Physical Hardware Intelligence fork

This branch (`phi/so101`) runs [Squint](https://github.com/aalmuzairee/squint) on a real SO-101 follower arm, deployed from a MacBook (Apple M4, CPU only), trained on one RTX 4080 SUPER. It keeps upstream's code and adds the fixes, safety guards and tools we needed. Every change is one commit with its reason in the message: `git log master..phi/so101`.

## What is different from upstream

| commit | what | why | upstream |
|---|---|---|---|
| `fix(deploy): load the checkpoint on the device in use` | `DeployAgent(..., device=device)` | checkpoints are saved from GPU memory; without a device, `torch.load` refuses them on a CPU-only machine | [squint#7 comment](https://github.com/aalmuzairee/squint/issues/7) |
| `fix(deploy): work around ManiSkill Sim2RealEnv recursion` | `deploy_utils/sim2real_compat.py` | with `mani_skill_nightly` 2026.6.23.446 and later, `Sim2RealEnv.reset()` raises `RecursionError` | [ManiSkill#1498](https://github.com/mani-skill/ManiSkill/issues/1498), [squint#7](https://github.com/aalmuzairee/squint/issues/7) |
| `fix(so101): apply the gripper conversion under lerobot 0.4.3` | also match `"so_follower"` | with the pinned lerobot 0.4.3 the gripper was never switched to degrees and never remapped, silently | [squint#7 comment](https://github.com/aalmuzairee/squint/issues/7) |
| `feat(robot): rig settings via environment, safer connect and motion` | `deploy_utils/robot_config.py` | no machine paths in git; goal = present before torque-on; bus retries; torque-on rollback; goal lead limit | |
| `feat(safety): crash guard and an exit pose off the wrist-roll seam` | `deploy_utils/safety.py` | a crash between connect and deploy's exit handler left the arm standing with torque on; exit drove the wrist to the encoder seam | |
| `feat(deploy): flight recorder` | `--record_dir` also saves every policy input | to see exactly what the policy saw, step by step | |
| `config(phi): wrist camera pose tuned for our SO-101 mount` | `envs/base_random_env.py` | **rig-specific**: our camera mount differs from Squint's | |
| `fix(deploy): take each episode's first frame when it starts` | `Sim2RealEnv.observe()` + one line in `deploy.py` | `reset()` captured the first frame right after the arm returned, before the next cube was placed, so step 0 saw the previous cube position | |
| `feat(tools): ...` | `deploy_utils/*.py` | see Tools below | |

## Setup on macOS (Apple Silicon)

The versions we ran:

```bash
git clone https://github.com/physical-hardware-intelligence/squint && cd squint && git switch phi/so101
python3.11 -m venv .venv
.venv/bin/pip install torch==2.6.0 torchvision==0.21.0 tensordict==0.7.2 torchrl==0.7.2 \
    mani_skill_nightly==2026.8.2.400 gymnasium tyro tqdm wandb opencv-python==4.12.0.88 matplotlib "lerobot[feetech]==0.4.3"
curl -L -o mvk.tar https://github.com/KhronosGroup/MoltenVK/releases/download/v1.4.2/MoltenVK-macos.tar && tar -xf mvk.tar && rm mvk.tar
cp rig.example.sh rig.local.sh   # set your port, camera, camera rotation, calibration file (git-ignored)
source env.sh                    # every session, from the repo root
```

- macOS has no Vulkan driver; MoltenVK translates to Metal (`env.sh` points `VK_ICD_FILENAMES` at it). Deployment needs no GPU: `Sim2RealEnv` runs on the CPU physics backend and the policy takes under 1 ms per step on the CPU.
- Create the venv on a filesystem with symlinks (not exFAT).
- macOS grants camera access per app: run deploy from a terminal that has Camera permission.
- Retraining after a camera change happens on a CUDA machine with the same `train_squint.py`.

## Workflow, safest first

1. **Fake-arm dry run** (no hardware): `MPLBACKEND=Agg python deploy_utils/deploy_dryrun.py --checkpoint runs/<run>/ckpt.pt`. It runs `deploy.py`'s real `main()` against a perfect fake servo, with Squint's sim as the camera, and prints every command the arm would have received.
2. **Camera.** Lock exposure and white balance outside LeRobot (its `OpenCVCameraConfig` has no exposure setting; we used `uvcc`). `deploy_utils/camera_match.py --sweep` picks them by comparing the real image with the sim at the same pose; do it at the task's start pose (while `deploy.py` holds the arm there: macOS lets two processes read one camera), not with the arm folded, where the sim hides the jaws below the table. Our camera came back in auto exposure after every power cycle: `camera_match.py --apply <lock.json>` at the start of each session, `--check` before deploying. Set `SQUINT_CAMERA_ROTATION` if the camera is mounted rotated. Measure the camera pose with `deploy_utils/tune_camera_nomotion.py`, paste the printed values into `envs/base_random_env.py`, and **retrain**: the pose is a training-time input, so pasting alone changes nothing for an existing checkpoint.
3. **Hold test:** `python deploy_utils/hold_test.py`. Torque on, every joint must stay within 1 degree for 3 s, torque off.
4. **Deploy, one step per Enter:**

   ```bash
   python deploy_utils/hold_test.py && python deploy.py --checkpoint runs/<run>/ckpt.pt --env_id SO101ReachCube-v1 \
       --no-continuous_eval --control_freq 15 --debug --max_episode_steps 60 --record_dir runs/<run>/real
   ```

   On start the arm moves to the task's start pose by itself (about 2.5 s). `q` or Ctrl-C folds the arm and releases torque.
5. **Analyse:** `python deploy_utils/analyze_flight.py runs/<run>/real/flight --checkpoint runs/<run>/ckpt.pt`.

## Safety notes

| risk | what we saw | guard |
|---|---|---|
| jump at torque-on | `Goal_Position` is a RAM register; after power-up it read 0 on all six servos while the joints sat 799-3095 ticks away. LeRobot's `connect()` enables torque without setting it | goal = present position right before torque-on |
| half-powered arm | LeRobot enables torque servo by servo; one lost reply aborted a connect after servo 1 was on | bus retries (2) + roll every servo back to off on failure |
| arm left standing | building `Sim2RealEnv` drives the arm to the start pose before `deploy.py` installs its exit handler; a crash there left the arm holding torque | crash guard: fold, then release; if the fold fails, keep holding and say so |
| wrist at the encoder seam | the rest pose's wrist roll is -180 deg, raw tick 0 on a full-circle wrist calibration | exit pose uses -90 deg |
| pushing against an obstacle | the policy's targets are never clipped to joint limits; the gripper target keeps closing past the shut jaws | each goal at most 8 deg (arm) / 5 deg (gripper) ahead of the present position |
| the tuning tool moves the arm | upstream `tune_camera.py` drives the arm at start-up, on `r`/`s`, and on quit | `tune_camera_nomotion.py`: the sim follows the arm instead |

`deploy.py`'s "Achieved freq" printout leaves out the `--debug` drawing time; from command timestamps on the fake arm, `--control_freq 15` gives 14.7 Hz.

## Results so far (SO101ReachCube, our camera)

- **What success means:** gripper tip within 2 cm of a point 2 cm above the cube, arm still, gripper at least 90% closed (`envs/reach.py`). Closing the gripper over the cube is part of the task.
- **Sim:** 1.00 success at training settings; 5/5 at our deploy settings (action x0.15, 60 steps).
- **Real, cube in front of the arm:** in both episodes the arm went above the cube and closed the gripper (not yet scored against the 2 cm criterion).
- **Real, cube about 10 cm from the base and 10 cm to the left:** failed. That spot is outside the area the cube spawns in during training, and the same checkpoint fails there in sim too.
- **Where training puts the cube:** x 20-40 cm, y ±10 cm from the sim's origin, the URDF base origin, which sits 3.9 cm behind the pan axis. Measured on the table from the pan axis: **x 16-36 cm forward, y ±10 cm**. At the start pose the wrist camera sees the table only up to about 25 cm, so for most of that square the cube is not in the first frames.
- **Real, cube at ~(21.5, +4) cm from the pan axis, 4 episodes, continuous 15 Hz:** very repeatable, closest approach 3.4-3.7 cm from the goal point (success needs 2 cm), then the gripper closed. Unscored: the cube position (camera) and the tip (kinematics) are model estimates, not yet checked with a ruler.

## Known limits

- **Cube placement.** Keep the cube inside the training spawn area above, or widen the area and retrain.
- **Background.** Training images have a pure black background everywhere. Our black cloth reads about 33/255 on the camera, and on real frames the policy's actions are far more sensitive to the background level than in sim (mean action change 0.9 vs 0.1 when the background is set to black). The wall behind the robot can also come into view at some angles, as the Squint author notes in [squint#2](https://github.com/aalmuzairee/squint/issues/2).
- Every real run so far was paced by Enter (`--no-continuous_eval`), not yet continuous at 15 Hz.

## Tools (`deploy_utils/`)

| script | use |
|---|---|
| `tune_camera_nomotion.py` | camera pose tuning with the arm untouched; `--selftest` proves it |
| `hold_test.py` | torque-on check before any run |
| `deploy_dryrun.py` | `deploy.py` on a fake arm, open or closed loop |
| `flight_recorder.py` | used by `deploy.py --record_dir` |
| `analyze_flight.py` | what the policy saw, and how the background changes its actions |
| `sim_rollout_check.py` | when the policy reaches, stops and closes, in sim |
| `background_position_test.py` | background colour and cube position, in sim; estimates a recording's cube position |
| `measure_camera_fit.py` | how well a camera pose matches the real view at the policy's 16x16 |
| `check_deploy_camera.py` | deploy's camera path vs a direct capture; camera settings before/after |
| `camera_match.py` | real vs sim at the same pose, region by region (robot, cube, background); sweeps, applies and checks exposure, gain and white balance through `uvcc` |
| `score_reach.py` | Squint's success test on a recorded real episode; cube from a ruler CSV (from the pan axis) or the camera |

## License

MIT, as upstream ([LICENSE](../LICENSE)).
