"""Video of an insertion checkpoint: runs N episodes once, keeps every frame from a close camera on the arms' working
area, then writes the K episodes that spend the most steps in success as a 2 x 2 grid, each tile labelled with time,
depth and state, plus the single best episode on its own.

    python experiments/bimanual/insert_showcase.py --checkpoint CKPT --out_dir DIR [--env_id SO101InsertPegRetain-v1]
One pass on the GPU simulator: the episodes shown are the ones scored (a second pass is not reproducible there).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mani_skill.sensors.camera import CameraConfig  # noqa: E402
from mani_skill.utils import sapien_utils  # noqa: E402
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper  # noqa: E402

import envs  # noqa: E402,F401
from envs.insertion import Insertion  # noqa: E402
from train_squint import DeployAgent  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--env_id", default="SO101InsertPegRetain-v1")
ap.add_argument("--num_envs", type=int, default=64)
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--out_dir", required=True)
args = ap.parse_args()


def closeup(self):
    # WHY: the default render camera sits 0.6 m out, where the 22 mm peg is a few pixels wide. This one frames both
    # grippers' working area from about 0.33 m
    pose = sapien_utils.look_at(eye=[0.46, 0.0, 0.27], target=[0.18, 0.0, 0.09])
    return CameraConfig("render_camera", pose=pose, width=640, height=480, fov=np.deg2rad(62), near=0.01, far=100)


Insertion._default_human_render_camera_configs = property(closeup)
dev = torch.device("cuda")
env = gym.make(args.env_id, num_envs=args.num_envs, obs_mode="rgb+segmentation", sim_backend="gpu",
               sensor_configs=dict(width=128, height=128), domain_randomization=True, render_mode="rgb_array",
               domain_randomization_config={"background_level_range": [0, 50], "background_tint": 6})
env = FlattenActionSpaceWrapper(FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True))
obs, _ = env.reset(seed=args.seed)
agent = DeployAgent(env, sample_obs=obs, device=dev)
agent.load_checkpoint(args.checkpoint)
retain = env.unwrapped.RETAIN
T = gym.spec(args.env_id).max_episode_steps
keys = ("success", "axial", "both_lifted", "in_hole", "seated", "released", "retreat", "socket_pitch")
I = {k: [] for k in keys}
frames = []
with torch.no_grad():
    for t in range(T):
        obs, _, _, _, info = env.step(agent.get_action(obs))
        for k in keys:
            I[k].append(info[k].cpu())
        img = env.render()
        img = (img.cpu().numpy() if torch.is_tensor(img) else np.asarray(img)).astype(np.uint8)
        # WHY: JPEG in memory (~50 kB a frame), so 64 episodes x 150 steps fit (~0.5 GB, vs ~9 GB raw at 640 x 480)
        frames.append([cv2.imencode(".jpg", im[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])[1] for im in img])
I = {k: torch.stack(v) for k, v in I.items()}  # (T, n)
steps_in_success = I["success"].float().sum(0)
pick = [int(i) for i in torch.argsort(steps_in_success, descending=True)[: args.k]]
print("episodes with success", int(I["success"].any(0).sum()), "of", args.num_envs, "| at the last step", int(I["success"][-1].sum()))
print("shown", pick, "steps in success", [int(steps_in_success[i]) for i in pick])


def label(img, i, t):
    img = img.copy()
    s, d = bool(I["success"][t, i]), float(I["axial"][t, i]) * 1000
    if s and retain:
        state, col = f"LET GO, arm {100 * float(I['retreat'][t, i]):.0f} cm away, PEG STAYS IN ({d:.0f} mm)", (40, 200, 60)
    elif s:
        state, col = f"INSERTED {d:.0f} mm, both holding: SUCCESS", (40, 200, 60)
    elif retain and bool(I["seated"][t, i]) and bool(I["released"][t, i]):
        state, col = f"let go, peg seated {d:.0f} mm", (240, 200, 40)
    elif bool(I["in_hole"][t, i]):
        state, col = f"in the hole {d:.0f} mm, mouth {float(I['socket_pitch'][t, i]):+.0f} deg", (240, 200, 40)
    elif bool(I["both_lifted"][t, i]):
        state, col = f"both up, tip {max(-d, 0):.0f} mm from the mouth", (240, 200, 40)
    else:
        state, col = "grasp / lift", (230, 230, 230)
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, f"ep {i} t={t / 10:4.1f}s", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, state, (170, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2 if s else 1, cv2.LINE_AA)
    return img


def frame(t, i):
    return cv2.imdecode(frames[t][i], cv2.IMREAD_COLOR)[..., ::-1]


out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
grid = []
for t in range(T):
    tiles = [label(frame(t, i), i, t) for i in pick]
    tiles += [np.zeros_like(tiles[0])] * (4 - len(tiles))
    grid.append(np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:4], 1)], 0))
imageio.mimsave(out / "showcase_grid.mp4", grid, fps=10, macro_block_size=1)
imageio.mimsave(out / "showcase_best.mp4", [label(frame(t, pick[0]), pick[0], t) for t in range(T)], fps=10, macro_block_size=1)
print("wrote", out / "showcase_grid.mp4", out / "showcase_best.mp4")
