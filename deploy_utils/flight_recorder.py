"""
deploy.py flight recorder (on with --record_dir). Every policy step: the exact 128x128 image the policy received (PNG,
lossless) and one CSV row with the 12-number state, the raw action, the scaled action sent, and the time.

Written step by step (no buffering), so a crash or Ctrl-C loses nothing. Costs ~1 ms per step.
Layout: <dir>/ep000/step000.png ... + <dir>/ep000/log.csv
"""

import csv
import time
from pathlib import Path

import cv2
import numpy as np

STATE = [f"qpos_{j}" for j in ("pan", "lift", "elbow", "flex", "roll", "grip")] + \
        [f"target_{j}" for j in ("pan", "lift", "elbow", "flex", "roll", "grip")]


class FlightRecorder:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.episode, self.step, self._f, self._w = -1, 0, None, None

    def start_episode(self):
        self.close()
        self.episode += 1
        self.step = 0
        d = self.root / f"ep{self.episode:03d}"
        d.mkdir(parents=True, exist_ok=True)
        self._dir = d
        self._f = open(d / "log.csv", "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(["step", "t"] + STATE + [f"action_{i}" for i in range(6)] + [f"sent_{i}" for i in range(6)])
        self._f.flush()

    def log(self, obs, action, sent):
        """obs: the dict the policy received (rgb (1,128,128,3) uint8, state (1,12) radians); action/sent: (1,6)."""
        rgb = obs["rgb"][0].cpu().numpy() if hasattr(obs["rgb"], "cpu") else np.asarray(obs["rgb"][0])
        cv2.imwrite(str(self._dir / f"step{self.step:03d}.png"), cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))
        state = obs["state"][0].cpu().numpy() if hasattr(obs["state"], "cpu") else np.asarray(obs["state"][0])
        self._w.writerow([self.step, f"{time.time():.4f}"] + [f"{v:.6f}" for v in state]
                         + [f"{v:.6f}" for v in np.asarray(action).reshape(-1)]
                         + [f"{v:.6f}" for v in np.asarray(sent).reshape(-1)])
        self._f.flush()
        self.step += 1

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None
