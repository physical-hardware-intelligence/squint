"""Write a warm-start checkpoint that keeps a checkpoint's encoder, actor and alpha but replaces its critic with a freshly
initialised one (train_squint's own Critic class). For a warm start under a changed reward: the old critic's values
were learned for the old reward and steer the actor wrong until they are relearned.

    python experiments/bimanual/fresh_critic.py --checkpoint CKPT --out OUT --env_id ENV_ID [--num_atoms 101] [--num_q 2]
--num_atoms, --num_q, --v_min, --v_max must match the training run that will load OUT. Checks that every critic entry
has the same key and shape as the old one before saving.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper  # noqa: E402

import envs  # noqa: E402,F401
from train_squint import Critic, DeployAgent  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--env_id", required=True)
ap.add_argument("--num_atoms", type=int, default=101)
ap.add_argument("--num_q", type=int, default=2)
ap.add_argument("--v_min", type=float, default=-25)
ap.add_argument("--v_max", type=float, default=35)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
torch.manual_seed(args.seed)
dev = torch.device("cuda")
env = gym.make(args.env_id, num_envs=1, obs_mode="rgb+segmentation", sim_backend="gpu", sensor_configs=dict(width=128, height=128))
env = FlattenActionSpaceWrapper(FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True))
obs, _ = env.reset(seed=0)
agent = DeployAgent(env, sample_obs=obs, device=dev)
critic = Critic(n_obs=agent.encoder.repr_dim, n_state=int(np.prod(obs["state"].shape[1:])),
                n_act=int(env.action_space.shape[-1]), num_atoms=args.num_atoms,  # (12,) or (1, 12)
                v_min=args.v_min, v_max=args.v_max, num_q=args.num_q, device=dev)
ckpt = torch.load(args.checkpoint, map_location=dev)
old, new = ckpt["critic"], critic.state_dict()
assert set(old) == set(new), sorted(set(old) ^ set(new))
# WHY tensors only: the critic's state dict also holds non-tensor entries (torch.Size metadata), which have no .shape
size = lambda v: tuple(v.shape) if torch.is_tensor(v) else v  # noqa: E731
bad = [k for k in old if size(old[k]) != size(new[k])]
assert not bad, [(k, size(old[k]), size(new[k])) for k in bad]
ckpt["critic"] = new
torch.save(ckpt, args.out)
print(f"fresh critic: {len(new)} entries, keys and shapes match; encoder, actor, alpha kept; "
      f"global_step {ckpt.get('global_step')}; wrote {args.out}")
