"""Restore fidelity by source: each harvest policy continues restored carry starts for 11 steps; the share still held.

    SQUINT_CARRY_BANK=runs/carry_starts.pt python experiments/bimanual/check_carry_fidelity.py s1|s2 SPLIT_ROW
Rows below SPLIT_ROW came from the first checkpoint passed to the harvest (B003 seed 1), the rest from the second.
A faithful restore lets each policy keep its own carries about as often as in its original episodes (98-100%).
"""
import sys; sys.path.insert(0, ".")
import gymnasium as gym, torch, envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from train_squint import DeployAgent
K = int(sys.argv[2]); N = 128; dev = torch.device("cuda")
ck = {"s1": "runs/handover_b003_2026-10-24/ckpt_3M.pt", "s2": "runs/handover_b003s2_2026-10-25/ckpt_3M.pt"}[sys.argv[1]]
env = FlattenActionSpaceWrapper(FlattenRGBDObservationWrapper(gym.make("SO101HandoverCarry-v1", num_envs=N, obs_mode="rgb+segmentation",
      sim_backend="gpu", sensor_configs=dict(width=128, height=128), domain_randomization=True,
      domain_randomization_config={"background_level_range": [0, 50], "background_tint": 6}), rgb=True, depth=False, state=True))
u = env.unwrapped
carry_of = lambda info: (info["handed_over"] & info["receiver_grasp"] & ~info["picker_touch_bar"]).to(dev)
tally = {"from s1": [0, 0], "from s2": [0, 0]}
for ep in range(6):
    obs, _ = env.reset(seed=900 + ep)
    if ep == 0:
        agent = DeployAgent(env, sample_obs=obs, device=dev); agent.load_checkpoint(ck)
    src = u._restored_rows < K
    for _ in range(11):
        obs, r, te, tr, info = env.step(agent.get_action(obs))
    c = carry_of(info)
    for name, m in (("from s1", src), ("from s2", ~src)):
        tally[name][0] += int(m.sum()); tally[name][1] += int((m & c).sum())
for name, (n, k) in tally.items():
    print(f"policy {sys.argv[1]}, starts {name}: held 11 steps later {k} of {n} ({100*k/max(n,1):.0f}%)")
