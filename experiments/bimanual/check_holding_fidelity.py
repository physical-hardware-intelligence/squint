"""Are restored "holding" starts faithful? The same policy (B011 seed 1, wrist images) continues (a) its own episodes from
the moment the chain would switch (receiver gripping, picker not squeezing, 6 steps in a row) and (b) the saved starts
restored by the carry env. For each: still holding 1 and 11 steps later, and bars that jump more than 5 cm in one
step (a restore artifact shows up as a jump in (b) and not in (a)). Exits 1 unless the restored starts hold within 5
points of the own episodes at 1 and 11 steps and fewer than 3% jump (B018 amendment A1).

    SQUINT_CARRY_BANK=runs/carry_starts_holding_b011.pt python experiments/bimanual/check_holding_fidelity.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper  # noqa: E402

import envs  # noqa: E402,F401
from train_squint import DeployAgent  # noqa: E402

CKPT = "runs/b011_wrist_s1/ckpt_3M.pt"
N, EPISODES, SETTLE, JUMP = 128, 6, 5, 0.05
dev = torch.device("cuda")
holding = lambda i: (i["handed_over"] & i["receiver_grasp"] & ~i["picker_grasp"]).to(dev)


def make(env_id):
    env = gym.make(env_id, num_envs=N, obs_mode="rgb+segmentation", sim_backend="gpu", sensor_configs=dict(width=128, height=128),
                   domain_randomization=True, domain_randomization_config={"background_level_range": [0, 50], "background_tint": 6})
    return FlattenActionSpaceWrapper(FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True))


def wrist(obs):
    return {"rgb": obs["rgb"][..., :6], "state": obs["state"]}


def follow(env, agent, obs, active, steps=11):
    """From the current state, `steps` policy steps; per env: holding after 1 and after `steps`, and any bar jump."""
    u = env.unwrapped
    prev = u.bar.pose.p.clone()
    jump = torch.zeros(N, dtype=torch.bool, device=dev)
    for k in range(steps):
        obs, r, te, tr, info = env.step(agent.get_action(wrist(obs)))
        p = u.bar.pose.p
        jump |= torch.linalg.norm(p - prev, dim=1).to(dev) > JUMP
        prev = p.clone()
        if k == 0:
            after1 = holding(info)
    return after1 & active, holding(info) & active, jump & active


def report(name, n, h1, h11, jumps):
    print(f"{name}: {n} starts; holding 1 step later {h1} ({100 * h1 / max(n, 1):.0f}%), 11 steps later {h11} "
          f"({100 * h11 / max(n, 1):.0f}%); a bar jump over {JUMP * 100:.0f} cm in one step: {jumps}")
    return [h1 / max(n, 1), h11 / max(n, 1), jumps / max(n, 1)]


# (a) the policy's own episodes, from the switch moment (each env once per episode)
env = make("SO101HandoverBar-v1")
obs, _ = env.reset(seed=900)
agent = DeployAgent(env, sample_obs=wrist(obs), device=dev)
agent.load_checkpoint(CKPT)
tot = [0, 0, 0, 0]
for ep in range(EPISODES):
    obs, _ = env.reset(seed=900 + ep)
    run = torch.zeros(N, dtype=torch.long, device=dev)
    done = torch.zeros(N, dtype=torch.bool, device=dev)
    h1 = h11 = jumps = torch.zeros(N, dtype=torch.bool, device=dev)
    pending = torch.full((N,), -1, dtype=torch.long, device=dev)  # step at which each env hit the switch moment
    rec = []
    prev = env.unwrapped.bar.pose.p.clone()
    for t in range(120):
        obs, r, te, tr, info = env.step(agent.get_action(wrist(obs)))
        p = env.unwrapped.bar.pose.p
        rec.append((holding(info), torch.linalg.norm(p - prev, dim=1).to(dev) > JUMP))
        prev = p.clone()
        run = torch.where(holding(info), run + 1, torch.zeros_like(run))
        hit = (run == SETTLE + 1) & ~done
        pending[hit] = t
        done |= hit
    for e in torch.nonzero(done)[:, 0].tolist():
        t0 = int(pending[e])
        if t0 + 11 >= len(rec):
            continue  # too late in the episode to follow 11 steps
        tot[0] += 1
        tot[1] += int(rec[t0 + 1][0][e])
        tot[2] += int(rec[t0 + 11][0][e])
        tot[3] += int(any(bool(rec[t0 + k][1][e]) for k in range(1, 12)))
own = report("(a) own episodes", *tot)
env.close()

# (b) the saved starts, restored by the carry env, same policy on the wrist images
env = make("SO101HandoverCarryTopCamLeHome-v2")
tot = [0, 0, 0, 0]
for ep in range(EPISODES):
    obs, _ = env.reset(seed=900 + ep)
    if ep == 0:
        agent = DeployAgent(env, sample_obs=wrist(obs), device=dev)
        agent.load_checkpoint(CKPT)
    active = torch.tensor([len(r) > 0 for r in env.unwrapped._bank_rows], device=dev)
    a1, a11, j = follow(env, agent, obs, active)
    tot[0] += int(active.sum()); tot[1] += int(a1.sum()); tot[2] += int(a11.sum()); tot[3] += int(j.sum())
res = report("(b) restored starts", *tot)
# Gate (B018 amendment A1): restored within 5 points of the own episodes at 1 and 11 steps, bar jumps under 3%
ok = res[0] >= own[0] - 0.05 and res[1] >= own[1] - 0.05 and res[2] < 0.03
print(("PASS" if ok else "FAIL") + "  restored starts within 5 points of the own episodes, bar jumps under 3%")
sys.exit(0 if ok else 1)
