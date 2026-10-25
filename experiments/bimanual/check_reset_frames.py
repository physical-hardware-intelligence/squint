"""Are the frames returned at reset what the cameras see right then? And does the first-frame memory hold the right view?

    python experiments/bimanual/check_reset_frames.py SO101HandoverBarFirstFrame-v1   (or any two-arm env id)
Compares the observation returned by reset (seeded, and automatic at an episode's end) with a fresh render of the same
moment. Before the 2026-10-25 fix, reset returned frames rendered before the wrist cameras moved to the new arm pose:
this check fails on that code (3 of 9) and passes after (9 of 9). GPU simulation.
"""
import sys; sys.path.insert(0, ".")
import gymnasium as gym, torch, envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
EID = sys.argv[1]; N = 8; mem = "FirstFrame" in EID
base = gym.make(EID, num_envs=N, obs_mode="rgb+segmentation", sim_backend="gpu", domain_randomization=True,
                domain_randomization_config={"background_level_range": [0, 50], "background_tint": 6})
flat = FlattenRGBDObservationWrapper(base, rgb=True, depth=False, state=True)
env = ManiSkillVectorEnv(FlattenActionSpaceWrapper(flat), N, ignore_terminations=True, record_metrics=True)
u = env.unwrapped
fresh = lambda: flat.observation(u.get_obs())["rgb"]
ok = []
def check(name, cond): ok.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name)
for seed in (1, 2):  # a second reset: the cameras start from the previous episode's pose
    obs, _ = env.reset(seed=seed)
    check(f"reset(seed={seed}): returned frame = fresh render (cameras synced)", torch.equal(obs["rgb"], fresh()))
    if mem:
        check(f"reset(seed={seed}): first frame = current frame", torch.equal(obs["rgb"][..., 6:], obs["rgb"][..., :6]))
first0 = obs["rgb"][..., 6:].clone() if mem else None
a = torch.zeros((N, 12), device=obs["rgb"].device)
for t in range(1, 125):
    a[:] = torch.randn_like(a) * 0.5
    obs, r, term, trunc, info = env.step(a)
    if mem and t == 60:
        check("step 60: first frame unchanged", torch.equal(obs["rgb"][..., 6:], first0))
    if trunc.any():
        check("auto-reset: returned frame = fresh render", torch.equal(obs["rgb"], fresh()))
        if mem:
            fo = info["final_observation"]["rgb"]
            check("auto-reset: final observation keeps the old first frame", torch.equal(fo[..., 6:], first0))
            check("auto-reset: new first frame = new current frame", torch.equal(obs["rgb"][..., 6:], obs["rgb"][..., :6]))
            check("auto-reset: new first frame differs from the old", not torch.equal(obs["rgb"][..., 6:], first0))
        break
print(EID, f"{sum(ok)} of {len(ok)} passed")
