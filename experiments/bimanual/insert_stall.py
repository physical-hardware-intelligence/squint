"""Why does an inserted peg stop short? Runs a checkpoint on N episodes and, at each episode's deepest step with both
parts up and the face in the hole, reads what could hold the peg back, grouped by that depth.

    python experiments/bimanual/insert_stall.py --checkpoint CKPT [--env_id SO101InsertPegRetain-v1] [--num_envs 256]
Columns (p10/50/90): tilt off the socket axis; lateral offset of the tip from the axis; roll width (the face's extent
along the worse socket axis over the peg's width: 1 square to the hole, 1.41 at 45 deg roll); fit (> 0 = face edge
past the wall line, in-hole allows 1 mm); peg-socket contact force and its share along the socket axis (near 1 = the
peg pushes on a face, near 0 = on the side walls); over the 10 steps before the deepest step, how far the peg arm's TCP
moved inward relative to the mouth, how far the peg did, and how far the TCP slid along the peg (grip slip).
A wedge shows as axial share near 1 with the TCP moving in and the peg not; slip as TCP sliding along the peg; a
policy that stops pushing as TCP and peg both still.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper  # noqa: E402

import envs  # noqa: E402,F401
from train_squint import DeployAgent  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--env_id", default="SO101InsertPegRetain-v1")
ap.add_argument("--num_envs", type=int, default=256)
ap.add_argument("--seed", type=int, default=8)
args = ap.parse_args()
env = gym.make(args.env_id, num_envs=args.num_envs, obs_mode="rgb+segmentation", sim_backend="gpu",
               sensor_configs=dict(width=128, height=128), domain_randomization=True,
               domain_randomization_config={"background_level_range": [0, 50], "background_tint": 6})
env = FlattenActionSpaceWrapper(FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True))
obs, _ = env.reset(seed=args.seed)
u = env.unwrapped
agent = DeployAgent(env, sample_obs=obs, device=torch.device("cuda"))
agent.load_checkpoint(args.checkpoint)
T = gym.spec(args.env_id).max_episode_steps
R = {k: [] for k in ("axial", "ok", "align", "lat", "roll", "fit", "force", "axial_share", "tcp_in", "tcp_along")}
with torch.no_grad():
    for t in range(T):
        obs, _, _, _, info = env.step(agent.get_action(obs))
        g = u.geometry()
        s_mat, p_mat = u.socket.pose.to_transformation_matrix(), u.peg.pose.to_transformation_matrix()
        roll = torch.stack([(p_mat[..., :3, 1] * s_mat[..., :3, j]).sum(-1).abs() + (p_mat[..., :3, 2] * s_mat[..., :3, j]).sum(-1).abs()
                            for j in (1, 2)], -1).max(-1).values
        F = u.scene.get_pairwise_contact_forces(u.peg, u.socket)
        f = torch.linalg.norm(F, axis=-1)
        tcp = u.arms[1].tcp_pos
        for k, v in dict(axial=g["axial"], ok=info["both_lifted"] & info["in_hole"], align=g["align"], lat=g["lateral"],
                         roll=roll, fit=g["fit"], force=f, axial_share=(F * g["socket_axis"]).sum(-1).abs() / f.clamp(min=1e-6),
                         # WHY: relative to the mouth, so the socket arm's own motion cancels
                         tcp_in=-((tcp - g["mouth"]) * g["socket_axis"]).sum(-1),
                         tcp_along=((tcp - u.peg.pose.p) * g["peg_axis"]).sum(-1)).items():
            R[k].append(v.cpu())
R = {k: torch.stack(v) for k, v in R.items()}  # (T, n)
n = args.num_envs
depth = torch.where(R["ok"], R["axial"], torch.full_like(R["axial"], -1.0))
t_max, d_max = depth.argmax(0), depth.max(0).values
idx, t0 = torch.arange(n), (t_max - 10).clamp(min=0)
at = lambda k: R[k][t_max, idx]  # noqa: E731
moved = lambda k: R[k][t_max, idx] - R[k][t0, idx]  # noqa: E731
cols = dict(tilt_deg=torch.rad2deg(torch.acos(at("align").clamp(-1, 1))), lateral_mm=at("lat") * 1000, roll_width=at("roll"),
            fit_mm=at("fit") * 1000, contact_N=at("force"), axial_share=at("axial_share"), tcp_in_mm=moved("tcp_in") * 1000,
            peg_in_mm=moved("axial") * 1000, tcp_slid_mm=moved("tcp_along").abs() * 1000)
print("max depth mm | episodes | " + " | ".join(cols) + "   (p10/50/90)")
for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 45), (45, 60)):
    k = (d_max > 0) & (d_max * 1000 >= lo) & (d_max * 1000 < hi)
    if not k.any():
        print(f"{lo:>2}-{hi:<2} | 0")
        continue
    q = lambda x: "/".join(f"{float(v):.2f}" for v in x[k].float().quantile(torch.tensor([0.1, 0.5, 0.9])))  # noqa: E731
    print(f"{lo:>2}-{hi:<2} | {int(k.sum()):3d} | " + " | ".join(q(v) for v in cols.values()))
