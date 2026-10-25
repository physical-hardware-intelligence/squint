"""Give a trained checkpoint extra state inputs without changing what it does (B005's warm start).

    python experiments/bimanual/widen_state_input.py --src runs/<run>/ckpt_3M.pt --dst <out>.pt --extra 4
Adds `extra` input columns to the first state layer (`proj.state_proj.0.weight`) of the actor and the critic, filled
with zeros: the new inputs contribute nothing until training changes those weights, so the widened policy acts exactly
as the original. Everything else (encoder, other layers, entropy coefficient) is copied unchanged.
"""

import argparse

import torch

KEY = "proj.state_proj.0.weight"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--extra", type=int, required=True)
    args = ap.parse_args()
    ckpt = torch.load(args.src, map_location="cpu")
    for part in ("actor", "critic"):
        w = ckpt[part][KEY]
        ckpt[part][KEY] = torch.cat([w, torch.zeros(w.shape[0], args.extra, dtype=w.dtype)], dim=1)
        print(f"{part} {KEY}: {tuple(w.shape)} -> {tuple(ckpt[part][KEY].shape)}")
    ckpt["widened_from"] = args.src
    torch.save(ckpt, args.dst)


if __name__ == "__main__":
    main()
