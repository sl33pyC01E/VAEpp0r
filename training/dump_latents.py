#!/usr/bin/env python3
"""Dump pre-computed latents from a VAE (+ optional flattener) to disk.

Used to build a latent cache for `train_tokenizer.py --latent-cache DIR`
training mode, which lets the tokenizer train without any stem in memory.

Output: DIR/latent_NNNNNN.pt, one file per clip, containing the latent
tensor of shape (T', C_lat, H', W') — already stem-encoded.

Usage:
    python -m training.dump_latents \\
        --vae-ckpt PATH --out DIR --n-clips 2000 [--flatten-ckpt PATH]
        [--T 24] [--H 360] [--W 640] [--disco]
"""

import argparse
import os
import pathlib
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.generator import VAEpp0rGenerator
from core.tokenizer import StemChain
from training.train_tokenizer import _reconstruct_vae_from_ckpt


def _build_stem(args, device):
    """Load VAE (required) and optionally chain a flattener."""
    print(f"Loading VAE from {args.vae_ckpt} ...", flush=True)
    vae, _ = _reconstruct_vae_from_ckpt(args.vae_ckpt)
    vae = vae.to(device)
    if not args.flatten_ckpt:
        return vae

    from experiments.flatten import FlattenDeflatten
    print(f"Loading flattener from {args.flatten_ckpt} ...", flush=True)
    fckpt = torch.load(args.flatten_ckpt, map_location="cpu", weights_only=False)
    fcfg = fckpt.get("config", {})
    flattener = FlattenDeflatten(
        latent_channels=int(fcfg.get("latent_channels", vae.latent_channels)),
        bottleneck_channels=int(fcfg["bottleneck_channels"]),
        spatial_h=int(fcfg["spatial_h"]),
        spatial_w=int(fcfg["spatial_w"]),
        walk_order=fcfg.get("walk_order", "raster"),
        kernel_size=int(fcfg.get("kernel_size", 1)),
        deflatten_hidden=int(fcfg.get("deflatten_hidden", 0)),
    )
    flattener.load_state_dict(fckpt["model"], strict=False)
    flattener.eval().requires_grad_(False)
    flattener = flattener.to(device)
    return StemChain(vae, flattener)


def main():
    p = argparse.ArgumentParser(
        description="Dump stem-encoded latents from the procedural "
                    "generator for tokenizer training.")
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--flatten-ckpt", default="")
    p.add_argument("--out", required=True,
                   help="Output directory for latent_*.pt files")
    p.add_argument("--n-clips", type=int, default=2000,
                   help="Number of latent clips to dump")
    p.add_argument("--H", type=int, default=360)
    p.add_argument("--W", type=int, default=640)
    p.add_argument("--T", type=int, default=24)
    p.add_argument("--bank-size", type=int, default=5000)
    p.add_argument("--n-layers", type=int, default=128)
    p.add_argument("--pool-size", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4,
                   help="Clips encoded per pass (throughput only; output "
                        "is still one .pt per clip)")
    p.add_argument("--disco", action="store_true")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = _build_stem(args, device)
    print(f"Stem: latent_channels={stem.latent_channels}, "
          f"t_downscale={stem.t_downscale}, s_downscale={stem.s_downscale}",
          flush=True)

    # Generator (needs a bank + pool like the training script)
    gen = VAEpp0rGenerator(
        height=args.H, width=args.W, device=str(device),
        bank_size=args.bank_size, n_base_layers=args.n_layers,
    )
    bank_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bank")
    bank_files = [f for f in os.listdir(bank_dir)
                  if f.startswith("shapes_") and f.endswith(".pt")] \
        if os.path.isdir(bank_dir) else []
    if bank_files:
        gen.setup_dynamic_bank(bank_dir, working_size=args.bank_size)
        gen.build_base_layers()
    else:
        gen.build_banks()
    pool_kwargs = dict(
        use_fluid=True, use_ripple=True, use_shake=True, use_kaleido=True,
        fast_transform=True, use_flash=True, use_palette_cycle=True,
        use_text=True, use_signage=True, use_particles=True,
        use_raymarch=True, sphere_dip=True, use_arcade=True,
        use_glitch=True, use_chromatic=True, use_scanlines=True,
        use_fire=True, use_vortex=True, use_starfield=True, use_eq=True,
    )
    gen.build_motion_pool(
        n_clips=args.pool_size, T=args.T, random_mix=True, **pool_kwargs)
    if args.disco:
        gen.disco_quadrant = True

    idx = 0
    with torch.no_grad():
        while idx < args.n_clips:
            n = min(args.batch_size, args.n_clips - idx)
            clips = gen.generate_from_pool(n).to(device)
            z = stem.encode_video(clips)   # (N, T', C, H', W')
            for b in range(z.shape[0]):
                out_path = out_dir / f"latent_{idx:06d}.pt"
                # Save as dict so loaders can attach metadata later
                torch.save({"latent": z[b].detach().cpu(),
                            "T": args.T, "H": args.H, "W": args.W},
                           out_path)
                idx += 1
            print(f"  [{idx}/{args.n_clips}] latent shape={tuple(z.shape[1:])}",
                  flush=True)

    print(f"Done. Wrote {args.n_clips} latent files to {out_dir}")


if __name__ == "__main__":
    main()
