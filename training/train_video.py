#!/usr/bin/env python3
"""VAEpp0r Stage 2 — temporal training on animated synthetic data.

Enables VAE temporal compression (TPool/TGrow/MemBlock).
Generates T-frame clips with smooth motion, trains with temporal consistency loss.

Usage:
    python -m training.train_video
    python -m training.train_video --resume synthyper_logs/latest.pt --fresh-opt
"""

import argparse
import os
import pathlib
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import MiniVAE
from core.generator import VAEpp0rGenerator
from core.discriminator import PatchDiscriminator, hinge_d_loss, hinge_g_loss


# -- Haar wavelet helpers ------------------------------------------------------

def haar_down(x):
    """2x spatial downscale: (B, C, H, W) -> (B, 4C, H/2, W/2). Lossless."""
    a = x[:, :, 0::2, 0::2]
    b = x[:, :, 0::2, 1::2]
    c = x[:, :, 1::2, 0::2]
    d = x[:, :, 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a - b + c - d) * 0.5
    hl = (a + b - c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return torch.cat([ll, lh, hl, hh], dim=1)

def haar_up(x):
    """2x spatial upscale: (B, 4C, H, W) -> (B, C, H*2, W*2). Lossless inverse."""
    C = x.shape[1] // 4
    ll, lh, hl, hh = x[:, 0*C:1*C], x[:, 1*C:2*C], x[:, 2*C:3*C], x[:, 3*C:4*C]
    a = (ll + lh + hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    B, Ch, H, W = a.shape
    out = torch.zeros(B, Ch, H * 2, W * 2, device=x.device, dtype=x.dtype)
    out[:, :, 0::2, 0::2] = a
    out[:, :, 0::2, 1::2] = b
    out[:, :, 1::2, 0::2] = c
    out[:, :, 1::2, 1::2] = d
    return out

def haar_down_n(x, n):
    for _ in range(n):
        x = haar_down(x)
    return x

def haar_up_n(x, n):
    for _ in range(n):
        x = haar_up(x)
    return x


# -- Chunked inference (no-grad, for preview only) -----------------------------

_CHUNK_SIZE = 24

@torch.no_grad()
def _chunked_vae_inference(model, x, amp_dtype=torch.bfloat16):
    T = x.shape[1]
    trim = getattr(model, 'frames_to_trim', 0)
    if T <= _CHUNK_SIZE:
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            recon, latent = model(x)
        return recon, latent
    all_recon = []
    all_latent = []
    output_per_chunk = _CHUNK_SIZE - trim
    target_len = T - trim
    chunk_start = 0
    collected = 0
    while chunk_start < T and collected < target_len:
        chunk_end = min(chunk_start + _CHUNK_SIZE, T)
        chunk = x[:, chunk_start:chunk_end]
        if chunk.shape[1] < model.t_downscale:
            break
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            rc, lat = model(chunk)
        need = target_len - collected
        keep = min(rc.shape[1], need)
        all_recon.append(rc[:, :keep].float().cpu())
        all_latent.append(lat.float().cpu())
        collected += keep
        del rc, lat
        torch.cuda.empty_cache()
        if chunk_end >= T or collected >= target_len:
            break
        chunk_start += output_per_chunk
    return torch.cat(all_recon, dim=1), torch.cat(all_latent, dim=1)


# -- Preview -------------------------------------------------------------------

def _probe_fps(path):
    """Return video fps as float, default 30 on failure."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-of", "csv=p=0", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        s = r.stdout.decode().strip()
        num, den = s.split("/")
        return float(num) / float(den)
    except Exception:
        return 30.0


def _decode_video_frames(path, frame_skip, n_frames, W, H):
    """Decode n_frames from path starting at frame_skip. Returns list of HxWx3 uint8 arrays."""
    fps = _probe_fps(path)
    seek_s = frame_skip / fps
    result = subprocess.run(
        ["ffmpeg", "-v", "error",
         "-i", path,
         "-ss", str(seek_s),
         "-vf", f"scale={W}:{H}",
         "-vframes", str(n_frames),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    err = result.stderr.decode(errors="replace").strip()
    if err:
        print(f"  ffmpeg decode: {err}", flush=True)
    raw = result.stdout
    frame_bytes = H * W * 3
    return [
        np.frombuffer(raw[i*frame_bytes:(i+1)*frame_bytes], dtype=np.uint8).reshape(H, W, 3)
        for i in range(len(raw) // frame_bytes)
    ]


@torch.no_grad()
def save_preview(model, gen, logdir, step, device, amp_dtype, T=8, preview_T=None,
                  preview_image=None, preview_frame_skip=0, haar_rounds=0):
    """Save preview mp4 matching static train layout:
    reference (GT|Recon) scaled to synth grid width on top, synth clips grid below.
    preview_T: frames to use for preview clips (defaults to T if None).
    """
    try:
        model.eval()
        from PIL import Image as _PIL
        H, W = gen.H, gen.W
        cell_w = W * 2 + 4
        sep_v  = np.full((H, 4, 3), 14, dtype=np.uint8)
        trim   = getattr(model, 'frames_to_trim', 0)
        pT     = preview_T if preview_T else T

        # -- Synthetic: 2 clips side by side --
        # Same path-matching logic as train_video3d: sample from the live
        # recipe pool when T matches, else fall back to fresh render with
        # the training pool's kwargs. Guards against the "B=2 share one
        # effect roll" bug (both clips identical) and the "preview sees
        # disco but training sees pool" mismatch.
        with torch.random.fork_rng():
            torch.manual_seed(step + int(time.time()) % 100000)
            has_pool = (hasattr(gen, "_recipe_pool")
                        and gen._recipe_pool
                        and getattr(gen, "_motion_pool_T", None) == pT)
            if has_pool:
                import random as _rnd2
                N = len(gen._recipe_pool)
                idxs = _rnd2.sample(range(N), k=min(2, N))
                if len(idxs) < 2:
                    idxs = idxs + [idxs[0]]
                clips = torch.stack([
                    gen._render_recipe(gen._recipe_pool[i]) for i in idxs
                ], dim=0)
                if step < 5:
                    print(f"  [preview] pool path, idxs={idxs} of {N} recipes")
            else:
                # Fallback: render one clip at a time, re-rolling random_mix
                # per clip so effect stacks actually differ.
                kw_raw = getattr(gen, "_train_pool_kwargs", {}) or {}
                rmix = bool(getattr(gen, "_train_random_mix", True))
                def _roll(kw):
                    if not rmix:
                        return kw
                    import random as _rnd
                    drops = getattr(gen, "_RANDOM_MIX_DROPS", {}) or {}
                    out = dict(kw)
                    for k, p in drops.items():
                        if out.get(k) and _rnd.random() < p:
                            out[k] = False
                    return out
                parts = []
                for _ in range(2):
                    parts.append(gen.generate_sequence(1, T=pT, **_roll(kw_raw)))
                clips = torch.cat(parts, dim=0)
                if step < 5:
                    print(f"  [preview] fallback path (pool T="
                          f"{getattr(gen, '_motion_pool_T', None)} vs pT={pT})")
        clips = clips.to(device)
        if haar_rounds > 0:
            x = torch.stack([haar_down_n(clips[:, t], haar_rounds)
                              for t in range(pT)], dim=1)
        else:
            x = clips
        recon, _ = _chunked_vae_inference(model, x, amp_dtype=amp_dtype)
        T_out  = recon.shape[1]
        hH = H // (2 ** haar_rounds)
        hW = W // (2 ** haar_rounds)
        if haar_rounds > 0:
            syn_gt_raw = clips[:, trim:trim + T_out]   # (2, T_out, 3, H, W) — original RGB
            syn_rc_raw = torch.stack([haar_up_n(recon[:, t, :, :hH, :hW], haar_rounds)
                                       for t in range(T_out)], dim=1).clamp(0, 1)
            syn_gt = syn_gt_raw[:, :, :3, :H, :W].float().cpu().numpy()
            syn_rc = syn_rc_raw[:, :, :3, :H, :W].float().cpu().numpy()
        else:
            syn_gt = clips[:, trim:trim + T_out, :3, :H, :W].float().cpu().numpy()
            syn_rc = recon[:, :, :3, :H, :W].clamp(0, 1).float().cpu().numpy()
        T_show = T_out
        del recon, x, clips

        syn_w = cell_w * 2 + 2
        gap_v = np.full((H, 2, 3), 14, dtype=np.uint8)

        # -- Reference clip --
        ref_gt = None
        ref_rc = None
        if preview_image and os.path.exists(preview_image):
            ref_frames = _decode_video_frames(preview_image, preview_frame_skip or 0, pT, W, H)
            if len(ref_frames) < 2:
                print(f"  ref: only {len(ref_frames)} frames decoded", flush=True)
            else:
                ref_arr = np.stack(ref_frames).astype(np.float32) / 255.0
                ref_rgb = torch.from_numpy(ref_arr).permute(0, 3, 1, 2).to(device)  # (T, 3, H, W)
                if haar_rounds > 0:
                    ref_t = torch.stack([haar_down_n(ref_rgb[t].unsqueeze(0), haar_rounds).squeeze(0)
                                         for t in range(len(ref_frames))], dim=0).unsqueeze(0)
                else:
                    ref_t = ref_rgb.unsqueeze(0)  # (1, T, 3, H, W)
                ref_recon, _ = _chunked_vae_inference(model, ref_t, amp_dtype=amp_dtype)
                T_r_out = ref_recon.shape[1]
                if haar_rounds > 0:
                    ref_gt = ref_rgb[trim:trim + T_r_out, :3, :H, :W].float().cpu().numpy()
                    ref_rc_t = torch.stack([haar_up_n(ref_recon[0, t, :, :hH, :hW].unsqueeze(0), haar_rounds)
                                             for t in range(T_r_out)], dim=0)
                    ref_rc = ref_rc_t[:, 0, :3, :H, :W].clamp(0, 1).float().cpu().numpy()
                else:
                    ref_gt = ref_t[0, trim:trim + T_r_out, :3, :H, :W].float().cpu().numpy()
                    ref_rc = ref_recon[0, :, :3, :H, :W].clamp(0, 1).numpy()
                del ref_t, ref_recon
                T_show = min(T_show, T_r_out)

        model.train()

        # -- Compute frame dimensions: scale ref to syn_w (same logic as static train) --
        has_ref    = ref_gt is not None
        scale      = syn_w / cell_w
        ref_h      = int(H * scale)
        frame_w    = syn_w
        frame_h    = (ref_h + 6 + H) if has_ref else H

        stepped = os.path.join(logdir, f"preview_{step:06d}.mp4")
        latest  = os.path.join(logdir, "preview_latest.mp4")
        for out_path in [stepped, latest]:
            cmd = ["ffmpeg", "-y", "-v", "quiet",
                   "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", f"{frame_w}x{frame_h}", "-r", "30",
                   "-i", "pipe:0",
                   "-c:v", "libx264", "-crf", "18",
                   "-pix_fmt", "yuv420p", out_path]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            try:
                for t in range(T_show):
                    rows = []
                    if has_ref:
                        g = (ref_gt[t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                        r = (ref_rc[t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                        ref_cell = np.concatenate([g, sep_v, r], axis=1)  # (H, cell_w, 3)
                        ref_scaled = np.array(_PIL.fromarray(ref_cell).resize(
                            (syn_w, ref_h), _PIL.BILINEAR))
                        rows.append(ref_scaled)
                        rows.append(np.full((6, syn_w, 3), 14, dtype=np.uint8))
                    g0 = (syn_gt[0, t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                    r0 = (syn_rc[0, t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                    g1 = (syn_gt[1, t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                    r1 = (syn_rc[1, t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                    synth_row = np.concatenate([
                        g0, sep_v, r0, gap_v, g1, sep_v, r1
                    ], axis=1)
                    rows.append(synth_row)
                    proc.stdin.write(np.concatenate(rows, axis=0).tobytes())
                proc.stdin.close()
                proc.wait()
            except Exception:
                try: proc.stdin.close()
                except Exception: pass
                proc.kill()
                proc.wait()
                raise

        ref_note = " +ref" if has_ref else ""
        print(f"  preview: {stepped} ({T_show} frames{ref_note})", flush=True)

    except Exception as e:
        import traceback
        print(f"  preview failed: {e}", flush=True)
        traceback.print_exc()
        model.train()


# -- Training ------------------------------------------------------------------

_stop_requested = False

def _handle_stop(sig, frame):
    global _stop_requested
    _stop_requested = True
    print("\n[Stop requested]", flush=True)

signal.signal(signal.SIGTERM, _handle_stop)
signal.signal(signal.SIGINT, _handle_stop)
if sys.platform == "win32":
    signal.signal(signal.SIGBREAK, _handle_stop)


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    logdir = pathlib.Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    # -- Model (3ch RGB, temporal ENABLED) --
    # Read encoder/decoder/latent config from checkpoint if resuming
    enc_ch_str = args.enc_ch
    dec_ch = tuple(int(x) for x in args.dec_ch.split(","))
    latent_ch = args.latent_ch
    enc_time_str = args.enc_time
    dec_time_str = args.dec_time
    haar_mode = getattr(args, 'haar', 'none')
    _resume_cfg = {}
    if args.resume:
        _ckpt_peek = torch.load(args.resume, map_location="cpu", weights_only=False)
        _resume_cfg = _ckpt_peek.get("config", {})
        _cfg = _resume_cfg
        enc_ch_str = _cfg.get("encoder_channels", enc_ch_str)
        latent_ch = _cfg.get("latent_channels", latent_ch)
        dec_ch_str = _cfg.get("decoder_channels", args.dec_ch)
        if isinstance(dec_ch_str, str):
            dec_ch = tuple(int(x) for x in dec_ch_str.split(","))
        elif isinstance(dec_ch_str, (list, tuple)):
            dec_ch = tuple(dec_ch_str)
        if _cfg.get("encoder_time_downscale") is not None:
            enc_time_str = str(_cfg["encoder_time_downscale"])
        if _cfg.get("decoder_time_upscale") is not None:
            dec_time_str = str(_cfg["decoder_time_upscale"])
        if _cfg.get("residual_shortcut") is not None:
            args.residual_shortcut = bool(_cfg["residual_shortcut"])
        if _cfg.get("use_attention") is not None:
            args.use_attention = bool(_cfg["use_attention"])
        if _cfg.get("use_groupnorm") is not None:
            args.use_groupnorm = bool(_cfg["use_groupnorm"])
        # Haar: backward compat (old checkpoints stored bool)
        _haar_raw = _cfg.get("haar", haar_mode)
        if _haar_raw is True:
            haar_mode = '2x'
        elif _haar_raw is False or _haar_raw is None:
            haar_mode = 'none'
        else:
            haar_mode = str(_haar_raw)
        del _ckpt_peek
    # Parse encoder channels
    if isinstance(enc_ch_str, int):
        enc_ch = enc_ch_str
    elif isinstance(enc_ch_str, str) and "," in enc_ch_str:
        enc_ch = tuple(int(x) for x in enc_ch_str.split(","))
    elif isinstance(enc_ch_str, (list, tuple)):
        enc_ch = tuple(enc_ch_str)
    else:
        enc_ch = int(enc_ch_str)
    n_stages = len(dec_ch)
    # Haar config
    if haar_mode is True:
        haar_mode = '2x'
    elif not haar_mode or haar_mode is False:
        haar_mode = 'none'
    haar_rounds = {'none': 0, '2x': 1, '4x': 2}.get(haar_mode, 0)
    haar_ch_mult = 4 ** haar_rounds
    haar_spatial_mult = 2 ** haar_rounds
    # vae_in_ch is always derived from haar_rounds — static checkpoints save image_channels=3 (RGB), not haar-expanded
    vae_in_ch = 3 * haar_ch_mult
    if haar_rounds > 0:
        print(f"Haar {haar_mode}: 3ch -> {vae_in_ch}ch ({haar_spatial_mult}x spatial pre-compression)")
    # Parse temporal config
    enc_t = tuple(x.strip().lower() in ("true", "1", "yes")
                  for x in enc_time_str.split(","))
    dec_t = tuple(x.strip().lower() in ("true", "1", "yes")
                  for x in dec_time_str.split(","))
    assert len(enc_t) == n_stages, \
        f"--enc-time length {len(enc_t)} != {n_stages} stages"
    assert len(dec_t) == n_stages, \
        f"--dec-time length {len(dec_t)} != {n_stages} stages"
    # Parse spatial config
    enc_spatial_str = _resume_cfg.get("encoder_spatial_downscale",
                                       getattr(args, 'enc_spatial', 'true,true,true'))
    dec_spatial_str = _resume_cfg.get("decoder_spatial_upscale",
                                       getattr(args, 'dec_spatial', 'true,true,true'))
    if isinstance(enc_spatial_str, str):
        enc_spatial = tuple(x.strip().lower() in ("true", "1", "yes")
                            for x in enc_spatial_str.split(","))
    else:
        enc_spatial = tuple(bool(x) for x in enc_spatial_str)
    if isinstance(dec_spatial_str, str):
        dec_spatial = tuple(x.strip().lower() in ("true", "1", "yes")
                            for x in dec_spatial_str.split(","))
    else:
        dec_spatial = tuple(bool(x) for x in dec_spatial_str)
    if len(enc_spatial) < n_stages:
        enc_spatial = enc_spatial + (True,) * (n_stages - len(enc_spatial))
    if len(dec_spatial) < n_stages:
        dec_spatial = dec_spatial + (True,) * (n_stages - len(dec_spatial))
    enc_spatial = enc_spatial[:n_stages]
    dec_spatial = dec_spatial[:n_stages]

    model = MiniVAE(
        latent_channels=latent_ch,
        image_channels=vae_in_ch,
        output_channels=vae_in_ch,
        encoder_channels=enc_ch,
        decoder_channels=dec_ch,
        encoder_time_downscale=enc_t,
        decoder_time_upscale=dec_t,
        encoder_spatial_downscale=enc_spatial,
        decoder_spatial_upscale=dec_spatial,
        residual_shortcut=getattr(args, 'residual_shortcut', False),
        use_attention=getattr(args, 'use_attention', False),
        use_groupnorm=getattr(args, 'use_groupnorm', False),
    ).to(device)
    if args.grad_checkpoint:
        model.use_checkpoint = True
    pc = model.param_count()
    spatial = model.s_downscale * haar_spatial_mult
    print(f"MiniVAE ({vae_in_ch}ch, {spatial}x spatial): {pc['total']:,} params"
          f"{', grad-checkpoint' if args.grad_checkpoint else ''}")
    print(f"  t_downscale={model.t_downscale}, t_upscale={model.t_upscale}, "
          f"frames_to_trim={model.frames_to_trim}")

    # -- Generator --
    gen = VAEpp0rGenerator(
        height=args.H, width=args.W, device=str(device),
        bank_size=args.bank_size,
        n_base_layers=args.n_layers,
        alpha=args.alpha,
    )
    bank_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bank")
    bank_files = [f for f in os.listdir(bank_dir)
                  if f.startswith("shapes_") and f.endswith(".pt")] \
        if os.path.isdir(bank_dir) else []
    if bank_files:
        gen.setup_dynamic_bank(bank_dir, working_size=args.bank_size,
                                refresh_interval=50)
        gen.build_base_layers()
    else:
        gen.build_banks()
    # -- Effect flags for the motion pool. Earlier runs built the pool with
    # zero flags, so recipes had none of the procedural effects (ripple,
    # shake, kaleido, flash, palette, text, signage, particles, raymarch,
    # arcade, fire, vortex, starfield, eq). Now defaults are all ON and
    # random_mix drops each per-recipe so the pool has variety.
    pool_kwargs = dict(
        use_fluid=getattr(args, "use_fluid", True),
        use_ripple=getattr(args, "use_ripple", True),
        use_shake=getattr(args, "use_shake", True),
        use_kaleido=getattr(args, "use_kaleido", True),
        fast_transform=getattr(args, "fast_transform", True),
        use_flash=getattr(args, "use_flash", True),
        use_palette_cycle=getattr(args, "use_palette_cycle", True),
        use_text=getattr(args, "use_text", True),
        use_signage=getattr(args, "use_signage", True),
        use_particles=getattr(args, "use_particles", True),
        use_raymarch=getattr(args, "use_raymarch", True),
        sphere_dip=getattr(args, "sphere_dip", True),
        use_arcade=getattr(args, "use_arcade", True),
        use_glitch=getattr(args, "use_glitch", True),
        use_chromatic=getattr(args, "use_chromatic", True),
        use_scanlines=getattr(args, "use_scanlines", True),
        use_fire=getattr(args, "use_fire", True),
        use_vortex=getattr(args, "use_vortex", True),
        use_starfield=getattr(args, "use_starfield", True),
        use_eq=getattr(args, "use_eq", True),
    )
    gen._train_pool_kwargs = pool_kwargs
    gen._train_random_mix = getattr(args, "random_mix", True)

    def _pool_health(recipes, kwargs, low=0.02, high=0.85):
        """Flag stale pools by prevalence check — same logic as train_video3d."""
        flag_to_key = {
            "use_fluid": "fluid", "use_ripple": "fluid",
            "use_shake": "shake", "use_kaleido": "kaleido",
            "use_flash": "flash", "use_palette_cycle": "palette",
            "use_text": "text", "use_signage": "signage",
            "use_particles": "particles", "use_raymarch": "raymarch",
            "sphere_dip": "raymarch", "use_arcade": "arcade",
            "use_glitch": "glitch", "use_chromatic": "chromatic",
            "use_scanlines": "scanline",
            "use_fire": "fire", "use_vortex": "vortex",
            "use_starfield": "starfield", "use_eq": "eq",
        }
        import random as _rnd
        n = min(50, len(recipes))
        sample = _rnd.sample(recipes, n) if n else []
        problems = []
        for flag, key in flag_to_key.items():
            if not kwargs.get(flag):
                continue
            p = sum(1 for r in sample if key in r) / max(n, 1)
            if p < low:
                problems.append(f"MISSING {flag}->{key}")
            elif p > high:
                problems.append(f"OVER {flag}->{key} p={p:.2f}")
        return problems

    # Build motion clip pool. Default: always rebuild (same policy as
    # train_video3d). Pass --reuse-pool to opt into reusing an existing
    # motion_pool.json — but even then, validate it's not stale before use.
    pool_path = os.path.join(bank_dir, "motion_pool.json")
    reuse_requested = getattr(args, "reuse_pool", False)
    rebuild = True
    if reuse_requested and os.path.exists(pool_path):
        gen.load_motion_pool(pool_path)
        problems = _pool_health(gen._recipe_pool, pool_kwargs)
        if problems:
            print(f"  --reuse-pool requested but pool is stale: {problems}")
            print(f"  Rebuilding from scratch.")
        else:
            rebuild = False
            print(f"  Reusing existing pool ({len(gen._recipe_pool)} recipes)")
    elif os.path.exists(pool_path) and not reuse_requested:
        print(f"  Ignoring existing {pool_path} (pass --reuse-pool to reuse). "
              f"Rebuilding to guarantee current effect distribution.")

    if rebuild:
        gen.build_motion_pool(
            n_clips=args.pool_size, T=args.T,
            random_mix=getattr(args, "random_mix", True), **pool_kwargs)
        try:
            gen.save_motion_pool(pool_path)
        except Exception as e:
            print(f"  (could not save rebuilt pool: {e})")

    if args.disco:
        gen.disco_quadrant = True
    print(f"Generator: bank={gen.bank_size}, layers={args.n_layers}, "
          f"T={args.T}, pool={gen.motion_pool_stats()}, "
          f"disco={gen.disco_quadrant}")

    # -- PatchGAN discriminator --
    disc = None
    opt_disc = None
    if args.w_gan > 0:
        disc = PatchDiscriminator(in_ch=3, nf=args.disc_nf).to(device)
        opt_disc = torch.optim.Adam(disc.parameters(), lr=float(args.lr),
                                    betas=(0.5, 0.999))
        pc_d = sum(p.numel() for p in disc.parameters())
        print(f"PatchGAN: nf={args.disc_nf}, {pc_d:,} params, "
              f"w_gan={args.w_gan}, start_step={args.gan_start}")

    # -- LPIPS --
    lpips_fn = None
    if args.w_lpips > 0:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="squeeze").to(device)
            lpips_fn.eval()
            lpips_fn.requires_grad_(False)
            print(f"LPIPS (squeeze) on {device}")
        except ImportError:
            print("WARNING: pip install lpips")

    # -- Optimizer --
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr),
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.total_steps, eta_min=float(args.lr) * 0.01)

    # -- Resume --
    global_step = 0
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "model" in ckpt:
            src_sd = ckpt["model"]
            target_sd = model.state_dict()
            loaded, converted, skipped = 0, 0, 0
            for k, v in src_sd.items():
                if k not in target_sd:
                    skipped += 1
                    continue
                t = target_sd[k]
                if v.shape == t.shape:
                    target_sd[k] = v
                    loaded += 1
                elif v.ndim == 4 and t.ndim == 4:
                    # TPool: (n_f, n_f, 1, 1) -> (n_f, 2*n_f, 1, 1)
                    if t.shape[0] == v.shape[0] and t.shape[1] == v.shape[1] * 2:
                        t2 = torch.zeros_like(t)
                        t2[:, :v.shape[1]] = v
                        target_sd[k] = t2
                        converted += 1
                    # TGrow: (n_f, n_f, 1, 1) -> (2*n_f, n_f, 1, 1)
                    elif t.shape[1] == v.shape[1] and t.shape[0] == v.shape[0] * 2:
                        t2 = torch.zeros_like(t)
                        t2[:v.shape[0]] = v
                        t2[v.shape[0]:] = v
                        target_sd[k] = t2
                        converted += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1
                    print(f"  Skipped: {k} src={list(v.shape)} dst={list(target_sd[k].shape) if k in target_sd else 'missing'}")
            model.load_state_dict(target_sd)
            print(f"  Loaded {loaded} layers, {converted} converted "
                  f"(TPool/TGrow), {skipped} skipped")
            global_step = ckpt.get("global_step", 0)
            if not args.fresh_opt and ckpt.get("optimizer"):
                try:
                    opt.load_state_dict(ckpt["optimizer"])
                except Exception:
                    print("  Fresh optimizer (mismatch)")
            if disc is not None and ckpt.get("discriminator"):
                try:
                    disc.load_state_dict(ckpt["discriminator"])
                    if not args.fresh_opt and ckpt.get("optimizer_disc"):
                        opt_disc.load_state_dict(ckpt["optimizer_disc"])
                except Exception:
                    print("  Fresh discriminator (mismatch)")
        else:
            model.load_state_dict(ckpt, strict=False)
        print(f"Resumed from {args.resume} at step {global_step}")

    if args.fresh_opt and global_step > 0:
        opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr),
                                weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.total_steps - global_step,
            eta_min=float(args.lr) * 0.01)
        print(f"  Fresh optimizer from step {global_step}")

    # -- Precision --
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}[args.precision]
    scaler = torch.amp.GradScaler("cuda",
                                   enabled=(args.precision == "fp16"))

    if global_step > 0 and not args.fresh_opt and ckpt is not None:
        if ckpt.get("scheduler"):
            sched.load_state_dict(ckpt["scheduler"])
        else:
            # Fallback for old checkpoints: rebuild at correct position.
            # CosineAnnealingLR with last_epoch != -1 requires 'initial_lr'
            # in each param group — seed it from the current lr.
            for _pg in opt.param_groups:
                _pg.setdefault("initial_lr", _pg["lr"])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.total_steps, eta_min=float(args.lr) * 0.01,
                last_epoch=global_step)
        if ckpt.get("scaler") and args.precision == "fp16":
            scaler.load_state_dict(ckpt["scaler"])

    gen_bs = args.gen_batch if args.gen_batch > 0 else args.batch_size
    accum = args.grad_accum

    # -- Temporal warmup: freeze spatial weights, train only TPool/TGrow/MemBlock --
    _temporal_param_names = set()
    for _mname, _mod in model.named_modules():
        if type(_mod).__name__ in ("TPool", "TGrow", "MemBlock"):
            for _pname, _ in _mod.named_parameters(recurse=True):
                _full = f"{_mname}.{_pname}" if _mname else _pname
                _temporal_param_names.add(_full)

    _in_warmup = args.warmup_steps > 0 and global_step < args.warmup_steps
    if _in_warmup:
        for _n, _p in model.named_parameters():
            _p.requires_grad_(_n in _temporal_param_names)
        _warmup_params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(_warmup_params, lr=float(args.lr), weight_decay=0.01)
        # CosineAnnealingLR with last_epoch != -1 requires 'initial_lr' in
        # each param group. A freshly-built AdamW doesn't set it — seed it
        # from the current lr before constructing the scheduler.
        if global_step > 0:
            for _pg in opt.param_groups:
                _pg.setdefault("initial_lr", _pg["lr"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.total_steps, eta_min=float(args.lr) * 0.01,
            last_epoch=global_step - 1 if global_step > 0 else -1)
        _steps_left = args.warmup_steps - global_step
        print(f"Warmup: spatial frozen, training {len(_warmup_params)} temporal param tensors "
              f"({_steps_left} steps remaining)")

    print(f"Steps: {args.total_steps}, LR: {args.lr}, "
          f"Batch: {args.batch_size}, T: {args.T}"
          f"{f', accum={accum}' if accum > 1 else ''}"
          f"{f', gen-batch={gen_bs}' if gen_bs != args.batch_size else ''}"
          f"{f', warmup={args.warmup_steps}' if args.warmup_steps > 0 else ''}")
    print(f"Weights: mse={args.w_mse} lpips={args.w_lpips} "
          f"temporal={args.w_temporal}")
    print(f"Precision: {args.precision}, Device: {device}")
    print(flush=True)

    def _ckpt_name(step):
        ec_str = ",".join(str(x) for x in enc_ch) if isinstance(enc_ch, tuple) else str(enc_ch)
        dc_str = ",".join(str(x) for x in dec_ch)
        n_stages = len(dec_ch)
        spatial = 2 ** n_stages
        t_down = sum(1 for x in enc_t if x)
        temporal = 2 ** t_down if t_down > 0 else 1
        steps_k = f"{step // 1000}k" if step >= 1000 else str(step)
        return f"vae-3ch-lc{latent_ch}-ec{ec_str}-dc{dc_str}-S{spatial}x-T{temporal}x-{steps_k}.pt"

    # Glob pattern scoped to THIS run only — never touch other runs' checkpoints
    _ec_str = ",".join(str(x) for x in enc_ch) if isinstance(enc_ch, tuple) else str(enc_ch)
    _dc_str = ",".join(str(x) for x in dec_ch)
    _n_stages = len(dec_ch)
    _spatial = 2 ** _n_stages
    _t_down = sum(1 for x in enc_t if x)
    _temporal = 2 ** _t_down if _t_down > 0 else 1
    _run_glob = f"vae-3ch-lc{latent_ch}-ec{_ec_str}-dc{_dc_str}-S{_spatial}x-T{_temporal}x-*.pt"

    def _make_checkpoint():
        d = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "global_step": global_step,
            "config": {
                "latent_channels": latent_ch,
                "image_channels": 3,
                "output_channels": 3,
                "encoder_channels": ",".join(str(x) for x in enc_ch) if isinstance(enc_ch, tuple) else enc_ch,
                "decoder_channels": ",".join(str(x) for x in dec_ch),
                "encoder_time_downscale": ",".join(str(x) for x in enc_t),
                "decoder_time_upscale": ",".join(str(x) for x in dec_t),
                "encoder_spatial_downscale": ",".join(str(s).lower() for s in enc_spatial),
                "decoder_spatial_upscale": ",".join(str(s).lower() for s in dec_spatial),
                "residual_shortcut": getattr(args, 'residual_shortcut', False),
                "use_attention": getattr(args, 'use_attention', False),
                "use_groupnorm": getattr(args, 'use_groupnorm', False),
                "haar": haar_mode,
                "temporal": True,
                "T": args.T,
                "synthyper_stage": 2,
            },
        }
        if disc is not None:
            d["discriminator"] = disc.state_dict()
            d["optimizer_disc"] = opt_disc.state_dict()
        return d

    # -- Initial preview --
    preview_image = getattr(args, 'preview_image', None)
    save_preview(model, gen, str(logdir), global_step, device, amp_dtype,
                 args.T, preview_T=getattr(args, 'preview_T', None),
                 preview_image=preview_image,
                 preview_frame_skip=getattr(args, 'preview_frame_skip', 0),
                 haar_rounds=haar_rounds)

    # -- Loop --
    t0 = time.time()
    start_step = global_step

    stop_file = logdir / ".stop"
    if stop_file.exists():
        stop_file.unlink()

    while global_step < args.total_steps:
        if _stop_requested or stop_file.exists():
            if stop_file.exists():
                stop_file.unlink()
            print("[Stop detected, saving...]", flush=True)
            break

        model.train()
        if disc is not None:
            disc.train()
        opt.zero_grad(set_to_none=True)
        losses = {}

        _d_real = None
        _d_fake = None

        for _ai in range(accum):
            # Sample from motion pool (fast) or generate fresh (slow)
            if gen_bs < args.batch_size:
                chunks = []
                rem = args.batch_size
                while rem > 0:
                    n = min(gen_bs, rem)
                    chunks.append(gen.generate_from_pool(n))
                    rem -= n
                clips = torch.cat(chunks)
            else:
                clips = gen.generate_from_pool(args.batch_size)  # (B, T, 3, H, W)
            clips = clips.to(device)
            if haar_rounds > 0:
                B, T, C, H, W = clips.shape
                x = torch.stack([haar_down_n(clips[:, t], haar_rounds)
                                  for t in range(T)], dim=1)
            else:
                x = clips

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                recon, latent = model(x)

                # Align temporal and crop spatial to input dims
                T_out = recon.shape[1]
                T_in = x.shape[1]
                T_match = min(T_out, T_in)
                gt = x[:, T_in - T_match:]
                hH = args.H // haar_spatial_mult
                hW = args.W // haar_spatial_mult
                rc = recon[:, T_out - T_match:, :, :hH, :hW]

                # Reconstruction loss
                total = torch.tensor(0.0, device=device)
                if args.w_l1 > 0:
                    l1 = F.l1_loss(rc, gt[:, :, :, :hH, :hW])
                    total = total + args.w_l1 * l1
                    losses["l1"] = losses.get("l1", 0) + l1.item() / accum
                if args.w_mse > 0:
                    mse = F.mse_loss(rc, gt[:, :, :, :hH, :hW])
                    total = total + args.w_mse * mse
                    losses["mse"] = losses.get("mse", 0) + mse.item() / accum

                # Temporal consistency loss
                if T_match >= 2:
                    gt_diff = gt[:, 1:] - gt[:, :-1]
                    rc_diff = rc[:, 1:] - rc[:, :-1]
                    temp = F.l1_loss(rc_diff, gt_diff)
                    total = total + args.w_temporal * temp
                    losses["temp"] = losses.get("temp", 0) + temp.item() / accum

                # LPIPS + GAN: need RGB — decode haar once for both
                B_rc = rc.shape[0]
                if haar_rounds > 0:
                    rc_2d = rc.reshape(B_rc * T_match, rc.shape[2], hH, hW)
                    fake_2d = rc_2d
                    for _ in range(haar_rounds):
                        fake_2d = haar_up(fake_2d)
                    rc_rgb = fake_2d[:, :3, :args.H, :args.W].reshape(B_rc, T_match, 3, args.H, args.W)
                    gt_rgb = clips[:, T_in - T_match:, :3, :args.H, :args.W]
                else:
                    rc_rgb = rc[:, :, :3, :args.H, :args.W]
                    gt_rgb = gt[:, :, :3, :args.H, :args.W]

                # LPIPS (per-frame, on RGB)
                if lpips_fn is not None:
                    BT = B_rc * T_match
                    rc_lp = rc_rgb.reshape(BT, 3, args.H, args.W) * 2 - 1
                    gt_lp = gt_rgb.reshape(BT, 3, args.H, args.W) * 2 - 1
                    lp_chunks = []
                    for ci in range(0, BT, 4):
                        lp_chunks.append(lpips_fn(rc_lp[ci:ci+4], gt_lp[ci:ci+4]))
                    lp = torch.cat(lp_chunks, 0).mean()
                    total = total + args.w_lpips * lp
                    losses["lpips"] = losses.get("lpips", 0) + lp.item() / accum

                # GAN generator loss (per-frame discriminator)
                if disc is not None and global_step >= args.gan_start:
                    BT = B_rc * T_match
                    fake_bt = rc_rgb.reshape(BT, 3, args.H, args.W)
                    g_loss = hinge_g_loss(disc(fake_bt * 2 - 1))
                    total = total + args.w_gan * g_loss
                    losses["g"] = losses.get("g", 0) + g_loss.item() / accum
                    _d_real = gt_rgb.reshape(BT, 3, args.H, args.W).detach()
                    _d_fake = fake_bt.detach()

            if total.dim() > 0:
                total = total.mean()

            scaler.scale(total / accum).backward()
            del clips, x, recon, latent, rc, gt

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        # D update (once per step, after G)
        if disc is not None and global_step >= args.gan_start and _d_real is not None:
            opt_disc.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                d_loss = hinge_d_loss(disc(_d_real * 2 - 1), disc(_d_fake * 2 - 1))
            d_loss.backward()
            opt_disc.step()
            losses["d"] = d_loss.item()

        global_step += 1

        # -- Warmup -> full training transition --
        if args.warmup_steps > 0 and global_step == args.warmup_steps:
            print("Warmup complete — unfreezing all parameters", flush=True)
            for _p in model.parameters():
                _p.requires_grad_(True)
            opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr),
                                    weight_decay=0.01)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.total_steps - global_step,
                eta_min=float(args.lr) * 0.01)

        # -- Log --
        if global_step % args.log_every == 0:
            el = time.time() - t0
            steps_run = global_step - start_step
            sps = steps_run / max(el, 1)
            eta = (args.total_steps - global_step) / max(sps, 1e-6)
            lr = opt.param_groups[0]["lr"]
            vm = torch.cuda.memory_allocated() / 1e9
            ls = " ".join(f"{k}={v:.4f}" for k, v in losses.items())
            eta_str = f"{eta/3600:.1f}h" if eta > 3600 else f"{eta/60:.0f}m"
            print(f"[{global_step}/{args.total_steps}] {ls} "
                  f"lr={lr:.1e} vram={vm:.1f}G "
                  f"({sps:.1f} step/s, {eta_str} left)", flush=True)

        # -- Preview --
        if global_step % args.preview_every == 0:
            save_preview(model, gen, str(logdir), global_step,
                         device, amp_dtype, args.T,
                         preview_T=getattr(args, 'preview_T', None),
                         preview_image=preview_image,
                         preview_frame_skip=getattr(args, 'preview_frame_skip', 0),
                         haar_rounds=haar_rounds)

        # -- Checkpoint --
        if global_step % args.save_every == 0:
            d = _make_checkpoint()
            named = _ckpt_name(global_step)
            torch.save(d, logdir / named)
            torch.save(d, logdir / "latest.pt")
            print(f"  saved {named}", flush=True)

            ckpts = sorted([f for f in logdir.glob(_run_glob)
                           if f.name != "latest.pt"],
                           key=lambda x: x.stat().st_mtime)
            while len(ckpts) > 10:
                ckpts.pop(0).unlink()

    # Save on exit
    if global_step > start_step:
        d = _make_checkpoint()
        named = _ckpt_name(global_step)
        torch.save(d, logdir / named)
        torch.save(d, logdir / "latest.pt")
        print(f"  saved {named}", flush=True)

    print(f"\nDone. {global_step - start_step} steps in "
          f"{(time.time() - t0) / 60:.1f}min", flush=True)


def main():
    p = argparse.ArgumentParser(description="VAEpp0r Stage 2 — temporal")
    p.add_argument("--H", type=int, default=360)
    p.add_argument("--W", type=int, default=640)
    p.add_argument("--T", type=int, default=24)
    p.add_argument("--latent-ch", type=int, default=32)
    p.add_argument("--enc-ch", default="64",
                   help="Encoder channel width (int or comma-separated per stage)")
    p.add_argument("--dec-ch", default="256,128,64",
                   help="Decoder channel widths (comma-separated, one per stage)")
    p.add_argument("--enc-time", default="true,true,false",
                   help="Temporal downscale per encoder stage (comma-separated bools)")
    p.add_argument("--dec-time", default="false,true,true",
                   help="Temporal upscale per decoder stage (comma-separated bools)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", default="2e-4")
    p.add_argument("--total-steps", type=int, default=30000)
    p.add_argument("--precision", default="bf16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--w-l1", type=float, default=1.0)
    p.add_argument("--w-mse", type=float, default=0.0)
    p.add_argument("--w-lpips", type=float, default=0.5)
    p.add_argument("--w-temporal", type=float, default=2.0)
    p.add_argument("--w-gan", type=float, default=0.0,
                   help="PatchGAN adversarial loss weight (0=disabled)")
    p.add_argument("--gan-start", type=int, default=1000,
                   help="Step to start GAN training (let recon stabilize first)")
    p.add_argument("--disc-nf", type=int, default=64,
                   help="PatchGAN base channel count")
    p.add_argument("--bank-size", type=int, default=5000)
    p.add_argument("--n-layers", type=int, default=128)
    p.add_argument("--pool-size", type=int, default=200)
    p.add_argument("--alpha", type=float, default=3.0)
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps (effective batch = batch-size * grad-accum)")
    p.add_argument("--gen-batch", type=int, default=0,
                   help="Generator batch size (0 = same as batch-size). "
                        "Lower to reduce generator VRAM.")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="Gradient checkpointing (trade compute for memory)")
    p.add_argument("--disco", action="store_true",
                   help="Enable disco quadrant mode")
    # -- Pool / effect flags (mirror train_video3d). All default True,
    # random-mix drops them per-recipe so training sees variety. Pass
    # --reuse-pool to skip the "always rebuild" policy.
    def _bool(s):
        return str(s).strip().lower() in ("1", "true", "yes", "y", "on")
    p.add_argument("--reuse-pool", action="store_true",
                   help="Reuse bank/motion_pool.json if present and "
                        "coverage checks pass. Default: rebuild every run.")
    p.add_argument("--random-mix", type=_bool, default=True,
                   help="Per-recipe random subset of enabled effects")
    for flag, default, desc in [
        ("use-fluid",         True, "fluid flow"),
        ("use-ripple",        True, "ripple warp"),
        ("use-shake",         True, "camera shake"),
        ("use-kaleido",       True, "kaleidoscope"),
        ("fast-transform",    True, "fast transforms"),
        ("use-flash",         True, "flash frames"),
        ("use-palette-cycle", True, "palette cycling"),
        ("use-text",          True, "text overlays"),
        ("use-signage",       True, "signage overlays"),
        ("use-particles",     True, "particles"),
        ("use-raymarch",      True, "raymarch primitives"),
        ("sphere-dip",        True, "sphere-dip scene"),
        ("use-arcade",        True, "arcade scenes"),
        ("use-glitch",        True, "glitch bursts"),
        ("use-chromatic",     True, "chromatic aberration"),
        ("use-scanlines",     True, "scanlines"),
        ("use-fire",          True, "fire texture"),
        ("use-vortex",        True, "vortex swirl"),
        ("use-starfield",     True, "starfield zoom"),
        ("use-eq",            True, "EQ bars"),
    ]:
        p.add_argument(f"--{flag}",
                       dest=flag.replace("-", "_"),
                       default=default, type=_bool,
                       help=f"Enable {desc} (default {default})")
    p.add_argument("--haar", default="none", choices=["none", "2x", "4x"],
                   help="Haar wavelet pre-compression (none/2x/4x)")
    p.add_argument("--residual-shortcut", action="store_true",
                   help="DC-AE style residual shortcuts (pixel_unshuffle/shuffle bypasses)")
    p.add_argument("--use-attention", action="store_true",
                   help="Add linear attention at deepest encoder/decoder stage")
    p.add_argument("--use-groupnorm", action="store_true",
                   help="Add GroupNorm inside MemBlock conv layers")
    p.add_argument("--enc-spatial", default="true,true,true",
                   help="Spatial downscale per encoder stage (comma-separated bools)")
    p.add_argument("--dec-spatial", default="true,true,true",
                   help="Spatial upscale per decoder stage (comma-separated bools)")
    p.add_argument("--warmup-steps", type=int, default=500,
                   help="Steps to train only temporal layers (TPool/TGrow/MemBlock) "
                        "before unfreezing spatial weights. Set 0 to disable.")
    p.add_argument("--resume", default=None)
    p.add_argument("--fresh-opt", action="store_true")
    p.add_argument("--logdir", default="synthyper_video_logs")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--preview-every", type=int, default=100)
    p.add_argument("--preview-image", default=None,
                   help="Path to reference video (mp4) for tracking progress")
    p.add_argument("--preview-frame-skip", type=int, default=0,
                   help="Number of frames to skip into the reference video")
    p.add_argument("--preview-T", type=int, default=None,
                   help="Preview clip length in frames (default: same as --T)")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
