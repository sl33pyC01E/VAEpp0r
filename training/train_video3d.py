#!/usr/bin/env python3
"""VAEpp0r Stage 2 (3D) — temporal training with Cosmos-style causal 3D architecture.

Uses MiniVAE3D: factorized causal 3D convs, temporal + spatial attention,
hybrid down/up sampling, optional 3D Haar patching, optional residual FSQ.

Usage:
    python -m training.train_video3d
    python -m training.train_video3d --resume synthyper_video3d_logs/latest.pt --fresh-opt
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

from core.model import MiniVAE3D
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
        # Must produce TWO DIFFERENT clips. Both the pool path and the
        # live path have had "all-N-clips-identical" bugs; this block
        # guards against them both.
        with torch.random.fork_rng():
            torch.manual_seed(step + int(time.time()) % 100000)
            has_pool = (hasattr(gen, "_recipe_pool")
                        and gen._recipe_pool
                        and getattr(gen, "_motion_pool_T", None) == pT)
            if has_pool:
                # generate_from_pool picks a random recipe per sample in
                # its internal loop. Force distinct indices by sampling
                # and rendering one at a time so there's no way B sees
                # only one recipe.
                N = len(gen._recipe_pool)
                import random as _rnd2
                idxs = _rnd2.sample(range(N), k=min(2, N))
                if len(idxs) < 2:
                    idxs = idxs + [idxs[0]]  # degenerate 1-recipe pool
                parts = []
                for idx in idxs:
                    parts.append(gen._render_recipe(gen._recipe_pool[idx])
                                 .unsqueeze(0))
                clips = torch.cat(parts, dim=0)
                if step == 0 or step < 5:
                    print(f"  [preview] pool path, idxs={idxs} of {N} recipes")
            else:
                # Fallback: render fresh WITHOUT the B=N-shared-kwargs
                # bug that historically produced N identical clips.
                # Loop B=1 twice, re-rolling random_mix each time so
                # effect stacks differ between the two tiles.
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
                if step == 0 or step < 5:
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
    # Loud banner — if you don't see this line in the training log, the
    # process is running old code and the recent preview/pool fixes are
    # NOT active. Kill the subprocess (the GUI restart won't kill it)
    # and relaunch training from the GUI.
    print("=" * 60)
    print("  VAEpp0r train_video3d — preview/pool fixes v2 active")
    print("  - pool rebuilt on every run unless --reuse-pool")
    print("  - preview samples 2 distinct recipes explicitly")
    print("  - preview has [preview] path logs for first 5 steps")
    print("=" * 60, flush=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    logdir = pathlib.Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    # -- Model (MiniVAE3D: causal 3D + optional Haar patcher + optional FSQ) --
    latent_ch = args.latent_ch
    # Channel schedule: prefer explicit --enc-ch/--dec-ch if given; else
    # fall back to --base-ch/--ch-mult for back-compat. Same resolution
    # order both ways: legacy mult is shallow->deep, and the fallback
    # computes enc = base*mult shallow->deep, dec = reversed (deep->shallow).
    enc_ch_str = args.enc_ch
    dec_ch_str = args.dec_ch
    base_ch = args.base_ch
    ch_mult_str = args.ch_mult
    num_res_blocks = args.num_res_blocks
    temporal_down_str = args.temporal_down
    spatial_down_str = args.spatial_down
    haar_levels = args.haar_levels
    use_fsq = args.fsq
    fsq_levels_str = args.fsq_levels
    fsq_stages = args.fsq_stages
    # Arch extras (resolved from resume config if present, else CLI)
    use_attention = args.use_attention
    use_groupnorm = args.use_groupnorm
    residual_shortcut = args.residual_shortcut
    attn_heads = args.attn_heads
    gn_groups = args.gn_groups

    # Resume: read config from checkpoint. Newer checkpoints store
    # encoder_channels/decoder_channels directly; older ones only have
    # base_channels/channel_mult.
    _resume_cfg = {}
    if args.resume:
        _ckpt_peek = torch.load(args.resume, map_location="cpu", weights_only=False)
        _resume_cfg = _ckpt_peek.get("config", {})
        latent_ch = _resume_cfg.get("latent_channels", latent_ch)
        if "encoder_channels" in _resume_cfg and "decoder_channels" in _resume_cfg:
            enc_ch_str = _resume_cfg["encoder_channels"]
            dec_ch_str = _resume_cfg["decoder_channels"]
        else:
            base_ch = _resume_cfg.get("base_channels", base_ch)
            ch_mult_str = _resume_cfg.get("channel_mult", ch_mult_str)
        num_res_blocks = _resume_cfg.get("num_res_blocks", num_res_blocks)
        temporal_down_str = _resume_cfg.get("temporal_downsample", temporal_down_str)
        spatial_down_str = _resume_cfg.get("spatial_downsample", spatial_down_str)
        haar_levels = _resume_cfg.get("haar_levels", haar_levels)
        use_fsq = _resume_cfg.get("fsq", use_fsq)
        fsq_levels_str = _resume_cfg.get("fsq_levels", fsq_levels_str)
        fsq_stages = _resume_cfg.get("fsq_stages", fsq_stages)
        # Arch extras — if missing in older checkpoint, keep CLI values
        use_attention = _resume_cfg.get("use_attention", use_attention)
        use_groupnorm = _resume_cfg.get("use_groupnorm", use_groupnorm)
        residual_shortcut = _resume_cfg.get("residual_shortcut", residual_shortcut)
        attn_heads = _resume_cfg.get("attn_heads", attn_heads)
        gn_groups = _resume_cfg.get("gn_groups", gn_groups)
        del _ckpt_peek

    def _parse_tuple_int(v):
        if isinstance(v, (list, tuple)):
            return tuple(int(x) for x in v)
        return tuple(int(x.strip()) for x in str(v).split(","))

    def _parse_tuple_bool(v):
        if isinstance(v, (list, tuple)):
            return tuple(bool(x) for x in v)
        return tuple(x.strip().lower() in ("true", "1", "yes") for x in str(v).split(","))

    # Resolve enc/dec channel tuples
    if enc_ch_str and dec_ch_str:
        enc_channels = _parse_tuple_int(enc_ch_str)
        dec_channels = _parse_tuple_int(dec_ch_str)
    else:
        ch_mult = _parse_tuple_int(ch_mult_str)
        enc_channels = tuple(base_ch * m for m in ch_mult)
        dec_channels = tuple(reversed(enc_channels))
    temporal_down = _parse_tuple_bool(temporal_down_str)
    spatial_down = _parse_tuple_bool(spatial_down_str)
    n_levels = len(enc_channels)
    assert len(dec_channels) == n_levels, \
        f"--enc-ch and --dec-ch must have the same length: enc={enc_channels} dec={dec_channels}"
    assert enc_channels[-1] == dec_channels[0], \
        (f"bottleneck mismatch: enc_channels[-1]={enc_channels[-1]} must "
         f"equal dec_channels[0]={dec_channels[0]}. No projection layer "
         f"between encoder and decoder at the bottleneck.")
    assert len(temporal_down) == n_levels, \
        f"--temporal-down length {len(temporal_down)} != {n_levels} levels"
    assert len(spatial_down) == n_levels, \
        f"--spatial-down length {len(spatial_down)} != {n_levels} levels"
    fsq_levels = _parse_tuple_int(fsq_levels_str)

    model = MiniVAE3D(
        latent_channels=latent_ch,
        image_channels=3,
        output_channels=3,
        enc_channels=enc_channels,
        dec_channels=dec_channels,
        num_res_blocks=num_res_blocks,
        temporal_downsample=temporal_down,
        spatial_downsample=spatial_down,
        use_attention=bool(use_attention),
        use_groupnorm=bool(use_groupnorm),
        residual_shortcut=bool(residual_shortcut),
        attn_heads=int(attn_heads),
        gn_groups=int(gn_groups),
        dropout=0.0,
        haar_levels=haar_levels,
        fsq=use_fsq,
        fsq_levels=fsq_levels,
        fsq_stages=fsq_stages,
    ).to(device)
    pc = model.param_count()
    print(f"MiniVAE3D: {pc['total']:,} params  "
          f"(enc {pc['encoder']:,} + dec {pc['decoder']:,})")
    # Single source of truth for the arch summary — no hidden hardcodes:
    print(model.config_summary())
    if use_fsq:
        print(f"  fsq stages={fsq_stages}  levels={fsq_levels}")

    # Compat shims so downstream code that references old attrs still works.
    # MiniVAE3D doesn't have t_upscale (symmetric) or frames_to_trim.
    if not hasattr(model, 't_upscale'):
        model.t_upscale = model.t_downscale
    if not hasattr(model, 'frames_to_trim'):
        model.frames_to_trim = 0
    # Haar is now inside the model — disable training-loop Haar
    haar_rounds = 0
    haar_ch_mult = 1
    haar_spatial_mult = 1
    vae_in_ch = 3

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
    # -- Effect flags for the motion pool. Earlier training runs built
    # the pool with zero flags, which meant recipes had none of the new
    # effects (ripple/shake/kaleido/text/signage/particles/raymarch/
    # arcade/fire/vortex/starfield/eq/etc). Now defaults are all ON and
    # random_mix drops each per-recipe so the pool has variety.
    pool_kwargs = dict(
        use_fluid=args.use_fluid,
        use_ripple=args.use_ripple,
        use_shake=args.use_shake,
        use_kaleido=args.use_kaleido,
        fast_transform=args.fast_transform,
        use_flash=args.use_flash,
        use_palette_cycle=args.use_palette_cycle,
        use_text=args.use_text,
        use_signage=args.use_signage,
        use_particles=args.use_particles,
        use_raymarch=args.use_raymarch,
        sphere_dip=args.sphere_dip,
        use_arcade=args.use_arcade,
        use_glitch=args.use_glitch,
        use_chromatic=args.use_chromatic,
        use_scanlines=args.use_scanlines,
        use_fire=args.use_fire,
        use_vortex=args.use_vortex,
        use_starfield=args.use_starfield,
        use_eq=args.use_eq,
    )
    # Keep on the generator so preview/refresh reuse the same distribution
    gen._train_pool_kwargs = pool_kwargs
    gen._train_random_mix = args.random_mix

    # Build motion clip pool.
    #
    # Default: ALWAYS rebuild. The cached pool file has historically been
    # a source of bugs — built under an older build_motion_pool that rolled
    # effects once for the whole pool (so every recipe had EQ, or none did)
    # and silently loaded on startup. A 200-recipe rebuild takes seconds.
    # Pass --reuse-pool to opt into reusing an existing file.
    pool_path = os.path.join(bank_dir, "motion_pool.json")

    def _pool_health(recipes, kwargs, low=0.02, high=0.85):
        """Sample 50 recipes; for each enabled effect flag, compute the
        prevalence of the corresponding recipe block. Return a list of
        problems:
          - 'MISSING flag->key' if prevalence < low (effect never fires)
          - 'OVER flag->key p=0.97' if prevalence > high (single-roll bug)
        Either case means the pool is stale.
        """
        # Singular key names — matches what _generate_recipe stores.
        # Getting these wrong made the health check fire false alarms
        # ("MISSING use_flash->flashes") on otherwise-correct pools.
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
            random_mix=args.random_mix, **pool_kwargs)
        # Report effect coverage so you can see at a glance what's in the pool
        try:
            problems = _pool_health(gen._recipe_pool, pool_kwargs)
            if problems:
                print(f"  !!! fresh pool still shows issues: {problems}")
        except Exception:
            pass
        # Overwrite disk copy so --reuse-pool on next run picks up the fresh one.
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
    lr = float(args.lr)
    lr_min = float(args.lr_min) if args.lr_min else lr  # default: no decay
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.total_steps, eta_min=lr_min)

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
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.total_steps - global_step, eta_min=lr_min)
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
            # Fallback for old checkpoints: rebuild at correct position
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.total_steps, eta_min=lr_min,
                last_epoch=global_step)
        if ckpt.get("scaler") and args.precision == "fp16":
            scaler.load_state_dict(ckpt["scaler"])

    gen_bs = args.gen_batch if args.gen_batch > 0 else args.batch_size
    accum = args.grad_accum

    # -- Temporal warmup: freeze spatial weights, train only temporal params --
    # For MiniVAE3D, temporal params are (3,1,1) kernel convs and temporal attention.
    _temporal_param_names = set()
    for _mname, _mod in model.named_modules():
        # Temporal attention blocks
        if type(_mod).__name__ == "CausalTemporalAttention":
            for _pname, _ in _mod.named_parameters(recurse=True):
                _temporal_param_names.add(f"{_mname}.{_pname}" if _mname else _pname)
        # CausalConv3d with kernel_size (3,1,1) = temporal conv
        if type(_mod).__name__ == "CausalConv3d":
            k = _mod.conv.kernel_size
            if k[0] > 1 and k[1] == 1 and k[2] == 1:
                for _pname, _ in _mod.named_parameters(recurse=True):
                    _temporal_param_names.add(f"{_mname}.{_pname}" if _mname else _pname)

    _in_warmup = args.warmup_steps > 0 and global_step < args.warmup_steps
    if _in_warmup:
        for _n, _p in model.named_parameters():
            _p.requires_grad_(_n in _temporal_param_names)
        _warmup_params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(_warmup_params, lr=lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.total_steps, eta_min=lr_min,
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

    # Build a compact architecture tag from the explicit enc/dec widths.
    # "-e64x128x256x256" when symmetric (dec == reversed(enc)), else
    # "-e..-d..". Replaces the old "-b{base_ch}-m{mult}" naming which
    # was undefined when users pass --enc-ch/--dec-ch directly.
    def _arch_tag():
        e = "x".join(str(c) for c in enc_channels)
        if tuple(reversed(enc_channels)) == dec_channels:
            return f"-e{e}"
        d = "x".join(str(c) for c in dec_channels)
        return f"-e{e}-d{d}"

    def _ckpt_name(step):
        total_t = model.t_downscale
        total_s = model.s_downscale
        haar_tag = f"-h{haar_levels}" if haar_levels > 0 else ""
        fsq_tag = f"-fsq{fsq_stages}" if use_fsq else ""
        steps_k = f"{step // 1000}k" if step >= 1000 else str(step)
        return (f"vae3d-lc{latent_ch}{_arch_tag()}"
                f"-S{total_s}x-T{total_t}x{haar_tag}{fsq_tag}-{steps_k}.pt")

    # Glob pattern scoped to THIS run only
    _haar_tag = f"-h{haar_levels}" if haar_levels > 0 else ""
    _fsq_tag = f"-fsq{fsq_stages}" if use_fsq else ""
    _run_glob = (f"vae3d-lc{latent_ch}{_arch_tag()}"
                 f"-S{model.s_downscale}x-T{model.t_downscale}x{_haar_tag}{_fsq_tag}-*.pt")

    def _make_checkpoint():
        d = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "global_step": global_step,
            "config": {
                "model_class": "MiniVAE3D",
                "latent_channels": latent_ch,
                "image_channels": 3,
                "output_channels": 3,
                # New explicit schema. Readers should prefer these.
                "encoder_channels": ",".join(str(x) for x in enc_channels),
                "decoder_channels": ",".join(str(x) for x in dec_channels),
                # Legacy keys kept for old loaders. Only meaningful when
                # the schedule is a clean base*mult; otherwise they reflect
                # the enc list alone.
                "base_channels": int(enc_channels[0]),
                "channel_mult": ",".join(
                    str(c // enc_channels[0]) for c in enc_channels)
                    if all(c % enc_channels[0] == 0 for c in enc_channels)
                    else ",".join(str(c) for c in enc_channels),
                "num_res_blocks": num_res_blocks,
                "temporal_downsample": ",".join(str(s).lower() for s in temporal_down),
                "spatial_downsample": ",".join(str(s).lower() for s in spatial_down),
                "haar_levels": haar_levels,
                "fsq": use_fsq,
                "fsq_levels": ",".join(str(x) for x in fsq_levels),
                "fsq_stages": fsq_stages,
                # Arch extras — persist so resume picks them back up even
                # if CLI isn't given.
                "use_attention": bool(use_attention),
                "use_groupnorm": bool(use_groupnorm),
                "residual_shortcut": bool(residual_shortcut),
                "attn_heads": int(attn_heads),
                "gn_groups": int(gn_groups),
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
            opt = torch.optim.AdamW(model.parameters(), lr=lr,
                                    weight_decay=0.01)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.total_steps - global_step, eta_min=lr_min)

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
    p.add_argument("--latent-ch", type=int, default=16)
    # Preferred: pass explicit channel widths per level. enc is shallow->deep,
    # dec is deep->shallow (processing order; matches VideoTrainTab).
    p.add_argument("--enc-ch", default="",
                   help="Encoder channel widths per level, shallow->deep "
                        "(comma-separated ints, e.g. '64,128,256,256'). "
                        "If empty, falls back to --base-ch * --ch-mult.")
    p.add_argument("--dec-ch", default="",
                   help="Decoder channel widths per level, deep->shallow "
                        "(comma-separated ints, e.g. '256,256,128,64'). "
                        "Must satisfy dec[0]==enc[-1] and len(dec)==len(enc). "
                        "If empty, falls back to reversed(enc).")
    # Legacy (kept for back-compat with old checkpoints and scripts):
    p.add_argument("--base-ch", type=int, default=64,
                   help="[legacy] Base channel width; multiplied by --ch-mult "
                        "per level when --enc-ch/--dec-ch are not provided")
    p.add_argument("--ch-mult", default="1,2,4,4",
                   help="[legacy] Channel multipliers per level "
                        "(comma-separated ints)")
    p.add_argument("--num-res-blocks", type=int, default=2,
                   help="Factorized res blocks per resolution level")
    p.add_argument("--temporal-down", default="true,true,true,false",
                   help="Temporal 2x downsample per level (comma-separated bools)")
    p.add_argument("--spatial-down", default="true,true,true,true",
                   help="Spatial 2x downsample per level (comma-separated bools)")
    p.add_argument("--haar-levels", type=int, default=0,
                   help="3D Haar patcher levels (0=off, 1=8x, 2=64x)")
    p.add_argument("--fsq", action="store_true",
                   help="Enable residual FSQ quantizer in bottleneck")
    p.add_argument("--fsq-levels", default="8,8,8,5,5,5",
                   help="FSQ levels per dim (comma-separated ints)")
    p.add_argument("--fsq-stages", type=int, default=4,
                   help="Number of residual FSQ stages")
    # -- Arch extras (parity with 2D MiniVAE + DC-AE) --
    def _bool(s):
        return str(s).strip().lower() in ("1", "true", "yes", "y", "on")
    p.add_argument("--use-attention", type=_bool, default=True,
                   help="Spatial + temporal attention at deepest level "
                        "(default true; matches 2D MiniVAE convention)")
    p.add_argument("--use-groupnorm", type=_bool, default=True,
                   help="CausalGroupNorm inside res blocks + attention + "
                        "pre-output convs (default true)")
    p.add_argument("--residual-shortcut", type=_bool, default=False,
                   help="DC-AE-style spatial pixel-unshuffle shortcut "
                        "inside every HybridDownsample3d/HybridUpsample3d. "
                        "Non-parametric, spatial-only (preserves causality).")
    p.add_argument("--attn-heads", type=int, default=8,
                   help="num_heads for spatial attention; ch_mid must be "
                        "divisible by this")
    p.add_argument("--gn-groups", type=int, default=1,
                   help="num_groups for CausalGroupNorm. 1 = LayerNorm; "
                        "-1 = auto (matches 2D MiniVAE heuristic)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", default="1e-5")
    p.add_argument("--lr-min", default=None,
                   help="Cosine annealing floor LR (default: same as --lr, "
                        "i.e. no decay). Set e.g. 1e-7 to decay.")
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
    # -- Effect flags passed to build_motion_pool --
    # Defaults match the GUI's "everything on + random mix" behaviour so
    # training actually sees the effects we ship. random-mix drops each
    # flag per-recipe at its weighted rate so no single effect dominates.
    def _bflag(name, default, help_):
        p.add_argument(f"--{name}", dest=name.replace("-", "_"),
                       default=default, type=lambda s: s.lower() in
                       ("1", "true", "yes", "y", "on"),
                       help=f"{help_} (default {default})")
    p.add_argument("--random-mix", dest="random_mix", default=True,
                   type=lambda s: s.lower() in ("1", "true", "yes", "y", "on"),
                   help="Per-recipe random subset of enabled effects (default true)")
    _bflag("use-fluid",         True,  "Enable fluid flow")
    _bflag("use-ripple",        True,  "Enable ripple warp")
    _bflag("use-shake",         True,  "Enable camera shake")
    _bflag("use-kaleido",       True,  "Enable kaleidoscope")
    _bflag("fast-transform",    True,  "Enable fast transforms")
    _bflag("use-flash",         True,  "Enable flash frames")
    _bflag("use-palette-cycle", True,  "Enable palette cycling")
    _bflag("use-text",          True,  "Enable text overlays")
    _bflag("use-signage",       True,  "Enable signage overlays")
    _bflag("use-particles",     True,  "Enable particles")
    _bflag("use-raymarch",      True,  "Enable raymarch primitives")
    _bflag("sphere-dip",        True,  "Enable sphere-dip scene")
    _bflag("use-arcade",        True,  "Enable arcade scenes")
    _bflag("use-glitch",        True,  "Enable glitch bursts")
    _bflag("use-chromatic",     True,  "Enable chromatic aberration")
    _bflag("use-scanlines",     True,  "Enable scanlines")
    _bflag("use-fire",          True,  "Enable fire texture")
    _bflag("use-vortex",        True,  "Enable vortex swirl")
    _bflag("use-starfield",     True,  "Enable starfield zoom")
    _bflag("use-eq",            True,  "Enable EQ bars")
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="Steps to train only temporal params (3,1,1 convs + temporal attn) "
                        "before unfreezing spatial weights. Set 0 to disable.")
    p.add_argument("--resume", default=None)
    p.add_argument("--fresh-opt", action="store_true")
    p.add_argument("--logdir", default="synthyper_video3d_logs")
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
