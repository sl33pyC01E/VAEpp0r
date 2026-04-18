#!/usr/bin/env python3
"""Flatten/Deflatten experiment — TEMPORAL variant.

Freezes a trained temporal VAE encoder+decoder, inserts a 1D kernel-1
conv flatten/deflatten bottleneck in latent space (per-frame, no temporal
info), trains only the bottleneck to reconstruct through the frozen VAE.

Tests whether a flat 1D representation can preserve the spatial AND
temporal information after the VAE's TPool/TGrow compression.

Usage:
    python -m experiments.flatten_video \
        --vae-ckpt synthyper_video_logs/latest.pt
"""

import argparse
import math
import os
import pathlib
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import MiniVAE
from core.generator import VAEpp0rGenerator
from experiments.flatten import FlattenDeflatten


# -- Haar helpers (2D, per-frame on video tensors) ----------------------------
# Copy of the convention used by training/train_video.py. Needed here
# because the frozen VAE checkpoint expects post-Haar input (e.g. 48 ch
# for haar=4x) even though its config reports image_channels=3 (pre-Haar).
def _haar_down_2d(x):
    """(B, C, H, W) -> (B, 4C, H/2, W/2). Orthonormal 2D Haar wavelet."""
    a = x[:, :, 0::2, 0::2]
    b = x[:, :, 0::2, 1::2]
    c = x[:, :, 1::2, 0::2]
    d = x[:, :, 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a - b + c - d) * 0.5
    hl = (a + b - c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return torch.cat([ll, lh, hl, hh], dim=1)


def _haar_up_2d(x):
    """(B, 4C, H, W) -> (B, C, 2H, 2W). Inverse of _haar_down_2d."""
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


def _haar_down_video(x, n):
    """(B, T, C, H, W) -> (B, T, C*4^n, H/2^n, W/2^n). Per-frame."""
    if n <= 0:
        return x
    B, T, C, H, W = x.shape
    y = x.reshape(B * T, C, H, W)
    for _ in range(n):
        y = _haar_down_2d(y)
    return y.reshape(B, T, C * (4 ** n), H // (2 ** n), W // (2 ** n))


def _haar_up_video(x, n):
    """Inverse of _haar_down_video."""
    if n <= 0:
        return x
    B, T, C, H, W = x.shape
    y = x.reshape(B * T, C, H, W)
    for _ in range(n):
        y = _haar_up_2d(y)
    new_C = C // (4 ** n)
    return y.reshape(B, T, new_C, H * (2 ** n), W * (2 ** n))


# -- Chunked flatten inference (no-grad, for preview only) ---------------------

_CHUNK_SIZE = 24

@torch.no_grad()
def _chunked_flatten_inference(vae, bottleneck, x, amp_dtype=torch.bfloat16,
                                encode_fn=None, decode_fn=None):
    _encode = encode_fn or (lambda c: vae.encode_video(c))
    _decode = decode_fn or (lambda z: vae.decode_video(z))
    T = x.shape[1]
    if T <= _CHUNK_SIZE:
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            lat = _encode(x)
            recon_vae = _decode(lat)
            B, Tp, C, Hl, Wl = lat.shape
            lat_flat = lat.reshape(B * Tp, C, Hl, Wl)
            lat_recon, _ = bottleneck(lat_flat)
            lat_recon = lat_recon.reshape(B, Tp, C, Hl, Wl)
            recon_flat = _decode(lat_recon)
        return recon_vae, recon_flat, lat
    trim = getattr(vae, 'frames_to_trim', 0)
    output_per_chunk = _CHUNK_SIZE - trim
    target_len = T - trim
    all_vae, all_flat, all_lat = [], [], []
    chunk_start = 0
    collected = 0
    while chunk_start < T and collected < target_len:
        chunk_end = min(chunk_start + _CHUNK_SIZE, T)
        chunk = x[:, chunk_start:chunk_end]
        if chunk.shape[1] < getattr(vae, 't_downscale', 1):
            break
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            lat = _encode(chunk)
            rc_vae = _decode(lat)
            B, Tp, C, Hl, Wl = lat.shape
            lat_f = lat.reshape(B * Tp, C, Hl, Wl)
            lat_r, _ = bottleneck(lat_f)
            lat_r = lat_r.reshape(B, Tp, C, Hl, Wl)
            rc_flat = _decode(lat_r)
        need = target_len - collected
        keep = min(rc_vae.shape[1], rc_flat.shape[1], need)
        all_vae.append(rc_vae[:, :keep].float().cpu())
        all_flat.append(rc_flat[:, :keep].float().cpu())
        all_lat.append(lat.float().cpu())
        collected += keep
        del rc_vae, rc_flat, lat, lat_r, lat_f
        torch.cuda.empty_cache()
        if chunk_end >= T or collected >= target_len:
            break
        chunk_start += output_per_chunk
    return torch.cat(all_vae, dim=1), torch.cat(all_flat, dim=1), \
           torch.cat(all_lat, dim=1)


# -- Preview helpers -----------------------------------------------------------

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


# -- Preview (MP4 video) ------------------------------------------------------

@torch.no_grad()
def save_preview(vae, bottleneck, gen, logdir, step, device, amp_dtype, T=8,
                  preview_T=None, preview_image=None, preview_frame_skip=0,
                  encode_fn=None, decode_fn=None):
    """Save GT | VAE | Flatten reconstruction as MP4.
    Layout mirrors train_video: reference scaled to syn_w on top, 2 synth clips below.
    """
    try:
        vae.eval()
        bottleneck.eval()
        from PIL import Image as _PIL
        H, W = gen.H, gen.W
        sep_v = np.full((H, 4, 3), 14, dtype=np.uint8)
        gap_v = np.full((H, 2, 3), 14, dtype=np.uint8)
        trim  = getattr(vae, 'frames_to_trim', 0)
        pT    = preview_T if preview_T else T

        # cell_w = one clip: GT | sep | VAE | sep | Flat
        cell_w = W * 3 + 8
        # syn_w = two clips side by side (matches train_video pattern)
        syn_w  = cell_w * 2 + 2

        # -- Synthetic: 2 clips --
        # Sample two distinct pool recipes when possible so the preview
        # doesn't show two identical reconstructions. Fallback to fresh
        # generate_sequence if the pool isn't compatible with pT.
        with torch.random.fork_rng():
            torch.manual_seed(step + int(time.time()) % 100000)
            has_pool = (hasattr(gen, "_recipe_pool")
                        and gen._recipe_pool
                        and getattr(gen, "_motion_pool_T", None) == pT)
            if has_pool:
                import random as _rnd
                N = len(gen._recipe_pool)
                idxs = _rnd.sample(range(N), k=min(2, N))
                if len(idxs) < 2:
                    idxs = idxs + [idxs[0]]
                clips = torch.stack([
                    gen._render_recipe(gen._recipe_pool[i]) for i in idxs
                ], dim=0)
            else:
                # Last-resort live render; shares one effect roll across the
                # two clips so they may look similar — but avoids a crash
                # when preview T doesn't match the pool T.
                clips = gen.generate_sequence(2, T=pT)
        x = clips.to(device)
        recon_vae, recon_flat, _ = _chunked_flatten_inference(
            vae, bottleneck, x, amp_dtype=amp_dtype,
            encode_fn=encode_fn, decode_fn=decode_fn)
        T_out    = min(recon_vae.shape[1], recon_flat.shape[1])
        T_show   = T_out
        syn_gt   = x[:, trim:trim + T_out, :3, :H, :W].float().cpu().numpy()
        syn_vae  = recon_vae[:, :T_out, :3, :H, :W].clamp(0, 1).float().cpu().numpy()
        syn_flat = recon_flat[:, :T_out, :3, :H, :W].clamp(0, 1).float().cpu().numpy()
        del recon_vae, recon_flat, x

        # -- Reference clip --
        ref_gt = ref_vae = ref_flat = None
        if preview_image and os.path.exists(preview_image):
            ref_frames = _decode_video_frames(preview_image, preview_frame_skip or 0, pT, W, H)
            if len(ref_frames) < 2:
                print(f"  ref: only {len(ref_frames)} frames decoded", flush=True)
            else:
                ref_arr = np.stack(ref_frames).astype(np.float32) / 255.0
                ref_t = torch.from_numpy(ref_arr).permute(0, 3, 1, 2).unsqueeze(0).to(device)
                r_vae, r_flat, _ = _chunked_flatten_inference(
                    vae, bottleneck, ref_t, amp_dtype=amp_dtype,
                    encode_fn=encode_fn, decode_fn=decode_fn)
                T_r    = min(r_vae.shape[1], r_flat.shape[1])
                ref_gt   = ref_t[0, trim:trim + T_r, :3, :H, :W].float().cpu().numpy()
                ref_vae  = r_vae[0, :T_r, :3, :H, :W].clamp(0, 1).float().cpu().numpy()
                ref_flat = r_flat[0, :T_r, :3, :H, :W].clamp(0, 1).float().cpu().numpy()
                T_show = min(T_show, T_r)
                del ref_t, r_vae, r_flat

        bottleneck.train()

        has_ref = ref_gt is not None
        scale   = syn_w / cell_w
        ref_h   = int(H * scale)
        frame_w = syn_w
        frame_h = (ref_h + 6 + H) if has_ref else H

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
                        g  = (ref_gt[t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                        v  = (ref_vae[t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                        f_ = (ref_flat[t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                        ref_cell   = np.concatenate([g, sep_v, v, sep_v, f_], axis=1)
                        ref_scaled = np.array(_PIL.fromarray(ref_cell).resize(
                            (syn_w, ref_h), _PIL.BILINEAR))
                        rows.append(ref_scaled)
                        rows.append(np.full((6, syn_w, 3), 14, dtype=np.uint8))
                    g0  = (syn_gt[0, t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                    v0  = (syn_vae[0, t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                    f0  = (syn_flat[0, t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                    g1  = (syn_gt[1, t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                    v1  = (syn_vae[1, t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                    f1  = (syn_flat[1, t].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                    synth_row = np.concatenate([
                        g0, sep_v, v0, sep_v, f0, gap_v, g1, sep_v, v1, sep_v, f1
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
        print(f"  preview: {stepped} (GT|VAE|Flat, {T_show} frames{ref_note})", flush=True)

    except Exception as e:
        import traceback
        print(f"  preview failed: {e}", flush=True)
        traceback.print_exc()
        bottleneck.train()


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

    # -- Load frozen temporal VAE --
    print(f"Loading temporal VAE from {args.vae_ckpt}...")
    ckpt = torch.load(args.vae_ckpt, map_location="cpu", weights_only=False)
    vae_config = ckpt.get("config", {})
    # Haar rounds: the 2D MiniVAE training script applies Haar_down
    # _before_ encoding and Haar_up _after_ decoding. The checkpoint's
    # `image_channels` key stores the pre-Haar channel count (usually 3)
    # but the actual first conv was built to accept the post-Haar tensor
    # (3 * 4^haar_rounds channels). Match that here or the shape-matched
    # weight loader silently drops encoder.0.weight and the VAE stays at
    # random init — which is exactly how previous runs produced solid-
    # color garbage reconstructions in preview.
    haar_mode = vae_config.get("haar", "none")
    if haar_mode is True: haar_mode = "2x"
    elif not haar_mode or haar_mode is False: haar_mode = "none"
    haar_rounds = {"none": 0, "2x": 1, "4x": 2}.get(haar_mode, 0)
    ch_raw = int(vae_config.get("image_channels", 3))
    ch = ch_raw * (4 ** haar_rounds)   # post-Haar = what the conv sees
    lat_ch = vae_config.get("latent_channels", 32)
    print(f"  haar_mode={haar_mode} (rounds={haar_rounds})  "
          f"image_channels: {ch_raw} pre-Haar -> {ch} post-Haar")
    enc_ch_raw = vae_config.get("encoder_channels", 64)
    if isinstance(enc_ch_raw, str):
        enc_ch = tuple(int(x) for x in enc_ch_raw.split(","))
    elif isinstance(enc_ch_raw, (list, tuple)):
        enc_ch = tuple(int(x) for x in enc_ch_raw)
    else:
        enc_ch = int(enc_ch_raw)

    dec_ch_str = vae_config.get("decoder_channels", "256,128,64")
    if isinstance(dec_ch_str, str):
        dec_ch = tuple(int(x) for x in dec_ch_str.split(","))
    elif isinstance(dec_ch_str, (list, tuple)):
        dec_ch = tuple(dec_ch_str)
    else:
        dec_ch = (256, 128, 64)

    # encoder_time_downscale / decoder_time_upscale MUST come from the
    # checkpoint config. Guessing here was the whole problem: the
    # shape-matched weight loader silently drops TPool/TGrow weights
    # when the stride is wrong, so the VAE runs with random time layers
    # and the reconstruction is garbage. Fail loud instead.
    enc_t_raw = vae_config.get("encoder_time_downscale")
    dec_t_raw = vae_config.get("decoder_time_upscale")
    if enc_t_raw is None or dec_t_raw is None:
        raise SystemExit(
            f"VAE checkpoint at {args.vae_ckpt} is missing "
            f"encoder_time_downscale / decoder_time_upscale in its "
            f"config. Without these we can't build the MiniVAE with the "
            f"right TPool/TGrow strides, and the shape-match loader "
            f"will silently skip those weights -> random time layers "
            f"-> garbage reconstruction. Re-save the VAE checkpoint "
            f"with these keys in its config, or pass them explicitly.")
    if isinstance(enc_t_raw, str):
        enc_t = tuple(x.strip().lower() in ("true", "1", "yes") for x in enc_t_raw.split(","))
    else:
        enc_t = tuple(bool(x) for x in enc_t_raw)
    if isinstance(dec_t_raw, str):
        dec_t = tuple(x.strip().lower() in ("true", "1", "yes") for x in dec_t_raw.split(","))
    else:
        dec_t = tuple(bool(x) for x in dec_t_raw)

    residual_shortcut = vae_config.get("residual_shortcut", False)
    use_attention = vae_config.get("use_attention", False)
    use_groupnorm = vae_config.get("use_groupnorm", False)

    # Spatial downscale/upscale schedule. This was being left off the
    # constructor, which meant MiniVAE defaulted to all-True (downsample
    # every stage) and built a VAE with 2x more spatial compression than
    # the checkpoint — causing 27 weights to silently shape-mismatch and
    # the VAE to run partly at random init.
    enc_s_raw = vae_config.get("encoder_spatial_downscale")
    dec_s_raw = vae_config.get("decoder_spatial_upscale")
    if enc_s_raw is not None:
        if isinstance(enc_s_raw, str):
            enc_s = tuple(x.strip().lower() in ("true","1","yes") for x in enc_s_raw.split(","))
        else:
            enc_s = tuple(bool(x) for x in enc_s_raw)
    else:
        enc_s = tuple([True] * len(enc_ch))
    if dec_s_raw is not None:
        if isinstance(dec_s_raw, str):
            dec_s = tuple(x.strip().lower() in ("true","1","yes") for x in dec_s_raw.split(","))
        else:
            dec_s = tuple(bool(x) for x in dec_s_raw)
    else:
        dec_s = tuple([True] * len(dec_ch))

    vae = MiniVAE(
        latent_channels=lat_ch, image_channels=ch, output_channels=ch,
        encoder_channels=enc_ch, decoder_channels=dec_ch,
        encoder_time_downscale=enc_t,
        decoder_time_upscale=dec_t,
        encoder_spatial_downscale=enc_s,
        decoder_spatial_upscale=dec_s,
        residual_shortcut=residual_shortcut,
        use_attention=use_attention,
        use_groupnorm=use_groupnorm,
    ).to(device)

    src_sd = ckpt["model"] if "model" in ckpt else ckpt
    target_sd = vae.state_dict()
    loaded = 0
    skipped = []
    for k, v in src_sd.items():
        if k in target_sd and v.shape == target_sd[k].shape:
            target_sd[k] = v
            loaded += 1
        elif k in target_sd:
            skipped.append((k, list(v.shape), list(target_sd[k].shape)))
    vae.load_state_dict(target_sd)
    vae.eval()
    vae.requires_grad_(False)
    print(f"  VAE: {ch}ch, {lat_ch} latent, frozen "
          f"({loaded}/{len(src_sd)} weights)")
    if skipped:
        # Fail loud instead of silently running with random-init layers.
        # This is what produced the green-mush reconstruction before.
        print(f"  ERROR: {len(skipped)} VAE weights have shape mismatch "
              f"against the built model. Cannot proceed — the VAE would "
              f"run with {len(skipped)} random-init layers and produce "
              f"garbage.")
        for k, src_shape, tgt_shape in skipped[:10]:
            print(f"    {k}: ckpt={src_shape} built={tgt_shape}")
        raise SystemExit(1)

    # Load FSQ projections if present in checkpoint
    fsq_cfg = vae_config.get("fsq", {})
    fsq_layer = None
    pre_quant = None
    post_quant = None
    if fsq_cfg and fsq_cfg.get("levels"):
        from core.fsq import FSQ
        levels = fsq_cfg["levels"]
        if isinstance(levels, str):
            levels = [int(x) for x in levels.split(",")]
        fsq_dims = len(levels)
        fsq_layer = FSQ(levels=levels).to(device)
        pre_quant = nn.Conv2d(lat_ch, fsq_dims, 1).to(device)
        post_quant = nn.Conv2d(fsq_dims, lat_ch, 1).to(device)
        if ckpt.get("pre_quant"):
            pre_quant.load_state_dict(ckpt["pre_quant"])
        if ckpt.get("post_quant"):
            post_quant.load_state_dict(ckpt["post_quant"])
        pre_quant.eval()
        post_quant.eval()
        pre_quant.requires_grad_(False)
        post_quant.requires_grad_(False)
        print(f"  FSQ: levels={levels}, {fsq_dims} dims, frozen")

    has_fsq = fsq_layer is not None

    def encode_latent(x_in):
        """Encode a 3-channel video clip through the VAE.

        If the checkpoint was trained with Haar 2x/4x (haar_rounds > 0),
        we apply the per-frame Haar-down BEFORE the VAE so the VAE sees
        the same post-Haar tensor its weights expect. FSQ projection
        runs inside this function too.

        Contract: callers MUST pass pre-Haar (3-channel at full spatial
        resolution) video. The function performs the Haar internally.
        """
        if haar_rounds > 0:
            x_in = _haar_down_video(x_in, haar_rounds)
        lat = vae.encode_video(x_in)
        if has_fsq:
            B, Tp, C, Hl, Wl = lat.shape
            lf = lat.reshape(B * Tp, C, Hl, Wl)
            z_proj = pre_quant(lf)
            z_q, _ = fsq_layer(z_proj)
            lat = post_quant(z_q).reshape(B, Tp, C, Hl, Wl)
        return lat

    def decode_latent(lat_in):
        """Decode through VAE and Haar-up back to 3-channel pixels if
        the VAE was trained with Haar."""
        out = vae.decode_video(lat_in)  # (B, T, ch_post_haar, H, W)
        if haar_rounds > 0:
            out = _haar_up_video(out, haar_rounds)  # -> 3 channels
        return out

    # Probe latent spatial dims.
    # Feed the SAME shape the real training clips have: 3-channel video
    # at full resolution. encode_latent() does the Haar internally, so
    # we must not pre-Haar the probe or the latent spatial dims come
    # out wrong (which then mis-sizes the FlattenDeflatten bottleneck
    # and crashes at the first real training step with a shape reshape
    # error).
    with torch.no_grad():
        dummy = torch.randn(1, args.T, ch_raw, args.H, args.W, device=device)
        lat_dummy = encode_latent(dummy)
        _, Tp, lat_C, lat_H, lat_W = lat_dummy.shape
    print(f"  Latent: T'={Tp} (from T={args.T}), spatial ({lat_C}, {lat_H}, {lat_W}) "
          f"= {lat_C * lat_H * lat_W} values/frame")
    print(f"  Bottleneck: {args.bottleneck_ch}ch × {lat_H * lat_W} positions "
          f"= {args.bottleneck_ch * lat_H * lat_W} flat values/frame")
    print(f"  Compression: {lat_C / args.bottleneck_ch:.1f}:1 channel, "
          f"total {lat_C * lat_H * lat_W} -> {args.bottleneck_ch * lat_H * lat_W}")
    del dummy, lat_dummy

    # -- Flatten/Deflatten bottleneck --
    bottleneck = FlattenDeflatten(
        latent_channels=lat_C,
        bottleneck_channels=args.bottleneck_ch,
        spatial_h=lat_H, spatial_w=lat_W,
        walk_order=args.walk_order,
    ).to(device)
    print(f"  Bottleneck: {bottleneck.param_count():,} params, "
          f"walk={args.walk_order}")

    # -- Generator with motion pool --
    gen = VAEpp0rGenerator(
        height=args.H, width=args.W, device=str(device),
        bank_size=args.bank_size, n_base_layers=args.n_layers,
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

    # -- Effect flags for the motion pool (mirror train_video / train_video3d).
    # Earlier runs built the pool with zero flags, which meant recipes had
    # none of the procedural effects. All default True; random_mix drops
    # each effect per-recipe so the pool has variety.
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
        """Same stale-pool detector as train_video3d."""
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

    # Motion pool. Default: always rebuild (matches train_video3d policy).
    # Pass --reuse-pool to opt into reusing an existing pool file.
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
    print(f"  Generator: bank={gen.bank_size}, pool={gen.motion_pool_stats()}, "
          f"disco={getattr(gen, 'disco_quadrant', False)}")

    # -- Optimizer (bottleneck only) --
    opt = torch.optim.AdamW(bottleneck.parameters(), lr=float(args.lr),
                            weight_decay=0.01)

    # -- Resume --
    start_step = 0
    if args.resume:
        rk = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "bottleneck" in rk:
            bottleneck.load_state_dict(rk["bottleneck"])
            start_step = rk.get("step", 0)
            if rk.get("optimizer") and not args.fresh_opt:
                try:
                    opt.load_state_dict(rk["optimizer"])
                except Exception:
                    print("  Fresh optimizer (mismatch)")
            print(f"  Resumed bottleneck from {args.resume} at step {start_step}")
        else:
            print(f"  WARNING: no 'bottleneck' key in {args.resume}")

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}[args.precision]
    scaler = torch.amp.GradScaler("cuda",
                                   enabled=(args.precision == "fp16"))

    gen_bs = args.gen_batch if args.gen_batch > 0 else args.batch_size
    accum = args.grad_accum

    print(f"Steps: {args.total_steps}, LR: {args.lr}, Batch: {args.batch_size}, "
          f"T: {args.T}"
          f"{f', accum={accum}' if accum > 1 else ''}"
          f"{f', gen-batch={gen_bs}' if gen_bs != args.batch_size else ''}")
    print(flush=True)

    def _make_checkpoint():
        return {
            "model": vae.state_dict(),
            "bottleneck": bottleneck.state_dict(),
            "optimizer": opt.state_dict(),
            "step": step,
            "config": {
                # VAE arch (so inference tab can build MiniVAE without the original VAE ckpt)
                "image_channels": ch,
                "latent_channels": lat_ch,
                "encoder_channels": ",".join(str(x) for x in enc_ch) if isinstance(enc_ch, tuple) else enc_ch,
                "decoder_channels": ",".join(str(x) for x in dec_ch),
                "encoder_time_downscale": ",".join(str(x) for x in enc_t),
                "decoder_time_upscale": ",".join(str(x) for x in dec_t),
                "residual_shortcut": residual_shortcut,
                "use_attention": use_attention,
                "use_groupnorm": use_groupnorm,
                # Bottleneck
                "bottleneck_channels": args.bottleneck_ch,
                "spatial_h": lat_H,
                "spatial_w": lat_W,
                "walk_order": args.walk_order,
                "temporal": True,
                "T": args.T,
            },
        }

    # Initial preview
    save_preview(vae, bottleneck, gen, str(logdir), start_step, device,
                 amp_dtype, args.T,
                 preview_T=getattr(args, 'preview_T', None),
                 preview_image=getattr(args, 'preview_image', None),
                 preview_frame_skip=getattr(args, 'preview_frame_skip', 0),
                 encode_fn=encode_latent, decode_fn=decode_latent)

    # -- Loop --
    t0 = time.time()

    stop_file = logdir / ".stop"
    if stop_file.exists():
        stop_file.unlink()

    step = start_step
    while step < args.total_steps:
        step += 1
        if _stop_requested or stop_file.exists():
            if stop_file.exists():
                stop_file.unlink()
            break

        bottleneck.train()
        opt.zero_grad(set_to_none=True)

        for _ai in range(accum):
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
            x = clips.to(device)

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                # Encode through frozen VAE (+FSQ if present)
                with torch.no_grad():
                    latent = encode_latent(x)  # (B, T', C, H, W)

                B, Tp, C, Hl, Wl = latent.shape

                # Flatten + deflatten per-frame
                lat_flat = latent.reshape(B * Tp, C, Hl, Wl)
                lat_recon_flat, flat = bottleneck(lat_flat)
                lat_recon = lat_recon_flat.reshape(B, Tp, C, Hl, Wl)

                # Latent reconstruction loss (per-frame)
                lat_loss = F.mse_loss(lat_recon, latent)

                # Decode through frozen VAE for pixel loss
                with torch.no_grad():
                    gt_recon = decode_latent(latent)
                flat_recon = decode_latent(lat_recon)

                T_gt = gt_recon.shape[1]
                T_fl = flat_recon.shape[1]
                T_match = min(T_gt, T_fl)
                pixel_loss = F.mse_loss(
                    flat_recon[:, T_fl - T_match:],
                    gt_recon[:, T_gt - T_match:])

                # Temporal consistency in latent space
                temp_loss = torch.tensor(0.0, device=device)
                if Tp >= 2:
                    gt_diff = latent[:, 1:] - latent[:, :-1]
                    rc_diff = lat_recon[:, 1:] - lat_recon[:, :-1]
                    temp_loss = F.l1_loss(rc_diff, gt_diff)

                total = (args.w_latent * lat_loss
                         + args.w_pixel * pixel_loss
                         + args.w_temporal * temp_loss)

            scaler.scale(total / accum).backward()
            del latent, lat_recon, flat, gt_recon, flat_recon, clips, x

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(bottleneck.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if step % args.log_every == 0:
            el = time.time() - t0
            steps_run = step - start_step
            sps = steps_run / max(el, 1)
            eta = (args.total_steps - step) / max(sps, 1e-6)
            eta_str = f"{eta/60:.0f}m" if eta < 3600 else f"{eta/3600:.1f}h"
            parts = [f"lat={lat_loss.item():.6f}",
                     f"pix={pixel_loss.item():.6f}"]
            if Tp >= 2:
                parts.append(f"temp={temp_loss.item():.6f}")
            print(f"[{step}/{args.total_steps}] {' '.join(parts)} "
                  f"({sps:.1f} step/s, {eta_str} left)", flush=True)

        if step % args.preview_every == 0:
            save_preview(vae, bottleneck, gen, str(logdir), step,
                         device, amp_dtype, args.T,
                         preview_T=getattr(args, 'preview_T', None),
                         preview_image=getattr(args, 'preview_image', None),
                         preview_frame_skip=getattr(args, 'preview_frame_skip', 0),
                         encode_fn=encode_latent, decode_fn=decode_latent)
            # Preview peaks at several GB because it encodes+decodes two
            # full clips plus a reference video through the VAE. Without
            # explicit cache release the allocator's high-water mark
            # stays resident for the rest of training. ~5-15 GB recovery.
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if step % args.save_every == 0:
            d = _make_checkpoint()
            torch.save(d, logdir / f"step_{step:06d}.pt")
            torch.save(d, logdir / "latest.pt")
            print(f"  saved step {step}", flush=True)

            # Keep last 10
            ckpts = sorted(logdir.glob("step_*.pt"),
                           key=lambda x: x.stat().st_mtime)
            while len(ckpts) > 10:
                ckpts.pop(0).unlink()

    # Save on exit
    if step > start_step:
        d = _make_checkpoint()
        torch.save(d, logdir / f"step_{step:06d}.pt")
        torch.save(d, logdir / "latest.pt")
        print(f"  saved step {step}", flush=True)

    print(f"\nDone. {step - start_step} steps in {(time.time() - t0) / 60:.1f}min")


def main():
    p = argparse.ArgumentParser(
        description="Flatten/Deflatten experiment — temporal")
    p.add_argument("--vae-ckpt", required=True,
                   help="Path to trained temporal VAE checkpoint")
    p.add_argument("--H", type=int, default=360)
    p.add_argument("--W", type=int, default=640)
    p.add_argument("--T", type=int, default=24)
    p.add_argument("--bottleneck-ch", type=int, default=6,
                   help="Channels after flatten (6 = ~5:1 compression)")
    p.add_argument("--walk-order", default="raster",
                   choices=["raster", "hilbert", "morton"])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", default="1e-3")
    p.add_argument("--total-steps", type=int, default=10000)
    p.add_argument("--w-latent", type=float, default=1.0)
    p.add_argument("--w-pixel", type=float, default=0.5)
    p.add_argument("--w-temporal", type=float, default=1.0)
    p.add_argument("--precision", default="bf16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps (effective batch = batch-size * grad-accum)")
    p.add_argument("--gen-batch", type=int, default=0,
                   help="Generator batch size (0 = same as batch-size). "
                        "Lower to reduce generator VRAM.")
    p.add_argument("--logdir", default="flatten_video_logs")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--preview-every", type=int, default=100)
    p.add_argument("--preview-image", default=None,
                   help="Path to reference video (mp4) for tracking progress")
    p.add_argument("--preview-frame-skip", type=int, default=0,
                   help="Number of frames to skip into the reference video")
    p.add_argument("--preview-T", type=int, default=None,
                   help="Preview clip length in frames (default: same as --T)")
    p.add_argument("--resume", default=None,
                   help="Resume from bottleneck checkpoint")
    p.add_argument("--bank-size", type=int, default=5000)
    p.add_argument("--n-layers", type=int, default=128)
    p.add_argument("--pool-size", type=int, default=200)
    p.add_argument("--disco", action="store_true",
                   help="Enable disco quadrant mode")
    p.add_argument("--fresh-opt", action="store_true",
                   help="Fresh optimizer on resume")
    # -- Pool / effect flags (mirror train_video / train_video3d) --
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
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
