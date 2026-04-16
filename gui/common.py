#!/usr/bin/env python3
"""Shared GUI infrastructure — theme, helpers, process runner, chunked inference."""

import os
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageTk

# Pillow 10+ removed Image.BILINEAR; use Image.Resampling.BILINEAR instead
BILINEAR = getattr(Image, "Resampling", Image).BILINEAR

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAEPP_ROOT = PROJECT_ROOT  # bank/log paths should be project-relative, not gui-relative
VENV_PYTHON = os.path.join(PROJECT_ROOT, "venv", "Scripts", "python.exe")
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = sys.executable

# -- Theme ---------------------------------------------------------------------
BG       = "#0e0e16"
BG_PANEL = "#161622"
BG_INPUT = "#1c1c2e"
BG_LOG   = "#0a0a12"
FG       = "#c8c8d0"
FG_DIM   = "#55556a"
RED      = "#e05555"
GREEN    = "#44cc77"
BLUE     = "#5588dd"
ACCENT   = "#44aacc"

FONT       = ("Consolas", 10)
FONT_BOLD  = ("Consolas", 10, "bold")
FONT_TITLE = ("Consolas", 14, "bold")
FONT_SMALL = ("Consolas", 9)


# -- Process runner ------------------------------------------------------------
class ProcRunner:
    def __init__(self, log_widget):
        self.log = log_widget
        self.proc = None
        self._thread = None

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def run(self, cmd, cwd=None):
        if self.running:
            self._append("[Already running]\n")
            return
        self.log.delete("1.0", tk.END)
        self._append(f"$ {' '.join(cmd)}\n\n")
        self._thread = threading.Thread(target=self._run, args=(cmd, cwd), daemon=True)
        self._thread.start()

    def _run(self, cmd, cwd):
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=cwd or PROJECT_ROOT, text=True, bufsize=1,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                if sys.platform == "win32" else 0)
            for line in self.proc.stdout:
                self._append(line)
            self.proc.wait()
            self._append(f"\n[exit {self.proc.returncode}]\n")
        except Exception as e:
            self._append(f"\nERROR: {e}\n")
        finally:
            if self.proc and self.proc.stdout:
                self.proc.stdout.close()
            self.proc = None

    def stop(self):
        proc = self.proc  # snapshot to avoid race with finally block
        if proc:
            try:
                if sys.platform == "win32":
                    os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
                else:
                    proc.terminate()
                self._append("\n[Stopping...]\n")
            except (ProcessLookupError, OSError):
                pass

    def kill(self):
        proc = self.proc  # snapshot to avoid race with finally block
        if proc:
            try:
                proc.kill()
                self._append("\n[Killed]\n")
            except (ProcessLookupError, OSError):
                pass

    def _append(self, text):
        self.log.after(0, self._do_append, text)

    def _do_append(self, text):
        self.log.insert(tk.END, text)
        self.log.see(tk.END)


# -- Helpers -------------------------------------------------------------------
def parse_arch_config(config):
    """Extract architecture params from checkpoint config dict.

    Returns (encoder_channels, decoder_channels) with proper types,
    falling back to defaults for old checkpoints.
    """
    enc_ch = config.get("encoder_channels", 64)
    if isinstance(enc_ch, str):
        if "," in enc_ch:
            enc_ch = tuple(int(x.strip()) for x in enc_ch.split(","))
        else:
            enc_ch = int(enc_ch)
    dec_ch = config.get("decoder_channels", (256, 128, 64))
    if isinstance(dec_ch, str):
        dec_ch = tuple(int(x.strip()) for x in dec_ch.split(","))
    return enc_ch, dec_ch


def run_with_log(tab, fn, on_done=None):
    """Run fn() on a background thread with stdout tee'd to tab.log.
    Polling starts on the main thread before the worker launches.
    on_done() is called on the main thread when fn() completes."""
    import queue as _q
    q = _q.Queue()
    original_stdout = sys.stdout
    done = [False]

    class _Tee:
        def write(self, s):
            if original_stdout:
                original_stdout.write(s)
            if s.strip():
                q.put(s.rstrip())
        def flush(self):
            if original_stdout:
                original_stdout.flush()

    def _poll():
        batch = []
        try:
            while True:
                batch.append(q.get_nowait())
        except _q.Empty:
            pass
        if batch:
            tab.log.insert(tk.END, "\n".join(batch) + "\n")
            tab.log.see(tk.END)
        if not done[0]:
            tab.after(50, _poll)
        else:
            # Final flush
            rest = []
            try:
                while True:
                    rest.append(q.get_nowait())
            except _q.Empty:
                pass
            if rest:
                tab.log.insert(tk.END, "\n".join(rest) + "\n")
                tab.log.see(tk.END)

    def _worker():
        sys.stdout = _Tee()
        try:
            fn()
        finally:
            sys.stdout = original_stdout
            done[0] = True
        if on_done:
            tab.after(0, on_done)

    # Start polling on main thread FIRST, then launch worker
    tab.after(50, _poll)
    threading.Thread(target=_worker, daemon=True).start()


def make_log(parent):
    return tk.Text(parent, bg=BG_LOG, fg=FG, font=FONT_SMALL,
                   insertbackground=FG, height=10, wrap=tk.WORD,
                   borderwidth=0, highlightthickness=0)

def make_btn(parent, text, command, color=ACCENT, width=14):
    return tk.Button(parent, text=text, command=command,
                     bg=color, fg="#ffffff", font=FONT_BOLD,
                     activebackground=color, activeforeground="#ffffff",
                     borderwidth=0, padx=8, pady=3, width=width)

def make_spin(parent, label, from_=0, to=99999, default=0, width=8):
    frame = tk.Frame(parent, bg=BG_PANEL)
    tk.Label(frame, text=label, bg=BG_PANEL, fg=FG_DIM,
             font=FONT_SMALL, anchor="w").pack(anchor="w")
    var = tk.IntVar(value=default)
    tk.Spinbox(frame, textvariable=var, from_=from_, to=to,
               bg=BG_INPUT, fg=FG, font=FONT, width=width,
               buttonbackground=BG_PANEL, borderwidth=0).pack(fill="x")
    return frame, var

def make_float(parent, label, default=0.0, width=8):
    frame = tk.Frame(parent, bg=BG_PANEL)
    tk.Label(frame, text=label, bg=BG_PANEL, fg=FG_DIM,
             font=FONT_SMALL, anchor="w").pack(anchor="w")
    var = tk.StringVar(value=str(default))
    tk.Entry(frame, textvariable=var, bg=BG_INPUT, fg=FG, font=FONT,
             width=width, borderwidth=0, insertbackground=FG).pack(fill="x")
    return frame, var

def make_slider(parent, label, from_=0.0, to=1.0, default=0.5, resolution=0.01):
    frame = tk.Frame(parent, bg=BG_PANEL)
    tk.Label(frame, text=label, bg=BG_PANEL, fg=FG_DIM,
             font=FONT_SMALL, anchor="w").pack(anchor="w")
    var = tk.DoubleVar(value=default)
    s = tk.Scale(frame, variable=var, from_=from_, to=to, resolution=resolution,
                 orient="horizontal", bg=BG_PANEL, fg=FG, troughcolor=BG_INPUT,
                 highlightthickness=0, font=FONT_SMALL, length=120,
                 activebackground=ACCENT, sliderrelief="flat")
    s.pack(fill="x")
    return frame, var


# -- Inference saving helper ---------------------------------------------------
def save_inference_output(images, logdir, prefix="inference", label="output"):
    """Save inference images to logdir/inference/ with timestamp."""
    inf_dir = os.path.join(logdir, "inference")
    os.makedirs(inf_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    for i, img_np in enumerate(images):
        if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
            img_np = img_np.transpose(1, 2, 0)
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        if img_np.shape[-1] == 1:
            img_np = img_np.squeeze(-1)
        pil = Image.fromarray(img_np)
        path = os.path.join(inf_dir, f"{prefix}_{label}_{ts}_{i:02d}.png")
        pil.save(path)
    return inf_dir


# -- Haar wavelet helpers -------------------------------------------------------

def haar_down(x):
    """2x spatial downscale: (B, C, H, W) -> (B, 4C, H/2, W/2). Lossless."""
    x = x.contiguous()
    a = x[:, :, 0::2, 0::2].contiguous()
    b = x[:, :, 0::2, 1::2].contiguous()
    c = x[:, :, 1::2, 0::2].contiguous()
    d = x[:, :, 1::2, 1::2].contiguous()
    ll = (a + b + c + d) * 0.5
    lh = (a - b + c - d) * 0.5
    hl = (a + b - c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return torch.cat([ll, lh, hl, hh], dim=1)

def haar_down_video(x, n):
    """Apply haar_down_n to a (B, T, C, H, W) tensor without looping over T."""
    if n == 0:
        return x
    B, T, C, H, W = x.shape
    flat = x.reshape(B * T, C, H, W)
    for _ in range(n):
        flat = haar_down(flat)
    _, C2, H2, W2 = flat.shape
    return flat.reshape(B, T, C2, H2, W2)

def haar_up(x):
    """2x spatial upscale: (B, 4C, H, W) -> (B, C, H*2, W*2). Lossless inverse."""
    x = x.contiguous()
    C = x.shape[1] // 4
    ll = x[:, 0*C:1*C].contiguous()
    lh = x[:, 1*C:2*C].contiguous()
    hl = x[:, 2*C:3*C].contiguous()
    hh = x[:, 3*C:4*C].contiguous()
    a = (ll + lh + hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    B, C2, H, W = ll.shape
    out = torch.zeros(B, C2, H * 2, W * 2, device=x.device, dtype=x.dtype)
    out[:, :, 0::2, 0::2] = a
    out[:, :, 0::2, 1::2] = b
    out[:, :, 1::2, 0::2] = c
    out[:, :, 1::2, 1::2] = d
    return out

def haar_up_video(x, n):
    """Apply haar_up_n to a (B, T, C, H, W) tensor without looping over T."""
    if n == 0:
        return x
    B, T, C, H, W = x.shape
    flat = x.reshape(B * T, C, H, W)
    for _ in range(n):
        flat = haar_up(flat)
    _, C2, H2, W2 = flat.shape
    return flat.reshape(B, T, C2, H2, W2)

def haar_down_n(x, n):
    for _ in range(n): x = haar_down(x)
    return x

def haar_up_n(x, n):
    for _ in range(n): x = haar_up(x)
    return x


# -- Chunked video inference helper --------------------------------------------
CHUNK_SIZE = 24

@torch.no_grad()
def chunked_vae_inference(model, x, chunk_size=CHUNK_SIZE, amp_dtype=torch.bfloat16):
    """Run VAE encode+decode in overlapping chunks with cross-fade blending.

    Uses large overlap (half the output length) and linearly blends in the
    overlap region to eliminate flicker at chunk boundaries.

    Returns:
        recon: aligned to input[trim:], so T_out = T_in - trim
        latent: concatenated latents from all chunks
    """
    T = x.shape[1]
    trim = getattr(model, 'frames_to_trim', 0)

    if T <= chunk_size:
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            recon, latent = model(x)
        return recon, latent

    output_per_chunk = chunk_size - trim
    # Overlap by half the output length for smooth blending
    overlap = max(output_per_chunk // 2, trim)
    stride = output_per_chunk - overlap
    target_len = T - trim

    # Collect all chunk outputs with their positions
    chunks_out = []  # list of (start_frame, recon_tensor)
    all_latent = []

    chunk_start = 0
    while chunk_start < T:
        chunk_end = min(chunk_start + chunk_size, T)
        chunk = x[:, chunk_start:chunk_end]

        if chunk.shape[1] < model.t_downscale:
            break

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            rc, lat = model(chunk)

        # This chunk's output covers input frames [chunk_start+trim .. chunk_end)
        # which maps to output frames [chunk_start .. chunk_start + rc.shape[1])
        # (relative to the trim-offset output timeline)
        out_start = chunk_start  # in output frame space (0-indexed from trim)
        chunks_out.append((out_start, rc.float().cpu()))
        all_latent.append(lat.float().cpu())
        del rc, lat
        torch.cuda.empty_cache()

        if chunk_end >= T:
            break

        chunk_start += stride

    # Blend overlapping chunks
    if not chunks_out:
        return torch.zeros(x.shape[0], 0, *x.shape[2:]), torch.cat(all_latent, dim=1)

    # Allocate output buffer
    B = x.shape[0]
    sample_rc = chunks_out[0][1]
    C_out = sample_rc.shape[2]
    H_out = sample_rc.shape[3]
    W_out = sample_rc.shape[4]

    recon = torch.zeros(B, target_len, C_out, H_out, W_out)
    weight = torch.zeros(B, target_len, 1, 1, 1)

    for out_start, rc in chunks_out:
        n_frames = rc.shape[1]
        # Ramp weights: fade in at start, fade out at end
        w = torch.ones(n_frames)
        fade_len = min(overlap, n_frames // 2)
        if fade_len > 0 and out_start > 0:
            # Fade in (not for first chunk)
            w[:fade_len] = torch.linspace(0, 1, fade_len)
        # Note: we don't fade out — the next chunk's fade-in handles blending

        for t in range(n_frames):
            out_t = out_start + t
            if 0 <= out_t < target_len:
                wt = w[t]
                recon[:, out_t] += rc[:, t] * wt
                weight[:, out_t] += wt

    # Normalize by total weight
    weight = weight.clamp(min=1e-6)
    recon = recon / weight

    return recon, torch.cat(all_latent, dim=1)


@torch.no_grad()
def chunked_flatten_inference(vae, bottleneck, x, chunk_size=CHUNK_SIZE,
                               amp_dtype=torch.bfloat16,
                               encode_fn=None, decode_fn=None):
    """Run VAE encode -> flatten/deflatten -> VAE decode in chunks with cross-fade.
    encode_fn/decode_fn override vae paths when provided (for FSQ)."""
    _encode = encode_fn or (lambda c: vae.encode_video(c))
    _decode = decode_fn or (lambda z: vae.decode_video(z))
    T = x.shape[1]
    if T <= chunk_size:
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
    output_per_chunk = chunk_size - trim
    overlap = max(output_per_chunk // 2, trim)
    stride = output_per_chunk - overlap
    target_len = T - trim

    chunks_vae = []
    chunks_flat = []
    all_lat = []

    chunk_start = 0
    while chunk_start < T:
        chunk_end = min(chunk_start + chunk_size, T)
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

        chunks_vae.append((chunk_start, rc_vae.float().cpu()))
        chunks_flat.append((chunk_start, rc_flat.float().cpu()))
        all_lat.append(lat.float().cpu())
        del rc_vae, rc_flat, lat, lat_r, lat_f
        torch.cuda.empty_cache()

        if chunk_end >= T:
            break
        chunk_start += stride

    def _blend(chunks_list, target_len, B):
        if not chunks_list:
            return torch.zeros(B, 0)
        sample = chunks_list[0][1]
        C_o, H_o, W_o = sample.shape[2], sample.shape[3], sample.shape[4]
        out = torch.zeros(B, target_len, C_o, H_o, W_o)
        wgt = torch.zeros(B, target_len, 1, 1, 1)
        for out_start, rc in chunks_list:
            n = rc.shape[1]
            w = torch.ones(n)
            fade = min(overlap, n // 2)
            if fade > 0 and out_start > 0:
                w[:fade] = torch.linspace(0, 1, fade)
            for t in range(n):
                ot = out_start + t
                if 0 <= ot < target_len:
                    out[:, ot] += rc[:, t] * w[t]
                    wgt[:, ot] += w[t]
        return out / wgt.clamp(min=1e-6)

    B = x.shape[0]
    recon_vae = _blend(chunks_vae, target_len, B)
    recon_flat = _blend(chunks_flat, target_len, B)

    return recon_vae, recon_flat, torch.cat(all_lat, dim=1)


@torch.no_grad()
def chunked_fsq_inference(vae, fsq_layer, pre_quant, post_quant, x,
                           chunk_size=CHUNK_SIZE, amp_dtype=torch.bfloat16):
    """Run VAE encode -> FSQ -> VAE decode in chunks with cross-fade.

    Returns:
        recon_fsq: FSQ reconstruction (aligned to input[trim:])
    """
    T = x.shape[1]
    if T <= chunk_size:
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            lat = vae.encode_video(x)
            B, Tp, C, Hl, Wl = lat.shape
            lat_flat = lat.reshape(B * Tp, C, Hl, Wl)
            z_proj = pre_quant(lat_flat)
            z_q, _ = fsq_layer(z_proj)
            lat_q = post_quant(z_q)
            lat_q = lat_q.reshape(B, Tp, C, Hl, Wl)
            recon_fsq = vae.decode_video(lat_q)
        return recon_fsq

    trim = getattr(vae, 'frames_to_trim', 0)
    output_per_chunk = chunk_size - trim
    overlap = max(output_per_chunk // 2, trim)
    stride = output_per_chunk - overlap
    target_len = T - trim

    chunks_out = []

    chunk_start = 0
    while chunk_start < T:
        chunk_end = min(chunk_start + chunk_size, T)
        chunk = x[:, chunk_start:chunk_end]

        if chunk.shape[1] < getattr(vae, 't_downscale', 1):
            break

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            lat = vae.encode_video(chunk)
            B, Tp, C, Hl, Wl = lat.shape
            lat_flat = lat.reshape(B * Tp, C, Hl, Wl)
            z_proj = pre_quant(lat_flat)
            z_q, _ = fsq_layer(z_proj)
            lat_q = post_quant(z_q)
            lat_q = lat_q.reshape(B, Tp, C, Hl, Wl)
            rc_fsq = vae.decode_video(lat_q)

        chunks_out.append((chunk_start, rc_fsq.float().cpu()))
        del rc_fsq, lat, lat_flat, z_proj, z_q, lat_q
        torch.cuda.empty_cache()

        if chunk_end >= T:
            break
        chunk_start += stride

    if not chunks_out:
        return torch.zeros(x.shape[0], 0, *x.shape[2:])

    sample = chunks_out[0][1]
    B = x.shape[0]
    C_o, H_o, W_o = sample.shape[2], sample.shape[3], sample.shape[4]
    out = torch.zeros(B, target_len, C_o, H_o, W_o)
    wgt = torch.zeros(B, target_len, 1, 1, 1)

    for out_start, rc in chunks_out:
        n = rc.shape[1]
        w = torch.ones(n)
        fade = min(overlap, n // 2)
        if fade > 0 and out_start > 0:
            w[:fade] = torch.linspace(0, 1, fade)
        for t in range(n):
            ot = out_start + t
            if 0 <= ot < target_len:
                out[:, ot] += rc[:, t] * w[t]
                wgt[:, ot] += w[t]

    return out / wgt.clamp(min=1e-6)
