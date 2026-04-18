#!/usr/bin/env python3
"""GUI tabs for ElasticVideoTokenizer training and inference.

- TokenizerTrainTab: configure + launch `python -m training.train_tokenizer`
- TokenizerInfTab:   load a trained tokenizer checkpoint and reconstruct
                     a clip from the generator at multiple keep budgets
                     (shows GT vs keep=32/64/128 side-by-side).
"""

import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog

import numpy as np
from PIL import Image, ImageTk

import torch

from gui.common import (
    BG, BG_PANEL, BG_INPUT, FG, FG_DIM, ACCENT, GREEN, BLUE, RED,
    FONT, FONT_TITLE, FONT_BOLD, FONT_SMALL,
    PROJECT_ROOT, VENV_PYTHON, BILINEAR,
    ProcRunner, make_btn, make_spin, make_float, make_slider, make_log,
    TOKENIZER_PRESETS, TOKENIZER_PRESET_NAMES, TOKENIZER_DEFAULT_PRESET,
    estimate_tokenizer_dims,
)


# -- Training tab --------------------------------------------------------------

class TokenizerTrainTab(tk.Frame):
    """Configure and launch tokenizer training as a subprocess.

    Layout mirrors VideoTrain3DTab: preset dropdown with live dim/param
    info label, then rows (arch / data / optim / logging / resume /
    preview), then buttons, then live preview video + log panel.
    Stop writes a .stop sentinel file so the training can save a final
    checkpoint before exiting.
    """

    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._video_frames = []
        self._video_playing = False
        self._video_idx = 0
        self._play_gen = 0
        self._last_mtime = None
        self.build()

    # ------------------------------------------------------------------ build
    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="Tokenizer Train (ElasticVideoTokenizer)",
                 bg=BG_PANEL, fg=FG, font=FONT_TITLE).pack(anchor="w")

        # -- Preset row with live info label (matches VideoTrain3DTab) --
        self._presets = TOKENIZER_PRESETS
        preset_row = tk.Frame(top, bg=BG_PANEL)
        preset_row.pack(fill="x", pady=(10, 0))
        tk.Label(preset_row, text="Preset:", bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(side="left", padx=(0, 4))
        self.preset_var = tk.StringVar(value=TOKENIZER_DEFAULT_PRESET)
        self._preset_menu = ttk.Combobox(
            preset_row, textvariable=self.preset_var,
            values=TOKENIZER_PRESET_NAMES, state="readonly", width=32,
            font=FONT_SMALL)
        self._preset_menu.pack(side="left", padx=(0, 8))
        self._preset_menu.bind("<<ComboboxSelected>>", self._apply_preset)
        self.dim_info_var = tk.StringVar(value="")
        tk.Label(preset_row, textvariable=self.dim_info_var,
                 bg=BG_PANEL, fg=ACCENT, font=FONT_SMALL,
                 anchor="w").pack(side="left", fill="x", expand=True)

        # -- Data source rows --
        # Three modes (set via radio): VAE, VAE+Flatten, Latent cache.
        self.source_mode_var = tk.StringVar(value="vae")
        mode_row = tk.Frame(top, bg=BG_PANEL)
        mode_row.pack(fill="x", pady=(10, 0))
        tk.Label(mode_row, text="Source:", bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(side="left", padx=(0, 8))
        for label, val in [("VAE only", "vae"),
                           ("VAE + Flatten", "vae_flatten"),
                           ("Latent cache", "cache")]:
            tk.Radiobutton(mode_row, text=label, variable=self.source_mode_var,
                           value=val, bg=BG_PANEL, fg=FG,
                           selectcolor=BG_INPUT, activebackground=BG_PANEL,
                           font=FONT_SMALL,
                           command=self._on_mode_change).pack(side="left",
                                                              padx=(0, 8))

        # VAE checkpoint row (always visible — used by vae and vae_flatten modes)
        vae_row = tk.Frame(top, bg=BG_PANEL)
        vae_row.pack(fill="x", pady=(4, 0))
        tk.Label(vae_row, text="VAE ckpt:", bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(side="left", padx=(0, 4))
        self.vae_ckpt_var = tk.StringVar(value="")
        tk.Entry(vae_row, textvariable=self.vae_ckpt_var, bg=BG_INPUT,
                 fg=FG, insertbackground=FG, font=FONT_SMALL, width=50).pack(
                     side="left", fill="x", expand=True, padx=(0, 4))
        make_btn(vae_row, "Browse", self._pick_vae, BLUE, 8).pack(side="left")

        # Flattener checkpoint row (used only in vae_flatten mode)
        self._flat_row = tk.Frame(top, bg=BG_PANEL)
        self._flat_row.pack(fill="x", pady=(4, 0))
        tk.Label(self._flat_row, text="Flatten ckpt:", bg=BG_PANEL,
                 fg=FG_DIM, font=FONT_SMALL).pack(side="left", padx=(0, 4))
        self.flatten_ckpt_var = tk.StringVar(value="")
        tk.Entry(self._flat_row, textvariable=self.flatten_ckpt_var,
                 bg=BG_INPUT, fg=FG, insertbackground=FG,
                 font=FONT_SMALL, width=50).pack(
                     side="left", fill="x", expand=True, padx=(0, 4))
        make_btn(self._flat_row, "Browse", self._pick_flatten,
                 BLUE, 8).pack(side="left")

        # Latent cache row (used only in cache mode)
        self._cache_row = tk.Frame(top, bg=BG_PANEL)
        self._cache_row.pack(fill="x", pady=(4, 0))
        tk.Label(self._cache_row, text="Latent cache dir:", bg=BG_PANEL,
                 fg=FG_DIM, font=FONT_SMALL).pack(side="left", padx=(0, 4))
        self.latent_cache_var = tk.StringVar(value="")
        tk.Entry(self._cache_row, textvariable=self.latent_cache_var,
                 bg=BG_INPUT, fg=FG, insertbackground=FG,
                 font=FONT_SMALL, width=42).pack(
                     side="left", fill="x", expand=True, padx=(0, 4))
        make_btn(self._cache_row, "Browse", self._pick_cache_dir,
                 BLUE, 8).pack(side="left", padx=(0, 4))
        f, self.latent_ch_var = make_spin(self._cache_row, "C_lat",
                                           default=4, width=4)
        f.pack(side="left", padx=(0, 4))
        f, self.latent_t_ds_var = make_spin(self._cache_row, "t_ds",
                                             default=4, width=4)
        f.pack(side="left", padx=(0, 4))
        f, self.latent_s_ds_var = make_spin(self._cache_row, "s_ds",
                                             default=16, width=4)
        f.pack(side="left")

        # Apply initial visibility based on the default mode
        self._on_mode_change()

        # -- Arch row (N_q / min_keep / dim / depth / heads / bottleneck) --
        arch_row = tk.Frame(top, bg=BG_PANEL)
        arch_row.pack(fill="x", pady=(10, 0))
        f, self.nq_var = make_spin(arch_row, "N queries", default=128, width=6)
        f.pack(side="left", padx=(0, 10))
        f, self.min_keep_var = make_spin(arch_row, "Min keep", default=32, width=6)
        f.pack(side="left", padx=(0, 10))
        f, self.dim_var = make_spin(arch_row, "dim", default=384, width=6)
        f.pack(side="left", padx=(0, 10))
        f, self.depth_var = make_spin(arch_row, "depth", default=6, width=4)
        f.pack(side="left", padx=(0, 10))
        f, self.heads_var = make_spin(arch_row, "heads", default=6, width=4)
        f.pack(side="left", padx=(0, 10))
        f, self.bn_var = make_spin(arch_row, "d_bottleneck", default=8, width=4)
        f.pack(side="left")

        # -- Data row (H / W / T / pool / bank / layers / disco) --
        data_row = tk.Frame(top, bg=BG_PANEL)
        data_row.pack(fill="x", pady=(5, 0))
        f, self.H_var = make_spin(data_row, "H", default=360, width=5)
        f.pack(side="left", padx=(0, 10))
        f, self.W_var = make_spin(data_row, "W", default=640, width=5)
        f.pack(side="left", padx=(0, 10))
        f, self.T_var = make_spin(data_row, "T (frames)", default=9, width=5)
        f.pack(side="left", padx=(0, 10))
        f, self.pool_var = make_spin(data_row, "Pool", default=200, width=6)
        f.pack(side="left", padx=(0, 10))
        f, self.bank_var = make_spin(data_row, "Bank", default=5000, width=6)
        f.pack(side="left", padx=(0, 10))
        f, self.layers_var = make_spin(data_row, "Layers", default=128, width=6)
        f.pack(side="left", padx=(0, 10))
        self.disco_var = tk.BooleanVar(value=False)
        tk.Checkbutton(data_row, text="Disco BG", variable=self.disco_var,
                       bg=BG_PANEL, fg=FG, selectcolor=BG_INPUT,
                       font=FONT_SMALL).pack(side="left")

        # -- Optim row (LR / batch / total steps / precision / wd / grad clip) --
        opt_row = tk.Frame(top, bg=BG_PANEL)
        opt_row.pack(fill="x", pady=(5, 0))
        f, self.lr_var = make_float(opt_row, "LR", "1e-5")
        f.pack(side="left", padx=(0, 10))
        f, self.batch_var = make_spin(opt_row, "Batch", default=4)
        f.pack(side="left", padx=(0, 10))
        f, self.steps_var = make_spin(opt_row, "Total steps", default=200000)
        f.pack(side="left", padx=(0, 10))
        f, self.prec_var = make_float(opt_row, "Precision", "bf16")
        f.pack(side="left", padx=(0, 10))
        f, self.wd_var = make_float(opt_row, "Weight decay", "1e-4")
        f.pack(side="left", padx=(0, 10))
        f, self.grad_clip_var = make_float(opt_row, "Grad clip", "1.0")
        f.pack(side="left")

        # -- Logging row --
        log_row = tk.Frame(top, bg=BG_PANEL)
        log_row.pack(fill="x", pady=(5, 0))
        f, self.log_every_var = make_spin(log_row, "Log every", default=10)
        f.pack(side="left", padx=(0, 10))
        f, self.save_every_var = make_spin(log_row, "Save every", default=5000)
        f.pack(side="left", padx=(0, 10))
        f, self.preview_every_var = make_spin(log_row, "Preview every", default=500)
        f.pack(side="left", padx=(0, 10))
        f, self.logdir_var = make_float(log_row, "Log dir",
                                        "synthyper_tokenizer_logs")
        f.pack(side="left", padx=(0, 10))
        f, self.keeps_var = make_float(log_row, "Preview keeps", "32,64,128",
                                        width=12)
        f.pack(side="left")

        # -- Resume row --
        resume_row = tk.Frame(top, bg=BG_PANEL)
        resume_row.pack(fill="x", pady=(5, 0))
        f, self.resume_var = make_float(resume_row, "Resume",
            os.path.join(PROJECT_ROOT, "synthyper_tokenizer_logs", "latest.pt"),
            width=50)
        f.pack(side="left")
        res_chk = tk.Frame(top, bg=BG_PANEL)
        res_chk.pack(fill="x", pady=(5, 0))
        self.use_latest_var = tk.BooleanVar(value=False)
        tk.Checkbutton(res_chk, text="Resume from latest.pt",
                       variable=self.use_latest_var, bg=BG_PANEL, fg=FG,
                       selectcolor=BG_INPUT, activebackground=BG_PANEL,
                       font=FONT, command=self._toggle_latest).pack(side="left")
        self.fresh_opt_var = tk.BooleanVar(value=True)
        tk.Checkbutton(res_chk, text="Fresh optimizer",
                       variable=self.fresh_opt_var, bg=BG_PANEL, fg=FG,
                       selectcolor=BG_INPUT, activebackground=BG_PANEL,
                       font=FONT).pack(side="left", padx=(10, 0))

        # -- Buttons row --
        btn_row = tk.Frame(top, bg=BG_PANEL)
        btn_row.pack(fill="x", pady=(10, 0))
        make_btn(btn_row, "Train", self.start, GREEN).pack(
            side="left", padx=(0, 5))
        make_btn(btn_row, "Stop (save)", self.stop_save, BLUE).pack(
            side="left", padx=(0, 5))
        make_btn(btn_row, "Kill", self.kill, RED).pack(
            side="left", padx=(0, 5))

        # -- Preview video + log --
        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(pady=5)
        self.log = make_log(self)
        self.log.pack(fill="both", expand=True, padx=5, pady=5)
        self.runner = ProcRunner(self.log)

        # Wire preset + live dim info
        self._check_preview()
        self._apply_preset()

    # ---------------------------------------------------------------- presets
    def _apply_preset(self, event=None):
        name = self.preset_var.get()
        preset = self._presets.get(name)
        if preset:
            self.nq_var.set(preset["n_queries"])
            self.min_keep_var.set(preset["min_keep"])
            self.dim_var.set(preset["dim"])
            self.depth_var.set(preset["depth"])
            self.heads_var.set(preset["heads"])
            self.bn_var.set(preset["d_bottleneck"])
        self._update_dim_info()
        # Wire live updates once
        if not getattr(self, "_dim_trace_wired", False):
            for v in (self.nq_var, self.min_keep_var, self.dim_var,
                      self.depth_var, self.heads_var, self.bn_var,
                      self.T_var, self.H_var, self.W_var):
                try:
                    v.trace_add("write", lambda *a: self._update_dim_info())
                except Exception:
                    pass
            self._dim_trace_wired = True

    def _update_dim_info(self):
        """Debounced recompute."""
        if getattr(self, "_dim_job", None) is not None:
            try:
                self.after_cancel(self._dim_job)
            except Exception:
                pass
        self._dim_job = self.after(250, self._do_update_dim_info)

    def _do_update_dim_info(self):
        self._dim_job = None
        try:
            # Derive the VAE's t_downscale from its checkpoint if available,
            # else assume 8 (Cosmos canonical). This label is a rough
            # projection; the actual run uses the real VAE.
            t_ds = 8
            vae_ckpt = self.vae_ckpt_var.get().strip()
            if vae_ckpt and os.path.isfile(vae_ckpt):
                try:
                    import torch as _t
                    c = _t.load(vae_ckpt, map_location="cpu", weights_only=False)
                    cfg = c.get("config", {})
                    tdn_str = cfg.get("temporal_downsample", "")
                    haar = int(cfg.get("haar_levels", 0))
                    if tdn_str:
                        tdn = [x.strip().lower() in ("true", "1", "yes")
                               for x in str(tdn_str).split(",")]
                        t_ds = (2 ** sum(tdn)) * (2 ** haar)
                    del c
                except Exception:
                    pass
            d = estimate_tokenizer_dims(
                n_queries=int(self.nq_var.get()),
                T=int(self.T_var.get()),
                t_downscale=t_ds,
                d_bottleneck=int(self.bn_var.get()),
                H=int(self.H_var.get()), W=int(self.W_var.get()))
            preset = self._presets.get(self.preset_var.get())
            p_hint = preset.get("params_hint") if preset else 0
            p_str = f"Params ~{p_hint/1e6:.1f}M  |  " if p_hint else ""
            self.dim_info_var.set(p_str + d["label"])
        except Exception as e:
            self.dim_info_var.set(f"(dim calc error: {e})")

    def _toggle_latest(self):
        if self.use_latest_var.get():
            logdir = self.logdir_var.get()
            self.resume_var.set(os.path.join(PROJECT_ROOT, logdir, "latest.pt"))
            self.fresh_opt_var.set(False)

    def _pick_vae(self):
        path = filedialog.askopenfilename(
            initialdir=os.path.join(PROJECT_ROOT, "synthyper_video_logs"),
            title="Select MiniVAE or MiniVAE3D checkpoint",
            filetypes=[("Checkpoint", "*.pt"), ("All", "*.*")])
        if path:
            self.vae_ckpt_var.set(path)
            self._update_dim_info()

    def _pick_flatten(self):
        path = filedialog.askopenfilename(
            initialdir=os.path.join(PROJECT_ROOT, "flatten_video_logs"),
            title="Select FlattenDeflatten checkpoint",
            filetypes=[("Checkpoint", "*.pt"), ("All", "*.*")])
        if path:
            self.flatten_ckpt_var.set(path)
            self._update_dim_info()

    def _pick_cache_dir(self):
        path = filedialog.askdirectory(
            initialdir=os.path.join(PROJECT_ROOT, "bank"),
            title="Select latent cache directory")
        if path:
            self.latent_cache_var.set(path)
            self._update_dim_info()

    def _on_mode_change(self):
        """Kept as a hook for future visibility toggling. All three rows
        are always visible — start() only reads the ones that match the
        selected mode, so the "extra" fields are just ignored. This
        avoids fragile pack_forget/re-pack dances for minimal benefit."""
        self._update_dim_info()

    # ----------------------------------------------------------------- launch
    def start(self):
        mode = self.source_mode_var.get()
        vae = self.vae_ckpt_var.get().strip()
        flat = self.flatten_ckpt_var.get().strip()
        cache = self.latent_cache_var.get().strip()

        # Validate per mode
        if mode in ("vae", "vae_flatten"):
            if not vae or not os.path.isfile(vae):
                self.log.insert("end",
                    f"[Need a valid VAE ckpt for mode '{mode}': {vae}]\n")
                self.log.see("end")
                return
            if mode == "vae_flatten":
                if not flat or not os.path.isfile(flat):
                    self.log.insert("end",
                        f"[Need a valid Flatten ckpt for VAE+Flatten mode: "
                        f"{flat}]\n")
                    self.log.see("end")
                    return
        elif mode == "cache":
            if not cache or not os.path.isdir(cache):
                self.log.insert("end",
                    f"[Need a valid latent cache directory: {cache}]\n")
                self.log.see("end")
                return

        cmd = [VENV_PYTHON, "-m", "training.train_tokenizer",
               "--n-queries", str(self.nq_var.get()),
               "--min-keep", str(self.min_keep_var.get()),
               "--dim", str(self.dim_var.get()),
               "--depth", str(self.depth_var.get()),
               "--heads", str(self.heads_var.get()),
               "--d-bottleneck", str(self.bn_var.get()),
               "--H", str(self.H_var.get()),
               "--W", str(self.W_var.get()),
               "--T", str(self.T_var.get()),
               "--pool-size", str(self.pool_var.get()),
               "--bank-size", str(self.bank_var.get()),
               "--n-layers", str(self.layers_var.get()),
               "--batch-size", str(self.batch_var.get()),
               "--lr", str(self.lr_var.get()),
               "--weight-decay", str(self.wd_var.get()),
               "--grad-clip", str(self.grad_clip_var.get()),
               "--total-steps", str(self.steps_var.get()),
               "--precision", self.prec_var.get(),
               "--log-every", str(self.log_every_var.get()),
               "--save-every", str(self.save_every_var.get()),
               "--preview-every", str(self.preview_every_var.get()),
               "--keeps", self.keeps_var.get(),
               "--logdir", self.logdir_var.get()]
        # Inject mode-specific data-source args
        if mode == "vae":
            cmd += ["--vae-ckpt", vae]
        elif mode == "vae_flatten":
            cmd += ["--vae-ckpt", vae, "--flatten-ckpt", flat]
        elif mode == "cache":
            cmd += ["--latent-cache", cache,
                    "--latent-ch", str(self.latent_ch_var.get()),
                    "--latent-t-ds", str(self.latent_t_ds_var.get()),
                    "--latent-s-ds", str(self.latent_s_ds_var.get())]
        if self.disco_var.get():
            cmd.append("--disco")
        resume = self.resume_var.get().strip()
        if self.use_latest_var.get() and resume:
            cmd += ["--resume", resume]
            if self.fresh_opt_var.get():
                cmd.append("--fresh-opt")
        self.runner.run(cmd, cwd=PROJECT_ROOT)

    def stop_save(self):
        """Writes a .stop sentinel in the logdir. The training script can
        poll for this to flush a final checkpoint before exiting cleanly.
        Mirrors VideoTrain3DTab.stop_save."""
        logdir = os.path.join(PROJECT_ROOT, self.logdir_var.get())
        os.makedirs(logdir, exist_ok=True)
        Path(os.path.join(logdir, ".stop")).touch()
        self.runner._append("[Stop file written]\n")

    def kill(self):
        self.runner.kill()

    # ---------------------------------------------------------------- preview
    def _check_preview(self):
        logdir = os.path.join(PROJECT_ROOT, self.logdir_var.get())
        preview = os.path.join(logdir, "preview_latest.mp4")
        if os.path.exists(preview):
            try:
                mtime = os.path.getmtime(preview)
                if self._last_mtime != mtime:
                    self._last_mtime = mtime
                    self._video_playing = False
                    probe = subprocess.run(
                        ["ffprobe", "-v", "quiet", "-show_entries",
                         "stream=width,height", "-of", "csv=p=0", preview],
                        capture_output=True, text=True)
                    parts = probe.stdout.strip().split(",")
                    if len(parts) >= 2:
                        try:
                            w, h = int(parts[0]), int(parts[1])
                        except (ValueError, TypeError):
                            w, h = 0, 0
                        if w > 0 and h > 0:
                            cmd = ["ffmpeg", "-v", "quiet", "-i", preview,
                                   "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "pipe:1"]
                            raw = subprocess.run(cmd, capture_output=True).stdout
                            fs = w * h * 3
                            n = len(raw) // fs
                            if n > 0:
                                scale = min(900 / w, 350 / h, 1.0)
                                dw = int(w * scale) if scale < 1 else w
                                dh = int(h * scale) if scale < 1 else h
                                self._video_frames = []
                                for fi in range(n):
                                    arr = np.frombuffer(
                                        raw[fi*fs:(fi+1)*fs],
                                        dtype=np.uint8).reshape(h, w, 3)
                                    pil = Image.fromarray(arr)
                                    if scale < 1:
                                        pil = pil.resize((dw, dh), BILINEAR)
                                    self._video_frames.append(ImageTk.PhotoImage(pil))
                                self._video_idx = 0
                                self._play_gen += 1
                                self._play_preview_loop(self._play_gen)
            except Exception:
                pass
        self.after(5000, self._check_preview)

    def _play_preview_loop(self, gen_id=None):
        if gen_id != self._play_gen or not self._video_frames:
            return
        self._video_idx = self._video_idx % len(self._video_frames)
        self.preview_label.config(image=self._video_frames[self._video_idx])
        self._video_idx += 1
        self.after(33, self._play_preview_loop, self._play_gen)


# -- Inference tab -------------------------------------------------------------

class TokenizerInfTab(tk.Frame):
    """Load a trained tokenizer and reconstruct clips at multiple keep budgets."""

    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.tok = None
        self.vae = None
        self.gen = None
        self._preview_photo = None
        self.build()

    def build(self):
        top = tk.Frame(self, bg=BG_PANEL, padx=10, pady=10)
        top.pack(fill="x", padx=5, pady=5)

        tk.Label(top, text="Tokenizer Inference",
                 bg=BG_PANEL, fg=FG, font=FONT_TITLE).pack(anchor="w")

        row = tk.Frame(top, bg=BG_PANEL)
        row.pack(fill="x", pady=(8, 0))
        tk.Label(row, text="Tokenizer ckpt:", bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(side="left", padx=(0, 4))
        self.tok_ckpt_var = tk.StringVar(value="")
        tk.Entry(row, textvariable=self.tok_ckpt_var, bg=BG_INPUT, fg=FG,
                 insertbackground=FG, font=FONT_SMALL, width=55).pack(
                     side="left", fill="x", expand=True, padx=(0, 4))
        make_btn(row, "Browse", self._pick_tok, BLUE, 8).pack(side="left")

        btn = tk.Frame(top, bg=BG_PANEL)
        btn.pack(fill="x", pady=(8, 0))
        make_btn(btn, "Load", self._load, GREEN).pack(side="left", padx=(0, 5))
        make_btn(btn, "Generate + Reconstruct", self._reconstruct,
                 ACCENT).pack(side="left")

        # Keep budgets to render
        kr = tk.Frame(top, bg=BG_PANEL)
        kr.pack(fill="x", pady=(8, 0))
        f, self.keeps_var = make_float(kr, "Keeps", "32,64,128", width=14)
        f.pack(side="left", padx=(0, 8))
        f, self.T_var = make_spin(kr, "T (frames)", default=17, width=5)
        f.pack(side="left")

        # Status + preview
        self.status_var = tk.StringVar(value="Not loaded")
        tk.Label(top, textvariable=self.status_var, bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_SMALL).pack(anchor="w", pady=(8, 0))

        self.preview_label = tk.Label(self, bg=BG)
        self.preview_label.pack(fill="both", expand=True, padx=5, pady=5)

    def _pick_tok(self):
        path = filedialog.askopenfilename(
            initialdir=os.path.join(PROJECT_ROOT, "synthyper_tokenizer_logs"),
            title="Select tokenizer checkpoint",
            filetypes=[("Checkpoint", "*.pt"), ("All", "*.*")])
        if path:
            self.tok_ckpt_var.set(path)

    def _load(self):
        """Load tokenizer + its referenced VAE in a worker thread so the
        GUI doesn't block on big .pt files."""
        path = self.tok_ckpt_var.get().strip()
        if not os.path.isfile(path):
            self.status_var.set(f"Not found: {path}")
            return
        self.status_var.set("Loading tokenizer + VAE...")
        self.update_idletasks()

        def _work():
            try:
                sys.path.insert(0, PROJECT_ROOT)
                from training.train_tokenizer import _reconstruct_vae_from_ckpt
                from core.tokenizer import ElasticVideoTokenizer
                from core.generator import VAEpp0rGenerator

                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                tok_args = ckpt["args"]
                vae_path = ckpt.get("vae_ckpt_path") \
                    or tok_args.get("vae_ckpt") if isinstance(tok_args, dict) else None
                if not vae_path or not os.path.isfile(vae_path):
                    self.after(0, lambda: self.status_var.set(
                        f"VAE path in ckpt is missing: {vae_path}"))
                    return

                vae, _ = _reconstruct_vae_from_ckpt(vae_path)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                vae = vae.to(device)

                def _g(k, d=None):
                    return tok_args.get(k, d) if isinstance(tok_args, dict) \
                        else getattr(tok_args, k, d)

                tok = ElasticVideoTokenizer(
                    vae=vae,
                    n_queries=int(_g("n_queries", 128)),
                    dim=int(_g("dim", 384)),
                    depth=int(_g("depth", 6)),
                    heads=int(_g("heads", 6)),
                    mlp_mult=int(_g("mlp_mult", 4)),
                    d_bottleneck=int(_g("d_bottleneck", 8)),
                    min_keep=int(_g("min_keep", 32)),
                ).to(device)
                tok.load_state_dict(ckpt["model"])
                tok.eval()

                gen = VAEpp0rGenerator(
                    height=int(_g("H", 360)), width=int(_g("W", 640)),
                    device=str(device),
                    bank_size=int(_g("bank_size", 5000)),
                    n_base_layers=int(_g("n_layers", 128)),
                )
                bank_dir = os.path.join(PROJECT_ROOT, "bank")
                if os.path.isdir(bank_dir) and any(
                        f.startswith("shapes_") for f in os.listdir(bank_dir)):
                    gen.setup_dynamic_bank(
                        bank_dir, working_size=int(_g("bank_size", 5000)))
                    gen.build_base_layers()
                else:
                    gen.build_banks()

                self.tok, self.vae, self.gen = tok, vae, gen
                self.after(0, lambda: self.status_var.set(
                    f"Loaded. N_q={tok.n_queries}, "
                    f"VAE t_ds={vae.t_downscale}, s_ds={vae.s_downscale}"))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda: self.status_var.set(f"Load failed: {e}"))

        threading.Thread(target=_work, daemon=True).start()

    def _reconstruct(self):
        if self.tok is None:
            self.status_var.set("Load a tokenizer checkpoint first")
            return
        try:
            keeps = sorted({min(int(k.strip()), self.tok.n_queries)
                            for k in self.keeps_var.get().split(",")})
        except Exception:
            self.status_var.set("Bad keeps list — use comma-separated ints")
            return
        T = int(self.T_var.get())
        self.status_var.set(
            f"Rendering clip, reconstructing at keeps={keeps}...")
        self.update_idletasks()

        def _work():
            try:
                device = next(self.tok.parameters()).device
                with torch.no_grad():
                    clip = self.gen.generate_sequence(1, T=T).to(device)
                    recons = self.tok.reconstruct(clip, keeps=keeps)

                H, W = self.gen.H, self.gen.W
                gap = 4
                n_cols = 1 + len(keeps)
                grid_w = W * n_cols + gap * (n_cols - 1)
                grid_h = H

                # Use middle frame for the preview stills
                t_mid = clip.shape[1] // 2
                gt = (clip[0, t_mid].permute(1, 2, 0).float().cpu()
                      .numpy() * 255).clip(0, 255).astype(np.uint8)
                tiles = [("GT", gt)]
                for k in keeps:
                    rc = (recons[k][0, t_mid].permute(1, 2, 0).float()
                          .cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
                    tiles.append((f"keep={k}", rc))

                gap_v = np.full((H, gap, 3), 14, dtype=np.uint8)
                row = tiles[0][1]
                for _, img in tiles[1:]:
                    row = np.concatenate([row, gap_v, img], axis=1)

                pil = Image.fromarray(row)
                scale = min(1200 / grid_w, 500 / grid_h, 1.0)
                if scale < 1.0:
                    pil = pil.resize(
                        (int(grid_w * scale), int(grid_h * scale)), BILINEAR)

                def _show():
                    self._preview_photo = ImageTk.PhotoImage(pil)
                    self.preview_label.config(image=self._preview_photo)
                    self.status_var.set(
                        f"Reconstructed T={T}, keeps={keeps}, "
                        f"mid-frame t={t_mid}")
                self.after(0, _show)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda: self.status_var.set(f"Failed: {e}"))

        threading.Thread(target=_work, daemon=True).start()
