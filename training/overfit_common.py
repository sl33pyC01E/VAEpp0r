"""Shared helpers for the "overfit on a single video" mode across training
scripts. One module, one place to fix bugs, every training tab behaves
consistently.
"""

from __future__ import annotations

import os
import subprocess
from typing import Optional

import numpy as np
import torch


def _probe_fps(path: str) -> float:
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


def decode_video_frames(path: str, frame_skip: int, n_frames: int,
                        W: int, H: int) -> list:
    """Decode `n_frames` starting at `frame_skip` from `path`, resized to
    (W, H). Returns a list of (H, W, 3) uint8 arrays.
    """
    if not os.path.isfile(path):
        raise SystemExit(f"overfit video not found: {path}")
    fps = _probe_fps(path)
    seek_s = max(0, int(frame_skip)) / max(fps, 1e-6)
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ss", str(seek_s),
         "-vf", f"scale={W}:{H}", "-vframes", str(int(n_frames)),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw = result.stdout
    fb = H * W * 3
    return [np.frombuffer(raw[i * fb:(i + 1) * fb], dtype=np.uint8)
            .reshape(H, W, 3)
            for i in range(len(raw) // fb)]


def load_overfit_clip_video(path: str, frame_skip: int, T: int,
                             H: int, W: int, device) -> torch.Tensor:
    """Load T frames from `path` starting at `frame_skip` as (1, T, 3, H, W)
    float tensor in [0, 1] on `device`. Pads with the last frame if the
    source is too short."""
    frames = decode_video_frames(path, frame_skip, T, W, H)
    if not frames:
        raise SystemExit(
            f"overfit video {path} decoded 0 frames at skip={frame_skip}")
    while len(frames) < T:
        frames.append(frames[-1])
    arr = np.stack(frames[:T]).astype(np.float32) / 255.0   # (T, H, W, 3)
    return (torch.from_numpy(arr).permute(0, 3, 1, 2)
            .unsqueeze(0).to(device).contiguous())


def load_overfit_image(path: str, frame_skip: int, H: int, W: int,
                        device) -> torch.Tensor:
    """Load a single frame from `path` (works for stills and videos via
    ffmpeg) as (1, 3, H, W) float tensor in [0, 1] on `device`."""
    frames = decode_video_frames(path, frame_skip, 1, W, H)
    if not frames:
        raise SystemExit(
            f"overfit source {path} decoded 0 frames at skip={frame_skip}")
    arr = frames[0].astype(np.float32) / 255.0              # (H, W, 3)
    return (torch.from_numpy(arr).permute(2, 0, 1)
            .unsqueeze(0).to(device).contiguous())


def add_overfit_args(parser) -> None:
    """Adds --overfit-video / --overfit-frame-skip to any argparse parser."""
    parser.add_argument(
        "--overfit-video", default=None,
        help="If set, train on a single clip (or frame for static models) "
             "decoded from this video. Bypasses the generator.")
    parser.add_argument(
        "--overfit-frame-skip", type=int, default=0,
        help="Frame offset into --overfit-video (converted to seconds "
             "via ffprobe fps internally).")
