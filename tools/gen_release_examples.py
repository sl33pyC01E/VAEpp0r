#!/usr/bin/env python3
"""Generate clean example grids for each disco quadrant at low + high volatility.

Outputs PNG grids (3x3 static samples) and MP4 grids (2x2 video clips) for
all 8 quadrants at two volatility levels. Labels are burned into each file
name. Destination: examples/disco/
"""

import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.generator import VAEpp0rGenerator


OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "examples", "disco")
os.makedirs(OUT_DIR, exist_ok=True)

H, W = 360, 640
T = 17
STATIC_GRID = (3, 3)      # 9 static samples per grid
VIDEO_GRID = (2, 2)       # 4 clips per video grid

QUADRANT_NAMES = [
    "q0_pure_pattern",
    "q1_pattern_collage_shapes",
    "q2_dense_random",
    "q3_structured_scenes",
    "q4_fluid_ripples",
    "q5_3d_raymarch",
    "q6_arcade",
    "q7_signage_text",
]


def tile(tensor_list, rows, cols, gap=6, bg=(20, 20, 28)):
    """tensor_list: list of (3, H, W) float [0,1]. Returns (3, H_total, W_total) uint8."""
    h, w = tensor_list[0].shape[-2], tensor_list[0].shape[-1]
    out = np.full((rows * h + (rows - 1) * gap, cols * w + (cols - 1) * gap, 3),
                  bg, dtype=np.uint8)
    for i, t in enumerate(tensor_list[: rows * cols]):
        r, c = i // cols, i % cols
        img = (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        y0 = r * (h + gap)
        x0 = c * (w + gap)
        out[y0:y0 + h, x0:x0 + w] = img
    return out


def save_png(arr_hwc, path):
    from PIL import Image
    Image.fromarray(arr_hwc).save(path)


def save_mp4_grid(clips, path, fps=12, gap=6, bg=(20, 20, 28)):
    """clips: list of (T, 3, H, W) float [0,1]. Writes MP4 grid."""
    T_ = clips[0].shape[0]
    rows, cols = VIDEO_GRID
    h, w = clips[0].shape[-2], clips[0].shape[-1]
    grid_h = rows * h + (rows - 1) * gap
    grid_w = cols * w + (cols - 1) * gap
    # Pipe raw RGB24 frames to ffmpeg
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{grid_w}x{grid_h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           "-v", "error", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for ti in range(T_):
        frame = np.full((grid_h, grid_w, 3), bg, dtype=np.uint8)
        for i, c in enumerate(clips[: rows * cols]):
            r_, col_ = i // cols, i % cols
            img = (c[ti].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            y0 = r_ * (h + gap)
            x0 = col_ * (w + gap)
            frame[y0:y0 + h, x0:x0 + w] = img
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()


def force_quadrant(gen, q_idx):
    """Pin disco weights to a single quadrant."""
    w = torch.zeros(8, device=gen.device)
    w[q_idx] = 1.0
    gen._disco_weights = w


def vol_kwargs(vol):
    """Low/high volatility parameter sets for generate_sequence."""
    if vol == "low":
        return dict(
            pan_strength=0.35, viewport_pan=0.2,
            viewport_zoom=0.1, viewport_rotation=0.1,
            fast_transform=False,
            use_shake=False,
            use_ripple=False, ripple_warp_strength=4.0, ripple_n_drops=2,
            use_kaleido=False,
            use_flash=False, strobe_rate=0.0,
            use_palette_cycle=False,
            use_particles=False,
            use_glitch=False,
            use_chromatic=False,
            use_scanlines=False,
            grain_strength=0.0,
            use_fire=False, use_vortex=False,
            use_starfield=False, use_eq=False,
        )
    else:  # high
        return dict(
            pan_strength=1.0, viewport_pan=0.7,
            viewport_zoom=0.35, viewport_rotation=0.5,
            fast_transform=True, fast_scale=3.0,
            use_shake=True, shake_mode="earthquake", shake_amp_xy=0.04,
            use_ripple=True, ripple_warp_strength=12.0, ripple_n_drops=6,
            use_kaleido=False,
            use_flash=True, flash_n=2, strobe_rate=0.0,
            use_palette_cycle=True, palette_speed=0.12,
            use_particles=False,
            use_glitch=True, glitch_n=2,
            use_chromatic=True, chromatic_strength=0.015,
            use_scanlines=False, grain_strength=0.0,
            use_fire=False, use_vortex=False,
            use_starfield=False, use_eq=False,
        )


def main():
    print(f"Output: {OUT_DIR}", flush=True)
    gen = VAEpp0rGenerator(H, W, device="cuda",
                           bank_size=300, n_base_layers=32)
    gen.build_banks()
    gen.disco_quadrant = True

    for q_idx, q_name in enumerate(QUADRANT_NAMES):
        force_quadrant(gen, q_idx)
        print(f"\n=== {q_name} (Q{q_idx}) ===", flush=True)

        for vol in ("low", "high"):
            # --- Static grid ---
            t0 = time.time()
            batch = gen.generate(STATIC_GRID[0] * STATIC_GRID[1])
            samples = [batch[i] for i in range(batch.shape[0])]
            grid = tile(samples, *STATIC_GRID)
            png_path = os.path.join(OUT_DIR, f"{q_name}__static__{vol}.png")
            save_png(grid, png_path)
            print(f"  static {vol:4s}: {png_path}  ({time.time()-t0:.1f}s)",
                  flush=True)

            # --- Video grid ---
            t0 = time.time()
            clips = []
            kw = vol_kwargs(vol)
            for _ in range(VIDEO_GRID[0] * VIDEO_GRID[1]):
                # generate_sequence returns (B, T, 3, H, W); take sample 0
                clip = gen.generate_sequence(1, T=T, **kw)[0]
                clips.append(clip.cpu().float())
                del clip
                torch.cuda.empty_cache()
            mp4_path = os.path.join(OUT_DIR, f"{q_name}__video__{vol}.mp4")
            save_mp4_grid(clips, mp4_path, fps=12)
            print(f"  video  {vol:4s}: {mp4_path}  ({time.time()-t0:.1f}s)",
                  flush=True)

    print(f"\nDone. All files in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
