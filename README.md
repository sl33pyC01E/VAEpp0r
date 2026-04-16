# VAEpp0r
Variable Auto Encoder: procedural priors, zero real
<img width="1490" height="423" alt="training_data1" src="https://github.com/user-attachments/assets/4e4531aa-9ade-4f04-817f-de3c76127fae" />
<img width="1488" height="422" alt="training_data2" src="https://github.com/user-attachments/assets/a9ed4dfa-8c34-4e5e-a9e0-91acb4ce3768" />

*No dataset? No problem.*

Procedural synthetic data pretraining for causal temporal video autoencoders.

VAEpp0r trains a video VAE entirely on procedurally generated data -- zero real images required. The generator produces structured synthetic images and video clips with controllable shape composition, scene templates, mathematical patterns, and physics-driven motion. A VAE trained on this synthetic data achieves **pixel-perfect out-of-distribution reconstruction** on real video after temporal training, closing ~90% of the domain gap before any real data is introduced.

https://github.com/user-attachments/assets/c1cf2996-5148-4e41-bd4f-80eff7d2b278

> **Left:** Ground truth Samsung phone ad footage (never seen during training). **Right:** VAE reconstruction from 3-channel latent at 8x spatial + 4x temporal compression. Trained exclusively on procedural synthetic data.

https://github.com/user-attachments/assets/8b2cbde3-8ad3-4c39-abf1-2c5f3cfcdb01

> **Left:** Ground truth training data; procedurally generated prior soup. **Right:** VAE reconstruction from 3-channel latent at 8x spatial + 4x temporal compression.




## Architecture

Two VAE families are provided:

### MiniVAE (2D + MemBlock)

Original architecture built on [TAEHV](https://github.com/madebyollin/taehv). Processes frames as a 2D batch `(N*T, C, H, W)` with per-frame causal state carried through `MemBlock` (current + previous frame concat → conv). `TPool`/`TGrow` handle temporal stride via 1x1 channel-stacking.

- 8x spatial compression, configurable 1-4x temporal compression
- Configurable latent channels (3-32), encoder/decoder widths
- Optional DC-AE residual shortcuts, spatial linear attention, GroupNorm
- Optional Haar wavelet 2x/4x spatial pre-compression
- Optional FSQ post-hoc quantization (via `experiments/fsq.py`)
- Flatten/Deflatten bottleneck for 1D latent serialization (downstream world models)

### MiniVAE3D (causal 3D, Cosmos-style)

Native 5D tensor architecture `(B, C, T, H, W)` with learned temporal kernels at every layer — no 2D reshape trick. Directly inspired by NVIDIA's Cosmos-Tokenizer paper (Agarwal et al., 2025), adapted for much smaller compute budget and 360p-focused training.

- **Causal 3D convolutions** — first frame replicated in temporal pad so no future-frame leakage
- **Factorized res blocks** — spatial `(1,3,3)` then temporal `(3,1,1)` convs, sequential not fused
- **Causal temporal attention** at bottleneck — lower-triangular mask, per-spatial-location
- **Hybrid down/up sampling** — strided conv + avg pool, summed (more stable gradients than pure conv)
- **Optional 3D Haar patcher** — lossless 8x per level (spatial 2x2 + temporal 2x), acts as a free frequency-domain front-end
- **Optional residual FSQ** — N stacked FSQuantizer stages each operating on the residual of the previous, no codebook (deterministic `round()` + straight-through gradient)

### Parameter counts

All values are for RGB input/output (3 channels in, 3 channels out). Total compression is `temporal × spatial × spatial`.

| Config | Base ch | Mult | Haar | FSQ | Compression | Params |
|--------|--------:|------|------|-----|-------------|-------:|
| **MiniVAE Pico** (2D) | — | — | no | no | 8s | 1.0M |
| **MiniVAE Nano** (2D) | — | — | no | no | 8s | 1.2M |
| **MiniVAE Small** (2D) | 64 | 256,128,64 | no | no | 8s × 4t | 4.0M |
| **MiniVAE Medium** (2D) | 64 | 256,128,64 | no | no | 8s × 4t | 11.3M |
| **MiniVAE3D Tiny** | 48 | 1,2,4 | 1 lvl | no | 16s × 8t | 11.1M |
| **MiniVAE3D Base** | 64 | 1,2,4,4 | no | no | 16s × 8t | 27.6M |
| **MiniVAE3D Large** | 128 | 1,2,4,4 | 2 lvl | yes (4 stage) | 16s × 8t | 98.0M |
| Cosmos-Tokenize1-DV8x16x16 (NVIDIA) | 128 | 1,2,4,4 | 2 lvl | yes (4 stage) | 16s × 8t | 105M |

The Large preset is deliberately configured to nearly match Cosmos at the same compression ratio, so you can run apples-to-apples comparisons on your own data.

## Procedural Generator

Three-tier GPU-accelerated generation pipeline:

1. **Shape Bank** -- 10 SDF primitives (circle, rect, triangle, ellipse, blob, line, stroke, hatch, stipple, fractal) x 4 textures (flat, perlin, gradient, voronoi) x 3 edges (hard, soft, textured)
2. **Base Layers** -- Composited scenes from shape bank with 19 scene templates (horizon, perspective, block city, landscape, road, water, forest, etc.)
3. **Final Output** -- Fast layer compositing with transforms, stamps, micro-stamps, post-processing

### Pattern System (38 generators)

Clean mathematical/structural patterns with zero shape compositing:
- Gradients (linear, radial, angular, diamond, multi-stop)
- Tilings (checkerboard, stripes, hexagonal, brick, herringbone, basketweave, fish scale, chevron, argyle)
- Waves (sine, interference/moire, concentric rings, spirals, ripples)
- Mathematical surfaces (quadratic contours, Lissajous, rose curves, spirograph, Julia sets)
- Symmetry/Op Art (kaleidoscope, warped grids, Islamic star patterns)
- Procedural natural (reaction-diffusion, contour maps, wood grain, marble, cracked earth)
- Art exercises (zentangle, maze generation, contour lines, squiggle fill)
- Fine-grain (halftone, ordered dither, stipple density)

### Disco Quadrant Mode

Balanced training data diversity:
- 25% pure mathematical patterns
- 25% pattern collages with sparse shape overlay
- 25% dense random compositing (cranked micro-stamps)
- 25% structured scene templates

### Temporal

Physics-driven motion (gravity, velocity, bounce), viewport transforms (pan, zoom, rotation), fluid advection, parallax layers. Motion stored as compact JSON recipes (~1KB vs ~9MB rendered).

## Training Pipeline

| Stage | Data | Temporal | Description |
|-------|------|----------|-------------|
| 1 | Static synthetic | No | RGB image reconstruction (MiniVAE) |
| 2 (2D) | Temporal synthetic | 1-4x | Video with MemBlock per-frame causal state |
| 2 (3D) | Temporal synthetic | 2-16x | MiniVAE3D causal 3D with full temporal modeling |
| 3 | Flatten | Optional | 1D bottleneck for sequence input (downstream world models) |

FSQ is now an inline option during Stage 2 (3D) training rather than a separate pass.

## GUI

Tkinter desktop app with nested tab layout:

- **Data** -- Static generator controls, video generator with motion pool
- **Models** -- Static train/inf, convert, **Video Train** (2D+MemBlock), **Video Train 3D** (causal 3D), Video Inf
- **Compress** -- Flatten/deflatten experiments for 1D latent serialization

All tabs support disco quadrant mode, resume from checkpoint, and auto-save inference outputs.

## Usage

## Setup

```bash
# Windows
setup.bat

# Manual
python -m venv venv
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Usage

```bash
# Launch GUI
gui.bat
# or: python -m gui.app

# Train static VAE (Stage 1)
python -m training.train_static --disco

# Train temporal VAE (Stage 2, 2D + MemBlock)
python -m training.train_video --disco

# Train temporal VAE (Stage 2, causal 3D — Cosmos-style)
python -m training.train_video3d --disco --base-ch 64 --ch-mult 1,2,4,4

# Cosmos-like large config with Haar + residual FSQ
python -m training.train_video3d --disco --base-ch 128 --ch-mult 1,2,4,4 --haar-levels 2 --fsq

# Flatten experiment
python -m experiments.flatten --vae-ckpt synthyper_logs/latest.pt
```

## Project Structure

```
setup.bat               # Create venv + install deps
gui.bat                 # Launch GUI
requirements.txt

core/                   # Core modules
  generator.py          # Procedural image/video generator
  patterns.py           # 38 mathematical pattern generators
  pattern_collage.py    # Pattern combination operations
  model.py              # MiniVAE architecture
  fsq.py                # Finite Scalar Quantization layer

gui/                    # Tkinter desktop GUI
  app.py                # Main window (Data | Models | Compress)
  common.py             # Shared theme, helpers, process runner
  data_tabs.py          # Static + Video generator tabs
  models_tabs.py        # Training + Inference + Convert tabs
  compress_tabs.py      # Flatten experiment tabs

training/               # Training scripts
  train_static.py       # Stage 1: static image VAE
  train_video.py        # Stage 2: temporal video VAE (2D + MemBlock)
  train_video3d.py      # Stage 2 (3D): causal 3D VAE (MiniVAE3D)

experiments/            # Compression experiments
  flatten.py            # Flatten/deflatten bottleneck (static)
  flatten_video.py      # Flatten/deflatten bottleneck (temporal)

pretrained/             # Pretrained checkpoints
  3ch_S8x.pt            # 3ch RGB, 8x spatial, static
  3ch_S8x_T4x.pt       # 3ch RGB, 8x spatial, 4x temporal
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- ffmpeg (for video preview/inference)
- PIL/Pillow, numpy

## Acknowledgments

- **[TAEHV](https://github.com/madebyollin/taehv)** by madebyollin — the causal temporal VAE architecture (MemBlock, TPool, TGrow) that MiniVAE builds on
- **[Cosmos Tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer)** & **[Cosmos World Foundation Model Platform](https://arxiv.org/abs/2501.03575)** (Agarwal et al., 2025) — the causal 3D tokenizer design (factorized convolutions, causal temporal attention, hybrid down/up sampling, 3D Haar patching, residual FSQ) that MiniVAE3D adapts to a smaller, 360p-focused budget
- **[Revisiting Dead Leaves Model: Training with Synthetic Data](https://ieeexplore.ieee.org/document/9633158/)** (Madhusudana et al., 2021) — demonstrated that neural networks trained on procedural dead leaves images can approach the performance of networks trained on real data
- **[Finite Scalar Quantization: VQ-VAE Made Simple](https://arxiv.org/abs/2309.15505)** (Mentzer et al., ICLR 2024) — the FSQ quantization method used for discrete latent tokens

## License

MIT with Attribution — free to use, modify, and distribute, but you must credit the original author and link back to this repository. See [LICENSE](LICENSE) for details.
