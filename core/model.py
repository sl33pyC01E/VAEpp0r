#!/usr/bin/env python3
"""Mini VAE — causal temporal video autoencoder.

Architecture follows TAEHV (madebyollin/taehv) with custom channel widths.
Building blocks (conv, Clamp, MemBlock, TPool, TGrow) inlined below.

Input:  (N, T, C_in, H, W)
Latent: (N, T', C_lat, H/S, W/S)  — configurable spatial (default 8x), up to 4x temporal
Output: (N, T'', C_out, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from tqdm.auto import tqdm


# -- TAEHV building blocks (from madebyollin/taehv) ---------------------------

TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))

def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class ResidualDownsample(nn.Module):
    """Stride-2 downsample with non-parametric pixel_unshuffle shortcut (DC-AE style).

    Main path: conv(in_ch, out_ch, stride=2)
    Shortcut: pixel_unshuffle(2) -> channel average to match out_ch
    Output: main + shortcut
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = conv(in_channels, out_channels, stride=2, bias=False)
        # Shortcut: pixel_unshuffle gives in_ch * 4 channels
        self.group_size = in_channels * 4 // out_channels

    def forward(self, x):
        main_out = self.main(x)
        # Pad to even dims for pixel_unshuffle
        _, _, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        x_padded = F.pad(x, (0, pad_w, 0, pad_h)) if (pad_h or pad_w) else x
        shortcut = F.pixel_unshuffle(x_padded, 2)
        # Crop shortcut to match main_out spatial dims
        shortcut = shortcut[:, :, :main_out.shape[2], :main_out.shape[3]]
        B, C, H, W = shortcut.shape
        shortcut = shortcut.view(B, main_out.shape[1], self.group_size, H, W).mean(dim=2)
        return main_out + shortcut


class ResidualUpsampleSave(nn.Module):
    """Save input for pixel_shuffle shortcut, then apply nn.Upsample(2) (DC-AE style).

    Paired with ResidualUpsampleAdd after the channel-change conv.
    Keeps TGrow as a separate Sequential layer for temporal walker compatibility.
    """
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self._saved = None
        self._ref_count = 0  # track consumers for TGrow expansion

    def forward(self, x):
        self._saved = x  # save pre-upsample tensor for shortcut
        self._ref_count = 0
        return self.upsample(x)


class ResidualUpsampleAdd(nn.Module):
    """Add pixel_shuffle shortcut from paired ResidualUpsampleSave (DC-AE style).

    Takes the output of the channel-change conv and adds the non-parametric
    pixel_shuffle shortcut computed from the saved pre-upsample input.
    Handles TGrow batch expansion: if main path batch > saved batch,
    repeats the shortcut to match (temporal stride).
    """
    def __init__(self, in_channels, out_channels, save_module):
        super().__init__()
        self.save_module = save_module  # reference to paired Save module
        self.repeats = out_channels * 4 // in_channels

    def forward(self, x):
        saved = self.save_module._saved
        shortcut = saved.repeat_interleave(self.repeats, dim=1)
        shortcut = F.pixel_shuffle(shortcut, 2)
        # Handle TGrow batch expansion: saved was (NT, ...), x may be (NT*stride, ...)
        if x.shape[0] != shortcut.shape[0]:
            t_stride = x.shape[0] // shortcut.shape[0]
            shortcut = shortcut.repeat_interleave(t_stride, dim=0)
        return x + shortcut

class MemBlock(nn.Module):
    def __init__(self, n_in, n_out, use_groupnorm=False):
        super().__init__()
        layers = [conv(n_in * 2, n_out)]
        if use_groupnorm:
            layers.append(nn.GroupNorm(min(8, n_out // 8) if n_out >= 8 else 1, n_out))
        layers += [nn.ReLU(inplace=True), conv(n_out, n_out)]
        if use_groupnorm:
            layers.append(nn.GroupNorm(min(8, n_out // 8) if n_out >= 8 else 1, n_out))
        layers += [nn.ReLU(inplace=True), conv(n_out, n_out)]
        self.conv = nn.Sequential(*layers)
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class SpatialLinearAttention(nn.Module):
    def __init__(self, n_ch, num_heads=8, use_groupnorm=False):
        super().__init__()
        assert n_ch % num_heads == 0
        self.norm = nn.GroupNorm(min(8, n_ch // 8), n_ch) if use_groupnorm else nn.Identity()
        self.qkv = nn.Conv2d(n_ch, 3 * n_ch, 1, bias=False)
        self.proj = nn.Conv2d(n_ch, n_ch, 1, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.heads = num_heads
        self.d = n_ch // num_heads

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B * self.heads, 3, self.d, N)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        k = k.softmax(dim=-1)
        ctx = k @ v.transpose(-2, -1)
        q = q.softmax(dim=-2)
        out = (ctx.transpose(-2, -1) @ q)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)

class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)
    def forward(self, x):
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))

class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)
    def forward(self, x):
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)

def apply_model_with_memblocks_parallel(model, x, show_progress_bar,
                                        use_checkpoint=False):
    assert x.ndim == 5
    N, T, C, H, W = x.shape
    x = x.reshape(N * T, C, H, W)
    for b in tqdm(model, disable=not show_progress_bar):
        if isinstance(b, MemBlock):
            NT, C, H, W = x.shape
            T = NT // N
            _x = x.reshape(N, T, C, H, W)
            block_memory = F.pad(_x, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :T].reshape(x.shape)
            if use_checkpoint:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(b, x, block_memory, use_reentrant=False)
            else:
                x = b(x, block_memory)
        else:
            x = b(x)
    NT, C, H, W = x.shape
    T = NT // N
    return x.view(N, T, C, H, W)

def apply_model_with_memblocks_sequential_single_step(model, memory, work_queue, progress_bar=None):
    while work_queue:
        xt, i = work_queue.pop(0)
        if progress_bar is not None and i == 0:
            progress_bar.update(1)
        if i == len(model):
            return xt.unsqueeze(1)
        b = model[i]
        if isinstance(b, MemBlock):
            if memory[i] is None:
                xt_new = b(xt, xt * 0)
            else:
                xt_new = b(xt, memory[i])
            memory[i] = xt
            work_queue.insert(0, TWorkItem(xt_new, i + 1))
        elif isinstance(b, TPool):
            if memory[i] is None:
                memory[i] = []
            memory[i].append(xt)
            if len(memory[i]) == b.stride:
                # Stack along temporal dim then reshape to (N*stride, C, H, W)
                # so each batch item's frames are contiguous — matching the
                # parallel path's x.reshape(N*T, C, H, W) layout.
                xt = b(torch.stack(memory[i], 1).reshape(-1, *xt.shape[1:]))
                memory[i] = []
                work_queue.insert(0, TWorkItem(xt, i + 1))
        elif isinstance(b, TGrow):
            xt = b(xt)  # (N*stride, C, H, W)
            N_batch = xt.shape[0] // b.stride
            # Reshape to (N, stride, C, H, W) and unbind along temporal dim
            for xt_next in reversed(list(
                    xt.reshape(N_batch, b.stride, *xt.shape[1:]).unbind(1))):
                work_queue.insert(0, TWorkItem(xt_next, i + 1))
        else:
            xt = b(xt)
            work_queue.insert(0, TWorkItem(xt, i + 1))
    return None

def apply_model_with_memblocks_sequential(model, x, show_progress_bar):
    assert x.ndim == 5
    work_queue = [TWorkItem(xt, 0) for xt in x.unbind(1)]
    memory = [None] * len(model)
    progress_bar = tqdm(range(len(work_queue)), disable=not show_progress_bar)
    out = []
    while work_queue:
        xt = apply_model_with_memblocks_sequential_single_step(model, memory, work_queue, progress_bar)
        if xt is not None:
            out.append(xt)
    progress_bar.close()
    return torch.cat(out, 1)

def apply_model_with_memblocks(model, x, parallel, show_progress_bar,
                               use_checkpoint=False):
    if parallel:
        return apply_model_with_memblocks_parallel(model, x, show_progress_bar,
                                                    use_checkpoint)
    else:
        return apply_model_with_memblocks_sequential(model, x, show_progress_bar)


# -- Model --------------------------------------------------------------------

OutputChannels = namedtuple("OutputChannels", ("rgb", "depth", "flow", "semantic"))


class MiniVAE(nn.Module):
    """Causal temporal VAE, 9ch.

    RGB(3) + depth(1) + flow(2) + semantic(3) = 9ch.
    TAEHV architecture. Continuous latent (FSQ added in Phase 2).
    """

    def __init__(
        self,
        latent_channels=16,
        image_channels=9,
        output_channels=9,
        encoder_channels=64,
        decoder_channels=(192, 96, 64),
        encoder_time_downscale=(True, True, False),
        decoder_time_upscale=(False, True, True),
        encoder_spatial_downscale=None,
        decoder_spatial_upscale=None,
        residual_shortcut=False,
        use_attention=False,
        use_groupnorm=False,
        checkpoint_path=None,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.image_channels = image_channels
        self.output_channels = output_channels

        n_stages = len(decoder_channels)
        assert len(encoder_time_downscale) == n_stages, \
            f"encoder_time_downscale length {len(encoder_time_downscale)} != {n_stages} stages"
        assert len(decoder_time_upscale) == n_stages, \
            f"decoder_time_upscale length {len(decoder_time_upscale)} != {n_stages} stages"

        # Spatial downscale/upscale per stage (default: all True = current behavior)
        if encoder_spatial_downscale is None:
            encoder_spatial_downscale = tuple([True] * n_stages)
        if decoder_spatial_upscale is None:
            decoder_spatial_upscale = tuple([True] * n_stages)
        assert len(encoder_spatial_downscale) == n_stages, \
            f"encoder_spatial_downscale length {len(encoder_spatial_downscale)} != {n_stages} stages"
        assert len(decoder_spatial_upscale) == n_stages, \
            f"decoder_spatial_upscale length {len(decoder_spatial_upscale)} != {n_stages} stages"

        # Normalize encoder_channels to a tuple matching n_stages
        if isinstance(encoder_channels, int):
            ec = tuple([encoder_channels] * n_stages)
        else:
            ec = tuple(encoder_channels)
            assert len(ec) == n_stages, \
                f"encoder_channels length {len(ec)} != {n_stages} stages"

        # Encoder: RGB -> latent (configurable channels per stage)
        enc_layers = [conv(image_channels, ec[0]), nn.ReLU(inplace=True)]
        for i in range(n_stages):
            prev_ch = ec[i - 1] if i > 0 else ec[0]
            cur_ch = ec[i]
            s_down = encoder_spatial_downscale[i]
            enc_layers.append(TPool(prev_ch, 2 if encoder_time_downscale[i] else 1))
            if s_down and residual_shortcut:
                enc_layers.append(ResidualDownsample(prev_ch, cur_ch))
            else:
                enc_layers.append(conv(prev_ch, cur_ch, stride=2 if s_down else 1, bias=False))
            if use_attention and i == n_stages - 1:
                enc_layers.extend([
                    MemBlock(cur_ch, cur_ch, use_groupnorm),
                    MemBlock(cur_ch, cur_ch, use_groupnorm),
                    SpatialLinearAttention(cur_ch, use_groupnorm=use_groupnorm),
                    MemBlock(cur_ch, cur_ch, use_groupnorm),
                ])
            else:
                enc_layers.extend([
                    MemBlock(cur_ch, cur_ch, use_groupnorm),
                    MemBlock(cur_ch, cur_ch, use_groupnorm),
                    MemBlock(cur_ch, cur_ch, use_groupnorm),
                ])
        enc_layers.append(conv(ec[-1], latent_channels))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder: latent -> multi-modal output (configurable stages)
        n_f = list(decoder_channels)
        dec_layers = [Clamp(), conv(latent_channels, n_f[0]), nn.ReLU(inplace=True)]
        for i in range(n_stages):
            out_ch = n_f[i + 1] if i < n_stages - 1 else n_f[i]
            if use_attention and i == 0:
                dec_layers.extend([
                    MemBlock(n_f[i], n_f[i], use_groupnorm),
                    MemBlock(n_f[i], n_f[i], use_groupnorm),
                    SpatialLinearAttention(n_f[i], use_groupnorm=use_groupnorm),
                    MemBlock(n_f[i], n_f[i], use_groupnorm),
                ])
            else:
                dec_layers.extend([
                    MemBlock(n_f[i], n_f[i], use_groupnorm),
                    MemBlock(n_f[i], n_f[i], use_groupnorm),
                    MemBlock(n_f[i], n_f[i], use_groupnorm),
                ])
            s_up = decoder_spatial_upscale[i]
            if s_up and residual_shortcut:
                save_mod = ResidualUpsampleSave()
                dec_layers.append(save_mod)
                dec_layers.append(TGrow(n_f[i], 2 if decoder_time_upscale[i] else 1))
                dec_layers.append(conv(n_f[i], out_ch, bias=False))
                dec_layers.append(ResidualUpsampleAdd(n_f[i], out_ch, save_mod))
            else:
                if s_up:
                    dec_layers.append(nn.Upsample(scale_factor=2))
                dec_layers.append(TGrow(n_f[i], 2 if decoder_time_upscale[i] else 1))
                dec_layers.append(conv(n_f[i], out_ch, bias=False))
        dec_layers.extend([nn.ReLU(inplace=True), conv(n_f[-1], output_channels)])
        self.decoder = nn.Sequential(*dec_layers)

        # Computed properties
        self.t_downscale = 1
        for m in self.encoder:
            if isinstance(m, TPool) and m.stride > 1:
                self.t_downscale *= m.stride

        self.t_upscale = 1
        for m in self.decoder:
            if isinstance(m, TGrow) and m.stride > 1:
                self.t_upscale *= m.stride

        self.s_downscale = 2 ** sum(encoder_spatial_downscale)
        self.s_upscale = 2 ** sum(decoder_spatial_upscale)

        self.frames_to_trim = self.t_upscale - 1
        self.use_checkpoint = False

        if checkpoint_path is not None:
            ckpt_data = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True)
            if isinstance(ckpt_data, dict) and "model" in ckpt_data:
                ckpt_data = ckpt_data["model"]
            self.load_state_dict(ckpt_data)

    @classmethod
    def from_pretrained_taehv(cls, taehv_path="taehv/taehv.pth", **kwargs):
        """Create MiniVAE initialized from pretrained 3ch TAEHV weights.

        Loads TAEHV weights, copies matching internal layers (MemBlocks,
        stride convs), inflates first encoder conv (3→6ch, zero-init semantic)
        and last decoder conv (3→6ch, zero-init semantic). Latent projection
        layers are randomly initialized if latent_channels differs from TAEHV's 16.
        """
        # Build with TAEHV-matching defaults for max weight transfer
        model = cls(
            latent_channels=kwargs.get("latent_channels", 16),
            image_channels=9,
            output_channels=9,
            encoder_channels=64,
            decoder_channels=(256, 128, 64),
            **{k: v for k, v in kwargs.items()
               if k not in ("latent_channels", "encoder_channels", "decoder_channels")},
        )

        # Load TAEHV state dict
        taehv_sd = torch.load(taehv_path, map_location="cpu", weights_only=True)
        model_sd = model.state_dict()

        loaded, skipped = 0, 0
        for key, param in taehv_sd.items():
            if key not in model_sd:
                skipped += 1
                continue
            target = model_sd[key]
            if param.shape == target.shape:
                model_sd[key] = param
                loaded += 1
            elif key == "encoder.0.weight":
                # First conv: (64, 3, 3, 3) → (64, 9, 3, 3)
                model_sd[key][:, :3] = param
                loaded += 1
            elif key == "decoder.22.weight":
                # Last conv: (3, 64, 3, 3) → (9, 64, 3, 3)
                model_sd[key][:3] = param
                loaded += 1
            else:
                skipped += 1

        model.load_state_dict(model_sd)
        print(f"TAEHV init: {loaded} layers loaded, {skipped} skipped")
        return model

    def encode_video(self, x, parallel=True, show_progress_bar=False):
        """Encode RGB video to latents.

        Args:
            x: (N, T, C, H, W) RGB tensor in [0, 1], C=image_channels.
            parallel: process all frames at once (fast, more memory).

        Returns:
            (N, T', latent_channels, H/8, W/8) latent tensor.
        """
        assert x.ndim == 5, f"Expected NTCHW, got {x.ndim}D"
        # Pad T to multiple of t_downscale
        if x.shape[1] % self.t_downscale != 0:
            n_pad = self.t_downscale - x.shape[1] % self.t_downscale
            padding = x[:, -1:].repeat_interleave(n_pad, dim=1)
            x = torch.cat([x, padding], 1)
        return apply_model_with_memblocks(
            self.encoder, x, parallel, show_progress_bar, self.use_checkpoint)

    def decode_video(self, x, parallel=True, show_progress_bar=False):
        """Decode latents to multi-modal output.

        Args:
            x: (N, T, latent_channels, H, W) latent tensor.
            parallel: process all frames at once (fast, more memory).

        Returns:
            (N, T'', output_channels, H*8, W*8) output tensor.
            T'' = T * t_upscale - frames_to_trim.
        """
        assert x.ndim == 5, f"Expected NTCHW, got {x.ndim}D"
        x = apply_model_with_memblocks(
            self.decoder, x, parallel, show_progress_bar, self.use_checkpoint)
        return x[:, self.frames_to_trim:]

    def forward(self, x):
        """Full encode-decode pass.

        Args:
            x: (N, T, 9, H, W) tensor in [0, 1].
               RGB(3) + depth(1) + flow(2) + semantic(3).

        Returns:
            (recon, latent) where recon is (N, T'', 9, H, W) and
            latent is (N, T', latent_channels, H/8, W/8).
        """
        latent = self.encode_video(x)
        recon = self.decode_video(latent)
        return recon, latent

    def postprocess_output(self, x):
        """Split and postprocess decoder output channels.

        Args:
            x: (N, T, 9, H, W) raw decoder output.

        Returns:
            OutputChannels namedtuple with:
                rgb: (N, T, 3, H, W) clamped to [0, 1]
                depth: (N, T, 1, H, W) clamped to [0, 1]
                flow: (N, T, 2, H, W)
                semantic: (N, T, 3, H, W) clamped to [0, 1]
        """
        return OutputChannels(
            rgb=x[:, :, :3].clamp(0, 1),
            depth=x[:, :, 3:4].clamp(0, 1),
            flow=x[:, :, 4:6],
            semantic=x[:, :, 6:9].clamp(0, 1),
        )

    def param_count(self):
        """Return parameter counts for encoder, decoder, and total."""
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        return {"encoder": enc, "decoder": dec, "total": enc + dec}


class StreamingMiniVAE(nn.Module):
    """Streaming wrapper for real-time encode/decode.

    Encode-decode (video-to-video) usage:
        streaming = StreamingMiniVAE(model)
        for frame in video_frames:
            latent = streaming.encode(frame_tensor)
            decoded = streaming.decode(latent)
            if decoded is not None:
                display(decoded)
        for frame in streaming.flush():
            display(frame)

    Decode-only (world model) usage:
        streaming = StreamingMiniVAE(model)
        while running:
            latent = world_model.step()
            frame = streaming.decode(latent)
            while frame is not None:
                display(frame)
                frame = streaming.decode()
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.reset()

    def reset(self):
        """Reset all internal state for a new stream."""
        self.encoder_work_queue = []
        self.encoder_memory = [None] * len(self.model.encoder)
        self.decoder_work_queue = []
        self.decoder_memory = [None] * len(self.model.decoder)
        self.n_frames_encoded = 0
        self.n_frames_decoded = 0
        self._last_encoder_input_frame = None

    def encode(self, x=None):
        """Feed an input frame and try to produce an encoder output.

        Args:
            x: (N, 1, C, H, W) RGB frame, or None to process pending work.

        Returns:
            (N, 1, latent_channels, H/8, W/8) latent, or None.
        """
        if x is not None:
            assert x.ndim == 5 and x.shape[2] == self.model.image_channels
            self._last_encoder_input_frame = x[:, -1:]
            self.encoder_work_queue.extend(
                TWorkItem(xt, 0) for xt in x.unbind(1))
            self.n_frames_encoded += x.shape[1]
        return apply_model_with_memblocks_sequential_single_step(
            self.model.encoder, self.encoder_memory, self.encoder_work_queue)

    def decode(self, x=None):
        """Feed a latent and try to produce a decoded frame.

        Args:
            x: (N, 1, latent_channels, H, W) latent, or None.

        Returns:
            (N, 1, output_channels, H*8, W*8) decoded frame, or None.
        """
        if x is not None:
            assert x.ndim == 5 and x.shape[2] == self.model.latent_channels
            self.decoder_work_queue.extend(
                TWorkItem(xt, 0) for xt in x.unbind(1))
        while True:
            xt = apply_model_with_memblocks_sequential_single_step(
                self.model.decoder, self.decoder_memory,
                self.decoder_work_queue)
            if xt is None:
                return None
            self.n_frames_decoded += 1
            # Skip startup frames (causal alignment)
            if self.n_frames_decoded <= self.model.frames_to_trim:
                continue
            return xt

    def flush_encoder(self):
        """Pad and drain remaining latents from the encoder."""
        latents = []
        if (self._last_encoder_input_frame is not None
                and self.n_frames_encoded % self.model.t_downscale != 0):
            n_pad = (self.model.t_downscale
                     - self.n_frames_encoded % self.model.t_downscale)
            for _ in range(n_pad):
                lat = self.encode(self._last_encoder_input_frame)
                if lat is not None:
                    latents.append(lat)
        while (lat := self.encode()) is not None:
            latents.append(lat)
        return latents

    def flush_decoder(self):
        """Drain all remaining decoded frames."""
        frames = []
        while (frame := self.decode()) is not None:
            frames.append(frame)
        return frames

    def flush(self):
        """Flush encoder then decoder, returning all remaining frames."""
        frames = []
        for latent in self.flush_encoder():
            frame = self.decode(latent)
            if frame is not None:
                frames.append(frame)
        frames.extend(self.flush_decoder())
        return frames


# ==========================================================================
# MiniVAE3D — Cosmos-style causal 3D architecture
# ==========================================================================

class Haar3DPatcher(nn.Module):
    """3D Haar wavelet decomposition: spatial 2x + causal temporal 2x, lossless.
    (B, C, T, H, W) -> (B, 8C, T', H/2, W/2) where T' handles causal temporal.
    Stack `n` times for 2^n compression.
    """

    def __init__(self, n_levels=2):
        super().__init__()
        self.n_levels = n_levels

    @staticmethod
    def _one_level(x):
        # Input T must be even. Caller is responsible for T divisibility.
        a = x[:, :, 0::2, 0::2, 0::2]
        b = x[:, :, 0::2, 0::2, 1::2]
        c = x[:, :, 0::2, 1::2, 0::2]
        d = x[:, :, 0::2, 1::2, 1::2]
        e = x[:, :, 1::2, 0::2, 0::2]
        f = x[:, :, 1::2, 0::2, 1::2]
        g = x[:, :, 1::2, 1::2, 0::2]
        h = x[:, :, 1::2, 1::2, 1::2]
        s = 2.0 ** -1.5
        bands = [
            (a + b + c + d + e + f + g + h),
            (a - b + c - d + e - f + g - h),
            (a + b - c - d + e + f - g - h),
            (a - b - c + d + e - f - g + h),
            (a + b + c + d - e - f - g - h),
            (a - b + c - d - e + f - g + h),
            (a + b - c - d - e - f + g + h),
            (a - b - c + d - e + f + g - h),
        ]
        return torch.cat([b_ * s for b_ in bands], dim=1)

    def forward(self, x):
        for _ in range(self.n_levels):
            x = self._one_level(x)
        return x


class Haar3DUnpatcher(nn.Module):
    """Inverse of Haar3DPatcher: (B, 8^n C, T', H, W) -> (B, C, T, H*2^n, W*2^n)."""

    def __init__(self, n_levels=2):
        super().__init__()
        self.n_levels = n_levels

    @staticmethod
    def _one_level(x):
        C8 = x.shape[1]
        C = C8 // 8
        s = 2.0 ** -1.5
        bands = [x[:, i * C:(i + 1) * C] * s for i in range(8)]
        lll, llh, lhl, lhh, hll, hlh, hhl, hhh = bands
        # Invert the mixing matrix
        a = (lll + llh + lhl + lhh + hll + hlh + hhl + hhh)
        b = (lll - llh + lhl - lhh + hll - hlh + hhl - hhh)
        c = (lll + llh - lhl - lhh + hll + hlh - hhl - hhh)
        d = (lll - llh - lhl + lhh + hll - hlh - hhl + hhh)
        e = (lll + llh + lhl + lhh - hll - hlh - hhl - hhh)
        f = (lll - llh + lhl - lhh - hll + hlh - hhl + hhh)
        g = (lll + llh - lhl - lhh - hll - hlh + hhl + hhh)
        h = (lll - llh - lhl + lhh - hll + hlh + hhl - hhh)
        B, Ch, T, H, W = a.shape
        out = torch.zeros(B, Ch, T * 2, H * 2, W * 2, device=x.device, dtype=x.dtype)
        out[:, :, 0::2, 0::2, 0::2] = a
        out[:, :, 0::2, 0::2, 1::2] = b
        out[:, :, 0::2, 1::2, 0::2] = c
        out[:, :, 0::2, 1::2, 1::2] = d
        out[:, :, 1::2, 0::2, 0::2] = e
        out[:, :, 1::2, 0::2, 1::2] = f
        out[:, :, 1::2, 1::2, 0::2] = g
        out[:, :, 1::2, 1::2, 1::2] = h
        return out

    def forward(self, x):
        for _ in range(self.n_levels):
            x = self._one_level(x)
        return x


class FSQuantizer(nn.Module):
    """Finite Scalar Quantization: round each channel to one of `levels[i]` values.
    No learnable codebook — straight-through estimator."""

    def __init__(self, levels=(8, 8, 8, 5, 5, 5)):
        super().__init__()
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.int64))
        basis = torch.cumprod(torch.tensor([1] + list(levels[:-1])), dim=0).long()
        self.register_buffer("basis", basis)
        self.dim = len(levels)
        self.codebook_size = int(torch.prod(self.levels).item())

    def _bound(self, z, eps=1e-3):
        shape = [1, -1] + [1] * (z.ndim - 2)
        half_l = ((self.levels.float() - 1) * (1 + eps) / 2).view(shape)
        offset = torch.where(self.levels % 2 == 0,
                             torch.tensor(0.5, device=z.device),
                             torch.tensor(0.0, device=z.device)).view(shape)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def _round_ste(self, z):
        return z + (z.round() - z).detach()

    def forward(self, z):
        """z: (B, dim, ...) -> (quantized codes [-1,1], indices [0, codebook_size))."""
        shape = [1, -1] + [1] * (z.ndim - 2)
        bounded = self._bound(z.float())
        quantized = self._round_ste(bounded)
        half_width = (self.levels // 2).float().view(shape)
        codes = quantized / half_width
        shifted = (quantized + half_width).long()
        indices = (shifted * self.basis.view(shape)).sum(dim=1)
        return codes.to(z.dtype), indices


class ResidualFSQuantizer(nn.Module):
    """Stack N FSQuantizers, each quantizing the residual from the previous."""

    def __init__(self, levels=(8, 8, 8, 5, 5, 5), n_quantizers=4):
        super().__init__()
        self.layers = nn.ModuleList([FSQuantizer(levels) for _ in range(n_quantizers)])
        self.n_quantizers = n_quantizers

    def forward(self, z):
        residual = z
        total_q = 0
        all_indices = []
        for layer in self.layers:
            codes, indices = layer(residual)
            all_indices.append(indices)
            residual = residual - codes.detach()
            total_q = total_q + codes
        # Stack indices along a new dim: (B, n_quantizers, ...)
        stacked = torch.stack(all_indices, dim=1)
        return total_q, stacked


class CausalConv3d(nn.Module):
    """3D convolution with causal temporal padding (replicate first frame)."""

    def __init__(self, in_ch, out_ch, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                 dilation=(1, 1, 1), bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        kt, kh, kw = kernel_size
        st, sh, sw = stride
        dt, dh, dw = dilation

        # Temporal: causal pad (replicate first frame)
        self.time_pad = dt * (kt - 1) + (1 - st)
        # Spatial: symmetric padding
        self.pad_h = dh * (kh - 1) // 2
        self.pad_w = dw * (kw - 1) // 2

        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=(0, self.pad_h, self.pad_w),
                              dilation=dilation, bias=bias)

    def forward(self, x):
        # x: (B, C, T, H, W)
        if self.time_pad > 0:
            first = x[:, :, :1].expand(-1, -1, self.time_pad, -1, -1)
            x = torch.cat([first, x], dim=2)
        return self.conv(x)


class CausalGroupNorm(nn.Module):
    """GroupNorm applied per-frame (causal-safe with num_groups=1 = LayerNorm)."""

    def __init__(self, num_channels, num_groups=1):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, eps=1e-6, affine=True)

    def forward(self, x):
        # x: (B, C, T, H, W) -> process per-frame
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.norm(x)
        return x.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)


class FactorizedResBlock(nn.Module):
    """Factorized 3D residual block: spatial (1,3,3) then temporal (3,1,1)."""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.norm1 = CausalGroupNorm(in_ch)
        self.conv1_spatial = CausalConv3d(in_ch, out_ch, (1, 3, 3))
        self.conv1_temporal = CausalConv3d(out_ch, out_ch, (3, 1, 1))

        self.norm2 = CausalGroupNorm(out_ch)
        self.conv2_spatial = CausalConv3d(out_ch, out_ch, (1, 3, 3))
        self.conv2_temporal = CausalConv3d(out_ch, out_ch, (3, 1, 1))

        self.skip = CausalConv3d(in_ch, out_ch, (1, 1, 1)) if in_ch != out_ch else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        h = self.act(self.norm1(x))
        h = self.conv1_temporal(self.conv1_spatial(h))
        h = self.act(self.norm2(h))
        h = self.drop(h)
        h = self.conv2_temporal(self.conv2_spatial(h))
        return h + self.skip(x)


class CausalSpatialAttention(nn.Module):
    """Per-frame spatial attention (no temporal interaction)."""

    def __init__(self, n_ch, num_heads=8):
        super().__init__()
        self.norm = CausalGroupNorm(n_ch)
        self.qkv = nn.Conv1d(n_ch, 3 * n_ch, 1, bias=False)
        self.proj = nn.Conv1d(n_ch, n_ch, 1, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.heads = num_heads
        self.d = n_ch // num_heads

    def forward(self, x):
        B, C, T, H, W = x.shape
        h = self.norm(x)
        # Process per-frame: (B*T, C, H*W)
        h = h.permute(0, 2, 1, 3, 4).reshape(B * T, C, H * W)
        qkv = self.qkv(h).reshape(B * T, 3, self.heads, self.d, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        # (B*T, heads, d, N) -> attention
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * (self.d ** -0.5)
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(B * T, C, H * W)
        out = self.proj(out).reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        return x + out


class CausalTemporalAttention(nn.Module):
    """Per-spatial-location temporal attention with causal mask."""

    def __init__(self, n_ch):
        super().__init__()
        self.norm = CausalGroupNorm(n_ch)
        self.q = nn.Conv1d(n_ch, n_ch, 1, bias=False)
        self.k = nn.Conv1d(n_ch, n_ch, 1, bias=False)
        self.v = nn.Conv1d(n_ch, n_ch, 1, bias=False)
        self.proj = nn.Conv1d(n_ch, n_ch, 1, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        B, C, T, H, W = x.shape
        h = self.norm(x)
        # (B*H*W, C, T)
        h = h.permute(0, 3, 4, 1, 2).reshape(B * H * W, C, T)
        q = self.q(h)  # (BHW, C, T)
        k = self.k(h)
        v = self.v(h)
        # Attention: (BHW, T, T)
        scale = C ** -0.5
        attn = torch.bmm(q.transpose(1, 2), k) * scale  # (BHW, T, T)
        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=attn.device, dtype=torch.bool))
        attn = attn.masked_fill(~mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        out = torch.bmm(attn, v.transpose(1, 2)).transpose(1, 2)  # (BHW, C, T)
        out = self.proj(out)
        out = out.reshape(B, H, W, C, T).permute(0, 3, 4, 1, 2)
        return x + out


class HybridDownsample3d(nn.Module):
    """Hybrid downsampling: conv + pool summed (Cosmos-style)."""

    def __init__(self, in_ch, out_ch, spatial_down=True, temporal_down=True):
        super().__init__()
        self.spatial_down = spatial_down
        self.temporal_down = temporal_down

        if spatial_down:
            self.s_conv = CausalConv3d(in_ch, in_ch, (1, 3, 3), stride=(1, 2, 2))
            # ceil_mode=True so odd input dims match the conv output
            # (conv: ceil(H/2), pool default: floor(H/2) — mismatch for odd H)
            self.s_pool = nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2), ceil_mode=True)
        if temporal_down:
            self.t_conv = CausalConv3d(in_ch, in_ch, (3, 1, 1), stride=(2, 1, 1))
            self.t_pool = nn.AvgPool3d((2, 1, 1), stride=(2, 1, 1), ceil_mode=True)
        self.out_conv = CausalConv3d(in_ch, out_ch, (1, 1, 1))

    def forward(self, x):
        if self.spatial_down:
            conv_out = self.s_conv(x)
            pool_out = self.s_pool(x)
            # Belt-and-suspenders: trim to the smaller shape if still off by 1
            h = min(conv_out.shape[3], pool_out.shape[3])
            w = min(conv_out.shape[4], pool_out.shape[4])
            x = conv_out[:, :, :, :h, :w] + pool_out[:, :, :, :h, :w]
        if self.temporal_down:
            # Causal temporal pad for pool (replicate first frame)
            x_padded = torch.cat([x[:, :, :1], x], dim=2)
            pooled = self.t_pool(x_padded)
            conv_out = self.t_conv(x)
            # Trim to matching length
            n = min(pooled.shape[2], conv_out.shape[2])
            x = conv_out[:, :, :n] + pooled[:, :, :n]
        return self.out_conv(x)


class HybridUpsample3d(nn.Module):
    """Hybrid upsampling: repeat-interleave + conv residual (Cosmos-style)."""

    def __init__(self, in_ch, out_ch, spatial_up=True, temporal_up=True):
        super().__init__()
        self.spatial_up = spatial_up
        self.temporal_up = temporal_up

        if temporal_up:
            self.t_conv = CausalConv3d(in_ch, in_ch, (3, 1, 1))
        if spatial_up:
            self.s_conv = CausalConv3d(in_ch, in_ch, (1, 3, 3))
        self.out_conv = CausalConv3d(in_ch, out_ch, (1, 1, 1))

    def forward(self, x):
        if self.temporal_up:
            T = x.shape[2]
            if T > 1:
                x = x.repeat_interleave(2, dim=2)
                x = x[:, :, 1:]  # causal offset
            # else single frame, no temporal up
            x = self.t_conv(x) + x
        if self.spatial_up:
            x = x.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)
            x = self.s_conv(x) + x
        return self.out_conv(x)


class MiniVAE3D(nn.Module):
    """Causal 3D video autoencoder with factorized convolutions.

    Cosmos-style architecture: causal 3D convs, factorized (1,3,3)+(3,1,1)
    blocks, hybrid down/up sampling, spatial + temporal attention at bottleneck.

    Input:  (N, T, C_in, H, W)
    Latent: (N, T', C_lat, H', W')
    Output: (N, T'', C_out, H, W)
    """

    def __init__(
        self,
        latent_channels=16,
        image_channels=3,
        output_channels=3,
        base_channels=64,
        channel_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        temporal_downsample=None,
        spatial_downsample=None,
        attn_at_deepest=True,
        dropout=0.0,
        haar_levels=0,               # 0=off, 1=8x, 2=64x (spatial*temporal)
        fsq=False,                   # enable residual FSQ quantizer
        fsq_levels=(8, 8, 8, 5, 5, 5),
        fsq_stages=4,
        fsq_embedding_dim=6,         # dim projected in/out of FSQ
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.image_channels = image_channels
        self.output_channels = output_channels
        self.haar_levels = haar_levels
        self.use_fsq = fsq

        n_levels = len(channel_mult)
        channels = [base_channels * m for m in channel_mult]

        if temporal_downsample is None:
            temporal_downsample = tuple([True] * n_levels)
        if spatial_downsample is None:
            spatial_downsample = tuple([True] * n_levels)
        assert len(temporal_downsample) == n_levels
        assert len(spatial_downsample) == n_levels

        self.temporal_downsample = temporal_downsample
        self.spatial_downsample = spatial_downsample

        # Haar 3D patcher/unpatcher (optional front-end)
        if haar_levels > 0:
            self.haar_patch = Haar3DPatcher(haar_levels)
            self.haar_unpatch = Haar3DUnpatcher(haar_levels)
            enc_in_ch = image_channels * (8 ** haar_levels)
            dec_out_ch = output_channels * (8 ** haar_levels)
        else:
            self.haar_patch = None
            self.haar_unpatch = None
            enc_in_ch = image_channels
            dec_out_ch = output_channels

        # FSQ quantizer (optional)
        if fsq:
            self.quant_proj_in = CausalConv3d(latent_channels, fsq_embedding_dim, (1, 1, 1))
            self.quant_proj_out = CausalConv3d(fsq_embedding_dim, latent_channels, (1, 1, 1))
            self.quantizer = ResidualFSQuantizer(fsq_levels, fsq_stages)
        else:
            self.quant_proj_in = None
            self.quant_proj_out = None
            self.quantizer = None

        # -- Encoder --
        enc = nn.ModuleList()
        # Input conv (factorized)
        enc.append(CausalConv3d(enc_in_ch, channels[0], (1, 3, 3)))
        enc.append(CausalConv3d(channels[0], channels[0], (3, 1, 1)))

        for i in range(n_levels):
            ch_in = channels[i - 1] if i > 0 else channels[0]
            ch_out = channels[i]
            # Res blocks
            for j in range(num_res_blocks):
                enc.append(FactorizedResBlock(ch_in if j == 0 else ch_out, ch_out, dropout))
            # Attention at deepest
            if attn_at_deepest and i == n_levels - 1:
                enc.append(CausalSpatialAttention(ch_out))
                enc.append(CausalTemporalAttention(ch_out))
            # Downsample
            enc.append(HybridDownsample3d(ch_out, ch_out,
                                          spatial_down=spatial_downsample[i],
                                          temporal_down=temporal_downsample[i]))
        # Mid block
        ch_mid = channels[-1]
        enc.append(FactorizedResBlock(ch_mid, ch_mid, dropout))
        enc.append(CausalSpatialAttention(ch_mid))
        enc.append(CausalTemporalAttention(ch_mid))
        enc.append(FactorizedResBlock(ch_mid, ch_mid, dropout))
        # Output conv
        enc.append(CausalGroupNorm(ch_mid))
        enc.append(nn.SiLU())
        enc.append(CausalConv3d(ch_mid, latent_channels, (1, 3, 3)))
        enc.append(CausalConv3d(latent_channels, latent_channels, (3, 1, 1)))
        self.encoder = enc

        # -- Decoder --
        dec = nn.ModuleList()
        # Input conv (factorized)
        dec.append(CausalConv3d(latent_channels, ch_mid, (1, 3, 3)))
        dec.append(CausalConv3d(ch_mid, ch_mid, (3, 1, 1)))
        # Mid block
        dec.append(FactorizedResBlock(ch_mid, ch_mid, dropout))
        dec.append(CausalSpatialAttention(ch_mid))
        dec.append(CausalTemporalAttention(ch_mid))
        dec.append(FactorizedResBlock(ch_mid, ch_mid, dropout))

        ch_prev = ch_mid
        for i in range(n_levels - 1, -1, -1):
            ch_out = channels[i]
            # Upsample
            dec.append(HybridUpsample3d(ch_prev, ch_out,
                                        spatial_up=spatial_downsample[i],
                                        temporal_up=temporal_downsample[i]))
            # Res blocks
            for j in range(num_res_blocks):
                dec.append(FactorizedResBlock(ch_out, ch_out, dropout))
            # Attention at deepest (which is first in decoder)
            if attn_at_deepest and i == n_levels - 1:
                dec.append(CausalSpatialAttention(ch_out))
                dec.append(CausalTemporalAttention(ch_out))
            ch_prev = ch_out

        # Output conv
        ch_final = channels[0]
        dec.append(CausalGroupNorm(ch_final))
        dec.append(nn.SiLU())
        dec.append(CausalConv3d(ch_final, dec_out_ch, (1, 3, 3)))
        dec.append(CausalConv3d(dec_out_ch, dec_out_ch, (3, 1, 1)))
        self.decoder = dec

        # Total compression (including Haar)
        self.t_downscale = (2 ** sum(temporal_downsample)) * (2 ** haar_levels)
        self.s_downscale = (2 ** sum(spatial_downsample)) * (2 ** haar_levels)

    def _run_modules(self, modules, x):
        for m in modules:
            x = m(x)
        return x

    def encode_video(self, x, return_indices=False, **kwargs):
        """Encode video to latent.
        Args: x: (N, T, C, H, W) in [0, 1]
        Returns: (N, T', latent_ch, H', W') (and FSQ indices if enabled)
        """
        assert x.ndim == 5
        # (N, T, C, H, W) -> (N, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        # Pad T, H, W to multiples of the compression factors. Spatial pads
        # use replicate so there's no zero-border artifact. Temporal pad
        # replicates the last frame (matches the causal convention).
        T, H, W = x.shape[2], x.shape[3], x.shape[4]
        t_pad = (-T) % self.t_downscale
        h_pad = (-H) % self.s_downscale
        w_pad = (-W) % self.s_downscale
        if t_pad:
            x = torch.cat([x, x[:, :, -1:].expand(-1, -1, t_pad, -1, -1)], dim=2)
        if h_pad or w_pad:
            x = F.pad(x, (0, w_pad, 0, h_pad, 0, 0), mode="replicate")
        # Haar patcher (if enabled)
        if self.haar_patch is not None:
            x = self.haar_patch(x)
        z = self._run_modules(self.encoder, x)
        indices = None
        if self.quantizer is not None:
            z = self.quant_proj_in(z)
            z, indices = self.quantizer(z)
        # (N, C, T', H', W') -> (N, T', C, H', W')
        z_out = z.permute(0, 2, 1, 3, 4)
        if return_indices:
            return z_out, indices
        return z_out

    def decode_video(self, z, target_shape=None, **kwargs):
        """Decode latent to video.
        Args:
            z: (N, T', latent_ch, H', W')
            target_shape: optional (T, H, W) to crop the decoder output to.
        Returns: (N, T'', C_out, H, W)
        """
        assert z.ndim == 5
        z = z.permute(0, 2, 1, 3, 4)
        if self.quant_proj_out is not None:
            z = self.quant_proj_out(z)
        x = self._run_modules(self.decoder, z)
        if self.haar_unpatch is not None:
            x = self.haar_unpatch(x)
        if target_shape is not None:
            T, H, W = target_shape
            x = x[:, :, :T, :H, :W]
        return x.permute(0, 2, 1, 3, 4)

    def forward(self, x):
        # Preserve the original (T, H, W) so the reconstruction comes back at
        # the same resolution even when encoder pads to multiples internally.
        _, T, _, H, W = x.shape
        latent = self.encode_video(x)
        recon = self.decode_video(latent, target_shape=(T, H, W))
        return recon, latent

    def param_count(self):
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        quant = sum(p.numel() for p in self.modules() if False)  # FSQ has no params
        quant_proj = 0
        if self.quant_proj_in is not None:
            quant_proj = sum(p.numel() for p in self.quant_proj_in.parameters()) \
                       + sum(p.numel() for p in self.quant_proj_out.parameters())
        return {"encoder": enc, "decoder": dec, "fsq_proj": quant_proj,
                "total": enc + dec + quant_proj}
