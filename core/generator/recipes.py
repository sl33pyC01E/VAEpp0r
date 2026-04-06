#!/usr/bin/env python3
"""Motion recipe pool system for VAEpp generator.

Lightweight motion parameter storage and rendering for fast temporal sampling.
"""

import os
import torch


class RecipesMixin:
    """Mixin providing motion recipe/pool methods for VAEppGenerator."""

    def _generate_recipe(self, T=8, **seq_kwargs):
        """Generate a single motion recipe — lightweight parameter dict.

        A recipe stores which shapes/layers to use and how they move,
        NOT the rendered pixels. ~1KB per recipe vs ~9MB per rendered clip.
        """
        H, W = self.H, self.W
        n_layers = torch.randint(
            self.layers_per_image[0], self.layers_per_image[1] + 1, (1,)).item()
        n_stamps = torch.randint(
            self.stamps_per_image[0], self.stamps_per_image[1] + 1, (1,)).item()

        recipe = {
            "T": T,
            "bg_color": torch.rand(3).tolist(),
            # Layers
            "layer_idx": torch.randint(0, self.n_base_layers, (n_layers,)).tolist(),
            "pan_dx": ((torch.rand(n_layers) - 0.5) * W * seq_kwargs.get("pan_strength", 0.5)).tolist(),
            "pan_dy": ((torch.rand(n_layers) - 0.5) * H * seq_kwargs.get("pan_strength", 0.5)).tolist(),
            "pan_speed": (torch.rand(n_layers) * 0.8 + 0.2).tolist(),
            "opacity_start": (torch.rand(n_layers) * 0.5 + 0.3).tolist(),
            "opacity_end": (torch.rand(n_layers) * 0.5 + 0.3).tolist(),
            "color_shift": ((torch.rand(n_layers, 3) - 0.5) * 0.3).tolist(),
            "zoom_start": (torch.rand(n_layers) * 0.3 + 0.85).tolist(),
            "zoom_end": (torch.rand(n_layers) * 0.3 + 0.85).tolist(),
            # Stamps
            "stamp_idx": torch.randint(0, max(self.bank_size, 1), (n_stamps,)).tolist(),
            "stamp_x": (torch.rand(n_stamps) * W).tolist(),
            "stamp_y": (torch.rand(n_stamps) * H * 0.5).tolist(),
            "stamp_vx": ((torch.rand(n_stamps) - 0.5) * 15).tolist(),
            "stamp_vy": ((torch.rand(n_stamps) - 0.5) * 5).tolist(),
            "stamp_gravity": (torch.rand(n_stamps) * 3 + 1).tolist(),
            "stamp_scale": (torch.rand(n_stamps) * 1.0 + 0.3).tolist(),
            "stamp_rot": (torch.rand(n_stamps) * 6.28).tolist(),
            "stamp_rot_speed": ((torch.rand(n_stamps) - 0.5) * 0.3).tolist(),
            # Viewport
            "vp_pan": ((torch.rand(2) - 0.5) * torch.tensor([W, H]).float() *
                        seq_kwargs.get("viewport_pan", 0.3)).tolist(),
            "vp_zoom": [1.0 - seq_kwargs.get("viewport_zoom", 0.15) / 2,
                         1.0 + seq_kwargs.get("viewport_zoom", 0.15) / 2],
            "vp_rot": ((torch.rand(2) - 0.5) * seq_kwargs.get("viewport_rotation", 0.2)).tolist(),
            # Fine grain
            "fine_layer_idx": torch.randint(0, self.n_base_layers, (1,)).item(),
            "fine_tile": torch.randint(3, 7, (1,)).item(),
            "fine_opacity": torch.rand(1).item() * 0.4 + 0.15,
            "use_fine": torch.rand(1).item() < 0.5,
            # Micro stamps
            "n_micro": torch.randint(20, 60, (1,)).item() if torch.rand(1).item() < 0.5 else 0,
            "micro_idx": torch.randint(0, max(self.bank_size, 1), (60,)).tolist(),
            "micro_x": (torch.rand(60) * W).tolist(),
            "micro_y": (torch.rand(60) * H).tolist(),
            "micro_dx": ((torch.rand(60) - 0.5) * W * 0.15).tolist(),
            "micro_dy": ((torch.rand(60) - 0.5) * H * 0.15).tolist(),
            # Settings
            "gamma": torch.rand(1).item() * 0.7 + 0.7,
            "seq_kwargs": seq_kwargs,
        }
        return recipe

    def build_motion_pool(self, n_clips=200, T=8, **seq_kwargs):
        """Generate motion recipes (not pixels). Lightweight, ~1KB each.

        Args:
            n_clips: number of recipes
            T: frames per clip
            **seq_kwargs: motion settings
        """
        print(f"Building motion recipe pool ({n_clips} recipes, T={T})...",
              flush=True)
        recipes = []
        for i in range(n_clips):
            recipes.append(self._generate_recipe(T=T, **seq_kwargs))
            if (i + 1) % 100 == 0 or i == n_clips - 1:
                print(f"  [{i+1}/{n_clips}]", flush=True)
        self._recipe_pool = recipes
        self._motion_pool_T = T
        self._motion_pool_call_count = 0
        print(f"Recipe pool done: {n_clips} recipes", flush=True)

    def _refresh_motion_pool(self, swap_frac=0.2, **seq_kwargs):
        """Swap a fraction of recipes with fresh ones."""
        if not hasattr(self, '_recipe_pool') or not self._recipe_pool:
            return
        N = len(self._recipe_pool)
        T = self._motion_pool_T
        n_swap = max(1, int(N * swap_frac))
        for _ in range(n_swap):
            idx = torch.randint(0, N, (1,)).item()
            self._recipe_pool[idx] = self._generate_recipe(
                T=T, **seq_kwargs)

    def _render_recipe(self, recipe):
        """Render a single recipe into a (T, 3, H, W) clip on GPU."""
        T = recipe["T"]
        # Reconstruct all the params and call generate_sequence with B=1
        # For simplicity, just call generate_sequence which already handles everything
        # But seed it with the recipe's parameters
        # For now, render fresh with the stored seq_kwargs
        clip = self.generate_sequence(1, T=T, **recipe.get("seq_kwargs", {}))
        return clip[0]  # (T, 3, H, W)

    def generate_from_pool(self, batch_size, refresh_interval=100):
        """Render B clips from recipe pool.

        Picks random recipes, renders them live from the shape bank.
        Applies anti-memorization transforms after rendering.

        Returns: (B, T, 3, H, W) tensor in [0, 1] on self.device.
        """
        if not hasattr(self, '_recipe_pool') or not self._recipe_pool:
            raise RuntimeError("Call build_motion_pool() first")

        B = batch_size
        N = len(self._recipe_pool)

        # Periodic refresh
        self._motion_pool_call_count += 1
        if self._motion_pool_call_count % refresh_interval == 0:
            self._refresh_motion_pool()

        # Render B recipes
        clips = []
        for bi in range(B):
            idx = torch.randint(0, N, (1,)).item()
            clip = self._render_recipe(self._recipe_pool[idx])
            clips.append(clip)
        clips = torch.stack(clips)  # (B, T, 3, H, W)

        # Anti-memorization transforms (fast, on GPU)
        for bi in range(B):
            if torch.rand(1).item() < 0.5:
                clips[bi] = clips[bi].flip(-1)
            if torch.rand(1).item() < 0.2:
                clips[bi] = clips[bi].flip(-2)
            if torch.rand(1).item() < 0.3:
                clips[bi] = clips[bi].flip(0)  # temporal reverse
            shift = (torch.rand(3, 1, 1, device=self.device) - 0.5) * 0.3
            clips[bi] = (clips[bi] + shift).clamp(0, 1)
            if torch.rand(1).item() < 0.3:
                clips[bi] = clips[bi].roll(torch.randint(1, 3, (1,)).item(), dims=1)

        return clips.clamp(0, 1)

    def save_motion_pool(self, path):
        """Save recipe pool to disk with timestamp. Accumulates, doesn't overwrite."""
        if not hasattr(self, '_recipe_pool') or not self._recipe_pool:
            return
        import json, time as _time
        # Timestamped filename in same directory
        d = os.path.dirname(path)
        ts = _time.strftime("%Y%m%d_%H%M%S")
        out = os.path.join(d, f"recipes_{ts}.json")
        os.makedirs(d, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(self._recipe_pool, f)
        kb = os.path.getsize(out) / 1e3
        print(f"Saved {len(self._recipe_pool)} recipes to {out} "
              f"({kb:.0f} KB)", flush=True)

    def load_motion_pool(self, path):
        """Load recipe pool from disk. If path is a directory, loads all recipe files."""
        import json
        if os.path.isdir(path):
            files = sorted(f for f in os.listdir(path)
                           if f.startswith("recipes_") and f.endswith(".json"))
            if not files:
                print(f"No recipe files in {path}", flush=True)
                return
            for f in files:
                fp = os.path.join(path, f)
                with open(fp) as fh:
                    recipes = json.load(fh)
                self._recipe_pool.extend(recipes)
                print(f"  Loaded {len(recipes)} recipes from {f}", flush=True)
        else:
            with open(path) as f:
                recipes = json.load(f)
            self._recipe_pool.extend(recipes)
        self._motion_pool_T = self._recipe_pool[0]["T"] if self._recipe_pool else 8
        self._motion_pool_call_count = 0
        print(f"Total recipes: {len(self._recipe_pool)}", flush=True)

    def motion_pool_stats(self):
        if not hasattr(self, '_recipe_pool') or not self._recipe_pool:
            return None
        return {
            "count": len(self._recipe_pool),
            "T": self._motion_pool_T,
            "kb": len(self._recipe_pool),  # ~1KB each
        }
