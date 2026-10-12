# Copyright 2026 Dagimawi Moges Wakumo
# SPDX-License-Identifier: Apache-2.0
"""
Dynamic DINO template memory for the adaptive tracker.

The memory holds three kernels, all stored as L2-normalised patch features
cropped from the template grid around the target:

    anchor      frame-1 features, never overwritten (keeps the target's identity)
    long-term   slow EMA of accepted observations
    short-term  ring buffer of the last K accepted observations (fast appearance changes)

Because correlation is linear in the kernel, the weighted sum of the three
response maps equals the response of the weighted sum of the kernels, so the
whole memory costs a single convolution per search scale.

Update gating uses *relative* statistics instead of absolute thresholds,
because DINO similarity values drift with the target, the scene and the
backbone:

    q_rel  = peak_t / median(peak over the last N frames)
    id_rel = cos(observation, anchor) / median(cos to anchor over the last N frames)

An observation updates the memory only when both are high enough, with a
strength that scales with q_rel.
"""

from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _renorm(kernel: torch.Tensor) -> torch.Tensor:
    """Re-normalise each patch vector of a (1, D, h, w) kernel to unit length."""
    return F.normalize(kernel, dim=1)


class RunningMedian:
    """Median over a sliding window, with a fallback before the window fills."""

    def __init__(self, window: int):
        self.values = deque(maxlen=window)

    def push(self, v: float):
        self.values.append(float(v))

    def median(self, default: float) -> float:
        if len(self.values) == 0:
            return default
        return float(np.median(self.values))


class FeatureMemory:
    """Anchor + long-term + short-term DINO kernel memory with weighted correlation."""

    def __init__(self, weights: Tuple[float, float, float] = (0.3, 0.3, 0.4),
                 long_lr: float = 0.05, short_len: int = 3, stats_window: int = 30):
        self.w_anchor, self.w_long, self.w_short = weights
        self.long_lr = long_lr
        self.short = deque(maxlen=short_len)
        self.peak_stats = RunningMedian(stats_window)
        self.id_stats = RunningMedian(stats_window)

        self.anchor: Optional[torch.Tensor] = None   # (1, D, kh, kw)
        self.long: Optional[torch.Tensor] = None
        self.patch_weights: Optional[torch.Tensor] = None  # (1, 1, kh, kw)
        self.num_updates = 0

    # ------------------------------------------------------------------
    def initialize(self, kernel: torch.Tensor, patch_weights: torch.Tensor):
        """Set the anchor from the first frame.

        Args:
            kernel: (1, D, kh, kw) L2-normalised features around the target.
            patch_weights: (1, 1, kh, kw) non-negative per-patch weights
                (foreground prior); background/context patches get low weight.
        """
        self.anchor = _renorm(kernel.clone())
        self.long = self.anchor.clone()
        self.short.clear()
        self.short.append(self.anchor.clone())
        self.patch_weights = patch_weights.clamp(min=0)
        self.peak_stats.values.clear()
        self.id_stats.values.clear()
        self.num_updates = 0

    @property
    def kernel_size(self) -> Tuple[int, int]:
        return tuple(self.anchor.shape[-2:])

    def combined_kernel(self) -> torch.Tensor:
        """Weighted sum of anchor, long-term and mean short-term kernels."""
        short_mean = _renorm(torch.stack(list(self.short), 0).mean(0))
        k = (self.w_anchor * self.anchor + self.w_long * self.long
             + self.w_short * short_mean)
        return k / (self.w_anchor + self.w_long + self.w_short)

    def kernels(self) -> dict:
        """All memory kernels (for debugging/visualisation)."""
        return {
            'anchor': self.anchor,
            'long-term': self.long,
            'short-term': _renorm(torch.stack(list(self.short), 0).mean(0)),
            'combined': self.combined_kernel(),
        }

    # ------------------------------------------------------------------
    def correlate(self, search_feats: torch.Tensor) -> torch.Tensor:
        """Weighted mean cosine similarity of the memory kernel over the search grid.

        Args:
            search_feats: (S, D, Hs, Ws) L2-normalised features (S = number of scales)

        Returns:
            (S, 1, Hs-kh+1, Ws-kw+1) response maps
        """
        k = self.combined_kernel() * self.patch_weights          # (1, D, kh, kw)
        resp = F.conv2d(search_feats, k)                          # (S, 1, Ho, Wo)
        return resp / self.patch_weights.sum().clamp(min=1e-6)

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Weighted mean per-patch cosine similarity between two kernels."""
        cos = (_renorm(a) * _renorm(b)).sum(1, keepdim=True)    # (1, 1, kh, kw)
        return float((cos * self.patch_weights).sum() / self.patch_weights.sum().clamp(min=1e-6))

    # ------------------------------------------------------------------
    def assess(self, peak: float, observation: torch.Tensor) -> dict:
        """Compute relative quality of the current frame (does not modify memory).

        Args:
            peak: raw (unwindowed) correlation peak of this frame
            observation: (1, D, kh, kw) features sampled at the predicted target

        Returns:
            dict with q_rel, id_rel, id_sim
        """
        id_sim = self.similarity(observation, self.anchor)
        q_ref = self.peak_stats.median(default=peak)
        id_ref = self.id_stats.median(default=id_sim)
        q_rel = peak / q_ref if q_ref > 1e-6 else 0.0
        id_rel = id_sim / id_ref if id_ref > 1e-6 else 0.0
        return {'q_rel': float(q_rel), 'id_rel': float(id_rel), 'id_sim': float(id_sim)}

    def record(self, peak: float, id_sim: float):
        """Push this frame's statistics into the running windows (all frames,
        so a permanent appearance change is eventually absorbed)."""
        self.peak_stats.push(peak)
        self.id_stats.push(id_sim)

    def update(self, observation: torch.Tensor, strength: float, push_short: bool):
        """Blend an accepted observation into the memory.

        Args:
            observation: (1, D, kh, kw) features at the accepted target location
            strength: in [0, 1], scales the long-term learning rate
            push_short: whether to append to the short-term ring buffer
        """
        if strength <= 0:
            return
        obs = _renorm(observation)
        lr = self.long_lr * float(np.clip(strength, 0.0, 1.0))
        self.long = _renorm((1 - lr) * self.long + lr * obs)
        if push_short:
            self.short.append(obs.clone())
        self.num_updates += 1
