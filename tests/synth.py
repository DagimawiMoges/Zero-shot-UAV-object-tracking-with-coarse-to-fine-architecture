# Copyright 2026 Dagimawi Moges Wakumo
# SPDX-License-Identifier: Apache-2.0
"""Fake DINO backbone + synthetic UAV-like sequences for CPU tests (no GPU, no model download)."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class FakeBackbone:
    """Deterministic patch features: pooled colour/gradient stats per 16px patch plus a
    3x3 neighbourhood, randomly projected and L2-normalised. Mimics DINO's interface."""
    patch_size = 16

    def __init__(self, dim=96, seed=0):
        self.device = torch.device('cpu')
        g = torch.Generator().manual_seed(seed)
        self.proj = torch.randn(dim, 48 + 27, generator=g)

    def preprocess(self, image, is_bgr=True):
        img = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        H, W = img.shape[-2:]
        ph, pw = (-H) % 16, (-W) % 16
        return F.pad(img, (0, pw, 0, ph))

    def extract_features(self, batch, layers=None):
        x = batch - batch.mean((2, 3), keepdim=True)
        B, _, H, W = x.shape
        h, w = H // 16, W // 16
        cells = F.avg_pool2d(x, 4)                                       # (B,3,H/4,W/4)
        sub = F.unfold(cells, 4, stride=4).view(B, 48, h, w)              # 4x4 cells per patch
        mean = F.avg_pool2d(x, 16)
        nb = F.unfold(F.pad(mean, (1, 1, 1, 1), mode='replicate'), 3).view(B, 27, h, w)
        f = torch.einsum('dk,bkhw->bdhw', self.proj, torch.cat([sub, 0.5 * nb], 1))
        return {'deep': F.normalize(f, dim=1)}


def _texture(h, w, rng, blur=3, corners=True):
    t = rng.integers(0, 255, (h // 4 + 1, w // 4 + 1, 3)).astype(np.uint8)
    t = cv2.resize(t, (w, h), interpolation=cv2.INTER_CUBIC)
    t = cv2.GaussianBlur(t, (0, 0), blur)
    if corners:   # high-contrast blobs/rectangles so SuperPoint has corners
        for _ in range(max(4, h * w // 600)):
            x, y = rng.integers(0, w), rng.integers(0, h)
            s = rng.integers(3, max(4, min(h, w) // 5))
            col = tuple(int(c) for c in rng.integers(0, 255, 3))
            if rng.random() < 0.5:
                cv2.rectangle(t, (int(x), int(y)), (int(x + s), int(y + s)), col, -1)
            else:
                cv2.circle(t, (int(x), int(y)), int(s // 2), col, -1)
    return t


def make_sequence(n=80, W=640, H=360, size0=(60, 40), scale_end=1.6, seed=1, noise=4.0, occl=None):
    """Target texture moving on a curve over a textured, slowly panning background,
    growing from size0 to size0*scale_end. Returns frames (BGR) and GT boxes (x,y,w,h)."""
    rng = np.random.default_rng(seed)
    bg = (0.6 * _texture(H + 200, W + 200, rng, blur=6, corners=False) + 50).astype(np.uint8)
    tex = _texture(160, int(160 * size0[0] / size0[1]), rng, blur=1.5, corners=True)
    frames, gts = [], []
    for t in range(n):
        u = t / max(n - 1, 1)
        s = scale_end ** u
        w, h = size0[0] * s, size0[1] * s
        cx = W * 0.3 + W * 0.4 * u + 25 * np.sin(6 * u)
        cy = H * 0.5 + 40 * np.sin(4 * u)
        ox, oy = int(100 + 60 * np.sin(2 * u)), int(100 + 30 * u)   # camera pan
        f = bg[oy:oy + H, ox:ox + W].copy()
        th_, tw_ = tex.shape[:2]
        M = np.array([[w / tw_, 0, cx - w / 2], [0, h / th_, cy - h / 2]])
        warped = cv2.warpAffine(tex, M, (W, H), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(np.ones((th_, tw_), np.uint8), M, (W, H), flags=cv2.INTER_NEAREST)
        if not (occl and occl[0] <= t < occl[1]):
            f[mask > 0] = warped[mask > 0]
        f = np.clip(f.astype(np.float32) + rng.normal(0, noise, f.shape), 0, 255).astype(np.uint8)
        frames.append(f)
        gts.append((cx - w / 2, cy - h / 2, w, h))
    return frames, np.array(gts)
