# Copyright 2026 Dagimawi Moges Wakumo
# SPDX-License-Identifier: Apache-2.0
"""
Per-frame debug dumps: the template being searched and the DINO heatmap.

Both trackers fill a ``debug`` dict in their ``track()`` result when
``tracker.debug = True``. The dict is captured *at query time*, i.e. it shows
exactly the template/kernel that was correlated with this frame (before any
memory update) and the response it produced.

debug dict schema
-----------------
    frame_idx      int
    template_images list[(label, BGR uint8)]    image crops behind the template
    kernels        list[(label, (D, h, w) float)] feature kernels; the last one is
                                                  the kernel actually correlated
    patch_weights  (h, w) float or None          per-patch correlation weights
    search_crops   list[BGR uint8]                search crop per scale (model input)
    scale_labels   list[str]
    responses      (S, Ho, Wo) float              raw response (mean cosine similarity)
    scored         (S, Ho, Wo) float              after window / scale penalty
    best           (si, r, c)                     argmax over scored
    sub            (sub_r, sub_c)                 sub-patch peak
    offset_patches float                          cell (r, c) <-> crop pixel ((c+o)*p, (r+o)*p)
    patch_px       int                            patch size in crop pixels
    target_wh_crop (w, h)                         current target size in crop pixels
    info           dict[str, Any]                 scalars printed on the panel

DebugRecorder writes, per frame:
    <out>/template/000123.png   template row (crops + PCA of kernels + weights)
    <out>/heatmap/000123.png    heatmap row (overlay on search crop + raw maps)
    <out>/raw/000123.npz        raw arrays (optional)
and optionally a stacked debug video.
"""

import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

TILE = 192          # display size of every tile (px)
LABEL_H = 22
BG = (24, 24, 24)
FG = (235, 235, 235)


# ──────────────────────────────────────────────────────────────────────
# Tiles
# ──────────────────────────────────────────────────────────────────────
def _labeled(img: np.ndarray, label: str, size: int = TILE) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    tile = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST
                      if min(img.shape[:2]) < 64 else cv2.INTER_LINEAR)
    bar = np.full((LABEL_H, size, 3), BG, np.uint8)
    cv2.putText(bar, label[:28], (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, FG, 1, cv2.LINE_AA)
    return np.vstack([bar, tile])


def _row(tiles: List[np.ndarray], title: str) -> np.ndarray:
    gap = np.full((tiles[0].shape[0], 6, 3), BG, np.uint8)
    parts = []
    for t in tiles:
        parts += [t, gap]
    body = np.hstack(parts[:-1])
    head = np.full((26, body.shape[1], 3), BG, np.uint8)
    cv2.putText(head, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, FG, 1, cv2.LINE_AA)
    return np.vstack([head, body])


def pca_rgb(kernels: List[np.ndarray]) -> List[np.ndarray]:
    """Project several (D, h, w) kernels onto one shared 3-component PCA basis.

    Sharing the basis means equal colours = equal features across tiles, so you
    can see how far long/short-term memory have moved from the anchor.
    """
    flat = [k.reshape(k.shape[0], -1).T for k in kernels]          # (h*w, D)
    X = np.concatenate(flat, 0).astype(np.float64)
    mu = X.mean(0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(X - mu, full_matrices=False)
        basis = vt[:3].T
    except np.linalg.LinAlgError:
        basis = np.eye(X.shape[1])[:, :3]
    proj = (X - mu) @ basis
    lo, hi = np.percentile(proj, 2, 0), np.percentile(proj, 98, 0)
    out, i = [], 0
    for k in kernels:
        n = k.shape[1] * k.shape[2]
        p = np.clip((proj[i:i + n] - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
        out.append((p.reshape(k.shape[1], k.shape[2], 3) * 255).astype(np.uint8)[..., ::-1])
        i += n
    return out


def _colormap(m: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    n = np.clip((m - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    return cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_JET)


# ──────────────────────────────────────────────────────────────────────
# Panels
# ──────────────────────────────────────────────────────────────────────
def render_template_row(dbg: Dict[str, Any]) -> np.ndarray:
    tiles = [_labeled(img, lbl) for lbl, img in dbg['template_images']]
    if dbg.get('kernels'):
        labels = [lbl for lbl, _ in dbg['kernels']]
        for lbl, img in zip(labels, pca_rgb([k for _, k in dbg['kernels']])):
            tiles.append(_labeled(img, f"PCA {lbl}"))
    if dbg.get('patch_weights') is not None:
        w = dbg['patch_weights']
        tiles.append(_labeled(_colormap(w, 0, max(float(w.max()), 1e-6)), "patch weights"))
    return _row(tiles, f"Frame {dbg['frame_idx']}  |  template queried (before update)")


def render_heatmap_row(dbg: Dict[str, Any]) -> np.ndarray:
    si, r, c = dbg['best']
    sub_r, sub_c = dbg['sub']
    p, o = dbg['patch_px'], dbg['offset_patches']
    crop = dbg['search_crops'][si].copy()
    raw = dbg['responses']
    Ho, Wo = raw.shape[1:]
    vmin, vmax = float(raw.min()), float(raw.max())

    # Overlay: cell (r, c) covers crop pixels centred on ((c+o)*p, (r+o)*p)
    x0, y0 = int(round((o - 0.5) * p)), int(round((o - 0.5) * p))
    hm = cv2.resize(_colormap(dbg['scored'][si], float(dbg['scored'][si].min()), float(dbg['scored'][si].max())),
                    (Wo * p, Ho * p), interpolation=cv2.INTER_LINEAR)
    H, W = crop.shape[:2]
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + Wo * p), min(H, y0 + Ho * p)
    if xb > xa and yb > ya:
        crop[ya:yb, xa:xb] = cv2.addWeighted(crop[ya:yb, xa:xb], 0.45,
                                             hm[ya - y0:yb - y0, xa - x0:xb - x0], 0.55, 0)
    px, py = (sub_c + o) * p, (sub_r + o) * p
    tw, th = dbg['target_wh_crop']
    cv2.rectangle(crop, (int(px - tw / 2), int(py - th / 2)), (int(px + tw / 2), int(py + th / 2)),
                  (255, 255, 255), 1, cv2.LINE_AA)
    cv2.drawMarker(crop, (int(px), int(py)), (255, 255, 255), cv2.MARKER_CROSS, 12, 2)
    cv2.rectangle(crop, (xa, ya), (xb - 1, yb - 1), (200, 200, 200), 1)
    tiles = [_labeled(crop, f"search x{dbg['scale_labels'][si]} + heatmap", TILE * 2 + LABEL_H)]

    small = []
    for i in range(raw.shape[0]):
        m = _colormap(raw[i], vmin, vmax)
        m = cv2.resize(m, (TILE, TILE), interpolation=cv2.INTER_NEAREST)
        if i == si:
            cc, rr = (sub_c + 0.5) * TILE / Wo, (sub_r + 0.5) * TILE / Ho
            cv2.drawMarker(m, (int(cc), int(rr)), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
            cv2.rectangle(m, (0, 0), (TILE - 1, TILE - 1), (255, 255, 255), 2)
        small.append(_labeled(m, f"raw x{dbg['scale_labels'][i]} [{vmin:.2f},{vmax:.2f}]"))
    # stack raw maps 2 per column to keep the row compact
    while len(small) % 2:
        small.append(np.full_like(small[0], BG))
    cols = [np.vstack([small[j], small[j + 1]]) for j in range(0, len(small), 2)]
    pad = tiles[0].shape[0] - cols[0].shape[0]
    cols = [np.vstack([c_, np.full((pad, c_.shape[1], 3), BG, np.uint8)]) if pad > 0 else c_ for c_ in cols]
    tiles += cols

    info = np.full((tiles[0].shape[0], 330, 3), BG, np.uint8)
    for k, (key, val) in enumerate(dbg.get('info', {}).items()):
        txt = f"{key}: {val:.3f}" if isinstance(val, float) else f"{key}: {val}"
        cv2.putText(info, txt, (6, 20 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, FG, 1, cv2.LINE_AA)
    tiles.append(info)
    return _row(tiles, "DINOv3 response right after the query")


def render_debug_panel(dbg: Dict[str, Any]) -> np.ndarray:
    top, bottom = render_template_row(dbg), render_heatmap_row(dbg)
    w = max(top.shape[1], bottom.shape[1])
    pad = lambda im: np.hstack([im, np.full((im.shape[0], w - im.shape[1], 3), BG, np.uint8)]) \
        if im.shape[1] < w else im
    return np.vstack([pad(top), pad(bottom)])


# ──────────────────────────────────────────────────────────────────────
# Recorder
# ──────────────────────────────────────────────────────────────────────
class DebugRecorder:
    """Saves the template and heatmap of every frame (and optionally a video)."""

    def __init__(self, out_dir: str, save_raw: bool = False,
                 video_path: Optional[str] = None, fps: float = 30.0):
        self.out_dir = out_dir
        self.save_raw = save_raw
        for sub in ('template', 'heatmap') + (('raw',) if save_raw else ()):
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        self.video_path, self.fps = video_path, fps
        self._writer = None
        self._size = None

    def write(self, dbg: Optional[Dict[str, Any]]):
        if not dbg:
            return
        i = dbg['frame_idx']
        top, bottom = render_template_row(dbg), render_heatmap_row(dbg)
        cv2.imwrite(os.path.join(self.out_dir, 'template', f'{i:06d}.png'), top)
        cv2.imwrite(os.path.join(self.out_dir, 'heatmap', f'{i:06d}.png'), bottom)
        if self.save_raw:
            np.savez_compressed(
                os.path.join(self.out_dir, 'raw', f'{i:06d}.npz'),
                responses=dbg['responses'], scored=dbg['scored'], best=np.array(dbg['best']),
                sub=np.array(dbg['sub']),
                **{f"kernel_{lbl.replace(' ', '_')}": k for lbl, k in dbg.get('kernels', [])},
                **({'patch_weights': dbg['patch_weights']} if dbg.get('patch_weights') is not None else {}))
        if self.video_path:
            panel = render_debug_panel(dbg)
            if self._writer is None:
                self._size = (panel.shape[1], panel.shape[0])
                self._writer = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                               self.fps, self._size)
            if (panel.shape[1], panel.shape[0]) != self._size:
                panel = cv2.resize(panel, self._size)
            self._writer.write(panel)

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
