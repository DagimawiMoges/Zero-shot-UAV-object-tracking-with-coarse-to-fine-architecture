# Copyright 2026 Dagimawi Moges Wakumo
# SPDX-License-Identifier: Apache-2.0
"""
KeypointTracker: SuperPoint + LightGlue only, no DINO (the "fine-only" baseline).

The ablation counterpart of AdaptiveTracker. It isolates what the geometric stage
does on its own, without the DINOv3 coarse stage telling it where to look. Everything
it shares with Blihat is the same code with the same frozen settings (the
``adaptive`` section of the config, e.g. configs/blihat_final.yaml):

  - KeypointMemory: frame-1 anchor + keyframes, all mapped to frame-1 coordinates,
    one RANSAC similarity transform fitted to the exact frame-1 box;
  - fine_fit_reliable(): the same reliability gate (inliers, inlier ratio, scale
    lock, center offset);
  - the same motion prior (velocity smoothing, coasting, decay) and keyframe interval.

What replaces DINO:

  - Search center: the motion-predicted center (DINO center in Blihat).
  - Fallback when the fit is unreliable: hold the box at the predicted center with
    the previous size (DINO box in Blihat).
  - Never lost: after widen_patience bad frames the match crop grows by widen_factor
    per frame, up to max_search_scale target sizes - the same search extent Blihat's
    coarse stage widens to. The allowed center offset grows with the crop.
  - Keyframes: added from reliable fits every keyframe_interval frames. Blihat also
    requires a healthy DINO frame; there is no such signal here.
  - No tiled re-detection (it is DINO-based in Blihat).

Per-frame diagnostics use AdaptiveDiagnostics (box_source 'fine' or 'hold', fine_reject
says why a fit was not used); the DINO fields stay at their defaults. Score results with
eval/analyze_uav123.py like any other run.
"""

import logging
import time
from collections import Counter, deque
from typing import Any, Dict, List, Optional

import numpy as np

from blihat.adaptive_tracker import AdaptiveDiagnostics, fine_fit_reliable
from blihat.config import Config

logger = logging.getLogger(__name__)


class KeypointTracker:
    """SuperPoint + LightGlue tracker without a DINO stage (ablation baseline)."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.default()
        self.cfg = self.config.adaptive
        cfg = self.cfg
        from blihat.fine.keyframes import KeypointMemory   # raises if LightGlue is missing
        self.kp_memory = KeypointMemory(
            max_keypoints=cfg.max_keypoints, target_norm_px=cfg.target_norm_px,
            crop_scale=cfg.fine_crop_scale, num_keyframes=cfg.num_keyframes,
            min_inliers=cfg.min_inliers, ransac_threshold=cfg.ransac_threshold,
            device=self.config.device)
        # Widening limit as a multiple of the normal match crop: the same search extent
        # (in target sizes) that Blihat's coarse stage widens to.
        self.max_widen = max(1.0, cfg.max_search_scale / cfg.fine_crop_scale)
        self.initialized = False
        self.diagnostics: List[AdaptiveDiagnostics] = []

    def init(self, frame: np.ndarray, bbox: tuple):
        cfg = self.cfg
        x, y, w, h = [float(v) for v in bbox]
        self.target_pos = (x + w / 2, y + h / 2)
        self.target_size = (w, h)
        self.velocity = deque(maxlen=cfg.velocity_smoothing)
        self.widen = 1.0
        self.low_count = 0
        self.frame_idx = 0
        self.last_keyframe = 0
        self.frames_since_fine = 0
        self.kp_memory.initialize(frame, (x, y, w, h))
        self.initialized = True
        self.diagnostics = [AdaptiveDiagnostics(frame_idx=0, bbox=(x, y, w, h), branch='init',
                                                overall_confidence=1.0, reason='Initialization')]

    def track(self, frame: np.ndarray) -> Dict[str, Any]:
        if not self.initialized:
            raise RuntimeError("Tracker not initialized. Call init() first.")
        cfg = self.cfg
        t0 = time.perf_counter()
        self.frame_idx += 1
        img_h, img_w = frame.shape[:2]
        tw, th = self.target_size

        # 1. Motion prior (same rule as AdaptiveTracker)
        pred = self.target_pos
        coasting_ok = self.low_count == 0 or self.low_count <= cfg.motion_coast_frames
        if coasting_ok and len(self.velocity) >= max(1, cfg.motion_min_history):
            v = np.mean(self.velocity, axis=0) * (cfg.velocity_decay ** self.low_count)
            pred = (pred[0] + v[0], pred[1] + v[1])

        # 2. Match around the prediction; the crop is widened while the target is not found
        crop_used = cfg.fine_crop_scale * self.widen
        self.kp_memory.crop_scale = crop_used
        fr = self.kp_memory.match(frame, pred, (tw, th))
        self.kp_memory.crop_scale = cfg.fine_crop_scale

        # 3. Same reliability gate as Blihat; the offset limit scales with the crop
        ok, reject = fine_fit_reliable(cfg, fr, pred, (tw, th), self.frames_since_fine,
                                       max_offset=cfg.fine_max_offset * self.widen)

        kf_added = False
        if ok:
            center = fr['center']
            a = cfg.fine_box_size_lr
            size = (tw * (fr['size'][0] / tw) ** a, th * (fr['size'][1] / th) ** a)
            branch, box_source, reason = 'fine', 'fine', 'Fine box'
            if self.frame_idx - self.last_keyframe >= cfg.keyframe_interval:
                kf_added = self.kp_memory.add_keyframe(fr, self.frame_idx)
                if kf_added:
                    self.last_keyframe = self.frame_idx
            self.velocity.append((center[0] - self.target_pos[0], center[1] - self.target_pos[1]))
            self.low_count = 0
            self.widen = max(1.0, self.widen / cfg.widen_factor)
            self.frames_since_fine = 0
        else:
            center, size = pred, (tw, th)
            branch, box_source, reason = 'hold', 'hold', f'Hold (fine: {reject})'
            self.low_count += 1
            if self.low_count >= cfg.widen_patience:
                self.widen = min(self.max_widen, self.widen * cfg.widen_factor)
            if self.low_count > cfg.motion_coast_frames:
                self.velocity.clear()
            self.frames_since_fine += 1

        cx = float(np.clip(center[0], 0, img_w - 1))
        cy = float(np.clip(center[1], 0, img_h - 1))
        w = float(np.clip(size[0], 4.0, img_w))
        h = float(np.clip(size[1], 4.0, img_h))
        self.target_pos, self.target_size = (cx, cy), (w, h)
        out = self._clip_bbox((cx - w / 2, cy - h / 2, w, h), img_w, img_h)
        conf = float(fr.get('inlier_ratio', 0.0)) if ok else 0.0

        fine_box = ((fr['center'][0] - fr['size'][0] / 2, fr['center'][1] - fr['size'][1] / 2,
                     fr['size'][0], fr['size'][1]) if fr['success'] else (0.0, 0.0, 0.0, 0.0))
        diag = AdaptiveDiagnostics(
            frame_idx=self.frame_idx, bbox=out, branch=branch,
            search_scale=crop_used,
            fine_success=bool(fr['success']), fine_num_matches=int(fr.get('num_matches', 0)),
            fine_num_inliers=int(fr.get('num_inliers', 0)), keyframe_added=kf_added,
            overall_confidence=conf, reason=reason, elapsed_ms=(time.perf_counter() - t0) * 1000,
            box_source=box_source, fine_inlier_ratio=float(fr.get('inlier_ratio', 0.0)),
            fine_num_keypoints=int(fr.get('num_keypoints', 0)), fine_reject=reject,
            fine_box=tuple(float(v) for v in fine_box))
        self.diagnostics.append(diag)
        return {'bbox': out, 'confidence': conf, 'branch': branch, 'diagnostics': diag,
                'heatmap': None, 'search_bbox': None, 'debug': None}

    @staticmethod
    def _clip_bbox(bbox, img_w, img_h):
        x, y, w, h = bbox
        x = max(0.0, min(x, img_w - 1.0))
        y = max(0.0, min(y, img_h - 1.0))
        return (x, y, max(1.0, min(w, img_w - x)), max(1.0, min(h, img_h - y)))

    def get_diagnostics_summary(self) -> Dict[str, Any]:
        if not self.diagnostics:
            return {}
        d = self.diagnostics[1:]
        times = [x.elapsed_ms for x in d]
        n = max(1, len(d))
        return {
            'total_frames': len(self.diagnostics),
            'branch_counts': dict(Counter(x.branch for x in self.diagnostics)),
            'avg_fps': 1000.0 / np.mean(times) if times else 0,
            'avg_ms': float(np.mean(times)) if times else 0,
            'median_ms': float(np.median(times)) if times else 0,
            'p95_ms': float(np.percentile(times, 95)) if times else 0,
            'fine_box_ratio': sum(x.box_source == 'fine' for x in d) / n,
            'fine_success_ratio': sum(x.fine_success for x in d) / n,
            'keyframes_added': sum(x.keyframe_added for x in d),
            'fine_reject_counts': dict(Counter(x.fine_reject for x in d if x.box_source == 'hold')),
            'fine_anchor_keypoints': int(self.kp_memory.anchor_keypoints),
            'fine_anchor_norm_px': float(self.kp_memory.anchor_norm_px),
            'box_source': 'keypoint',
        }
