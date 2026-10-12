# Copyright 2026 Dagimawi Moges Wakumo
# SPDX-License-Identifier: Apache-2.0
"""
SuperPoint + LightGlue keyframe memory anchored to the first frame.

Updating a geometric template naively makes box errors accumulate: each new
template inherits the (slightly wrong) box predicted in its own frame.
To avoid this, every keyframe stores the transform that maps its crop
coordinates back into the *anchor* (frame-1) crop. All correspondences are
expressed as (anchor coords -> current crop coords), so one similarity
transform is always fitted against the exact frame-1 box, no matter how
many keyframes contributed matches.

Crops are scale-normalised so the target always spans roughly
``target_norm_px`` pixels; this keeps SuperPoint usable on tiny UAV targets
and puts template and search at the same scale.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Any

import cv2
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


@dataclass
class _Keyframe:
    feats: Dict[str, torch.Tensor]   # SuperPoint feats restricted to the target box
    to_anchor: np.ndarray            # (2, 3) affine: this crop's coords -> anchor crop coords
    frame_idx: int


class KeypointMemory:
    """Anchor + recent keyframes for SuperPoint/LightGlue box refinement."""

    def __init__(self, max_keypoints: int = 1024, target_norm_px: float = 160.0,
                 crop_scale: float = 2.0, num_keyframes: int = 2, min_inliers: int = 8,
                 ransac_threshold: float = 3.0, box_margin: float = 0.1,
                 device: str = 'cuda'):
        try:
            from lightglue import LightGlue, SuperPoint
        except ImportError:
            raise ImportError(
                "LightGlue is not installed. Install with: "
                "pip install git+https://github.com/cvg/LightGlue.git"
            )
        self.device = device
        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
        self.matcher = LightGlue(features='superpoint').eval().to(device)

        self.target_norm_px = target_norm_px
        self.crop_scale = crop_scale
        self.min_inliers = min_inliers
        self.ransac_threshold = ransac_threshold
        self.box_margin = box_margin

        self.anchor: Optional[_Keyframe] = None
        self.anchor_box = None                  # (x, y, w, h) exact frame-1 box in anchor crop coords
        self.keyframes = deque(maxlen=num_keyframes)
        self.anchor_keypoints = 0
        self.anchor_norm_px = 0.0

    # ------------------------------------------------------------------
    # Crops and features
    # ------------------------------------------------------------------
    def _prepare_crop(self, frame: np.ndarray, center, size):
        """Scale-normalised context crop around (center, size).

        Returns:
            crop: (Hc, Wc, 3) BGR image
            params: (x0, y0, g) such that image = (x0, y0) + crop / g
            box: (x, y, w, h) of the target in crop coordinates
        """
        cx, cy = center
        w, h = size
        g = float(np.clip(self.target_norm_px / np.sqrt(max(w * h, 1.0)), 0.25, 8.0))
        cw, ch = self.crop_scale * w, self.crop_scale * h
        x0, y0 = cx - cw / 2, cy - ch / 2
        out_w, out_h = max(16, int(round(cw * g))), max(16, int(round(ch * g)))
        M = np.array([[g, 0, -g * x0], [0, g, -g * y0]], dtype=np.float64)
        crop = cv2.warpAffine(frame, M, (out_w, out_h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
        box = ((cx - x0) * g - w * g / 2, (cy - y0) * g - h * g / 2, w * g, h * g)
        return crop, (x0, y0, g), box

    @torch.no_grad()
    def _extract(self, crop: np.ndarray) -> Dict[str, torch.Tensor]:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        t = torch.from_numpy(gray).float().div(255.0)[None, None].to(self.device)
        return self.extractor.extract(t, resize=None)

    def _select_in_box(self, feats: Dict[str, torch.Tensor], box) -> Dict[str, torch.Tensor]:
        """Keep only keypoints inside the (slightly enlarged) target box."""
        x, y, w, h = box
        mx, my = self.box_margin * w, self.box_margin * h
        k = feats['keypoints'][0]
        keep = ((k[:, 0] >= x - mx) & (k[:, 0] <= x + w + mx)
                & (k[:, 1] >= y - my) & (k[:, 1] <= y + h + my))
        return {
            'keypoints': feats['keypoints'][:, keep],
            'keypoint_scores': feats['keypoint_scores'][:, keep],
            'descriptors': feats['descriptors'][:, keep],
            'image_size': feats['image_size'],
        }

    @staticmethod
    def _apply_affine(A: np.ndarray, pts: np.ndarray) -> np.ndarray:
        return pts @ A[:, :2].T + A[:, 2]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def initialize(self, frame: np.ndarray, bbox: tuple):
        x, y, w, h = bbox
        crop, _, box = self._prepare_crop(frame, (x + w / 2, y + h / 2), (w, h))
        feats = self._select_in_box(self._extract(crop), box)
        self.anchor = _Keyframe(feats=feats, to_anchor=np.array([[1, 0, 0], [0, 1, 0]], np.float64),
                                frame_idx=0)
        self.anchor_box = box
        self.keyframes.clear()
        n = feats['keypoints'].shape[1]
        self.anchor_keypoints = int(n)
        self.anchor_norm_px = float(np.sqrt(max(box[2] * box[3], 0.0)))   # target size in the normalised crop
        if n < self.min_inliers:
            logger.warning(f"Only {n} SuperPoint keypoints on the target at init; "
                           f"fine stage will rarely succeed for this sequence.")

    @property
    def num_keyframes(self) -> int:
        return len(self.keyframes)

    @torch.no_grad()
    def match(self, frame: np.ndarray, center, size) -> Dict[str, Any]:
        """Match the current crop against anchor + keyframes and fit one similarity
        transform from anchor coordinates to the current crop.

        Returns dict with: success, center, size (image coords), num_matches,
        num_inliers, inlier_ratio, plus private fields used by add_keyframe().
        """
        fail = {'success': False, 'num_matches': 0, 'num_inliers': 0, 'inlier_ratio': 0.0,
                'num_keypoints': 0, 'fail_reason': 'no_anchor'}
        if self.anchor is None:
            return fail

        crop, (x0, y0, g), _ = self._prepare_crop(frame, center, size)
        cur = self._extract(crop)
        fail['num_keypoints'] = int(cur['keypoints'].shape[1])
        fail['fail_reason'] = 'few_keypoints'
        if cur['keypoints'].shape[1] < self.min_inliers:
            return fail

        src_all, dst_all = [], []
        for kf in [self.anchor, *self.keyframes]:
            if kf.feats['keypoints'].shape[1] < 2:
                continue
            res = self.matcher({'image0': kf.feats, 'image1': cur})
            m = res['matches'][0]
            if m.shape[0] == 0:
                continue
            src = kf.feats['keypoints'][0][m[:, 0]].cpu().numpy().astype(np.float64)
            dst = cur['keypoints'][0][m[:, 1]].cpu().numpy().astype(np.float64)
            src_all.append(self._apply_affine(kf.to_anchor, src))
            dst_all.append(dst)

        fail['fail_reason'] = 'few_matches'
        if not src_all:
            return fail
        src = np.concatenate(src_all).astype(np.float32)
        dst = np.concatenate(dst_all).astype(np.float32)
        n = len(src)
        fail['num_matches'] = n
        if n < self.min_inliers:
            return fail

        M, inl = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_threshold,
            maxIters=2000, confidence=0.999, refineIters=10)
        fail['fail_reason'] = 'ransac'
        if M is None or inl is None:
            return fail
        num_inliers = int(inl.sum())
        fail['num_inliers'] = num_inliers
        fail['inlier_ratio'] = num_inliers / n
        fail['fail_reason'] = 'few_inliers'
        if num_inliers < self.min_inliers:
            return fail

        # Similarity scale only -> no axis-aligned inflation under rotation
        s = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
        ax, ay, aw, ah = self.anchor_box
        c_crop = self._apply_affine(M, np.array([[ax + aw / 2, ay + ah / 2]]))[0]
        w_crop, h_crop = aw * s, ah * s

        return {
            'success': True,
            'center': (x0 + c_crop[0] / g, y0 + c_crop[1] / g),
            'size': (w_crop / g, h_crop / g),
            'scale_vs_anchor': s,
            'num_matches': n,
            'num_inliers': num_inliers,
            'inlier_ratio': num_inliers / n,
            'num_keypoints': int(cur['keypoints'].shape[1]),
            'fail_reason': '',
            '_M': M, '_cur': cur,
            '_box_crop': (c_crop[0] - w_crop / 2, c_crop[1] - h_crop / 2, w_crop, h_crop),
        }

    def add_keyframe(self, result: Dict[str, Any], frame_idx: int) -> bool:
        """Store the current crop as a keyframe (call only for agreed, high-quality frames)."""
        if not result.get('success'):
            return False
        feats = self._select_in_box(result['_cur'], result['_box_crop'])
        if feats['keypoints'].shape[1] < self.min_inliers:
            return False
        to_anchor = cv2.invertAffineTransform(result['_M']).astype(np.float64)
        self.keyframes.append(_Keyframe(feats=feats, to_anchor=to_anchor, frame_idx=frame_idx))
        return True
