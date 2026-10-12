# Copyright 2026 Dagimawi Moges Wakumo
# SPDX-License-Identifier: Apache-2.0
"""
AdaptiveTracker: never-lost UAV tracker with dynamic DINO memory and
frame-1-anchored SuperPoint/LightGlue keyframes.

Per frame:
  1. Motion prior -> predicted center (disabled while quality is low).
  2. DINO search crops at 3 scales (batched) -> weighted correlation against the
     memory kernel (anchor + long-term + short-term) -> peak, scale, center.
  3. Relative quality of the frame: q_rel (peak vs its running median) and
     id_rel (similarity to the anchor vs its running median).
  4. Fine stage: match a scale-normalised crop against the anchor + keyframes,
     fit one similarity transform to the exact frame-1 box.
  5. Box source (cfg.box_source):
        "agree" (default, original): the fine box is used only if its center/size
                 agree with DINO.
        "fine":  DINO says where, LightGlue gives the box. The fine box is used
                 whenever the fit passes a reliability check (inliers, inlier ratio,
                 per-frame scale step, offset from the DINO center); otherwise the
                 DINO box is the fallback. Not a mutual verification.
        "dino":  the fine stage runs for diagnostics only (both boxes are logged);
                 output and memory are exactly those of a coarse-only run.
  6. Memory updates ("agree"):
        - both agree               -> update DINO memory fully, maybe add a keyframe
        - fine disagrees with DINO -> update nothing
        - no fine verdict          -> DINO-only update at reduced strength
     ("fine"): reliable fit -> update with the observation at the fine box, maybe add
     a keyframe; fallback -> DINO-only update exactly as without a fine stage.
     Keyframes are only ever added from reliable fits on DINO-healthy frames.
  7. Never lost: after several low-quality frames the search area widens
     instead of declaring the target lost.
  8. Optional re-detection (cfg.redetect): while quality stays low, search a
     grid of tiles around the last confident position and jump there if a
     match is as good as the target's healthy appearance.
"""

import logging
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from blihat.backbone.dinov2 import DINOBackbone
from blihat.coarse.memory import FeatureMemory, RunningMedian
from blihat.config import Config

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveDiagnostics:
    frame_idx: int = 0
    bbox: tuple = (0, 0, 0, 0)
    branch: str = "init"          # 'fine' | 'coarse'
    peak: float = 0.0
    q_rel: float = 0.0
    id_rel: float = 0.0
    psr: float = 0.0
    scale: float = 1.0
    search_scale: float = 0.0
    fine_success: bool = False
    fine_agree: bool = False
    fine_num_matches: int = 0
    fine_num_inliers: int = 0
    memory_strength: float = 0.0
    keyframe_added: bool = False
    overall_confidence: float = 0.0
    reason: str = ""
    elapsed_ms: float = 0.0
    # Re-detection attempt on this frame (if any)
    redetect_tried: bool = False
    redetect_accepted: bool = False
    redetect_peak: float = 0.0
    redetect_peak_thr: float = 0.0
    redetect_id: float = 0.0
    redetect_id_thr: float = 0.0
    redetect_reject: str = ""     # '' | 'no_history' | 'peak' | 'identity' | 'not_better' | 'fine'
    # Box source / fine-stage reliability (all modes; filled when the fine stage runs)
    box_source: str = "dino"      # where the output box came from: 'dino' | 'fine' | 'redetect' | 'redetect_fine'
    fine_inlier_ratio: float = 0.0
    fine_num_keypoints: int = 0   # SuperPoint keypoints in the current crop
    fine_reject: str = ""         # why the fit was not used: 'disabled' | matcher failure
                                  #   ('few_keypoints' | 'few_matches' | 'ransac' | 'few_inliers')
                                  #   | 'small' | 'inliers' | 'ratio' | 'scale_jump' | 'offset' | 'disagree' ('agree' mode)
    dino_fine_dist: float = float('nan')        # |fine - DINO center| in target sizes
    dino_fine_size_ratio: float = float('nan')  # sqrt(fine area / DINO area)
    dino_box: tuple = (0, 0, 0, 0)              # DINO box (x, y, w, h) for this frame
    fine_box: tuple = (0, 0, 0, 0)              # LightGlue box, zeros if no fit


def _crop_resize(frame: np.ndarray, center, region_wh, out_px: int) -> np.ndarray:
    """Crop region_wh around center and resample to out_px x out_px (edge-replicated)."""
    cx, cy = center
    rw, rh = max(region_wh[0], 1.0), max(region_wh[1], 1.0)
    gx, gy = out_px / rw, out_px / rh
    M = np.array([[gx, 0, -gx * (cx - rw / 2)], [0, gy, -gy * (cy - rh / 2)]], dtype=np.float64)
    return cv2.warpAffine(frame, M, (out_px, out_px), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


class AdaptiveTracker:
    """Never-lost tracker with dynamic DINO template memory and keyframe geometry."""

    def __init__(self, config: Optional[Config] = None, backbone: Optional[DINOBackbone] = None):
        self.config = config or Config.default()
        self.cfg = self.config.adaptive
        cfg = self.cfg

        logger.info(f"Initializing backbone: {self.config.backbone.name}")
        self.backbone = backbone or DINOBackbone(model_name=self.config.backbone.name,
                                                 device=self.config.device,
                                                 final_norm=self.config.backbone.final_norm,
                                                 dtype=self.config.backbone.dtype)
        self.patch = self.backbone.patch_size   # from the model, not the config

        # YAML / --set parse a bare `off` as False: normalise and validate the string options
        if cfg.redetect_fine is False or cfg.redetect_fine is None:
            cfg.redetect_fine = 'off'
        for key, allowed in (('box_source', ('agree', 'fine', 'dino')),
                             ('fine_memory_obs', ('search', 'template', 'dino')),
                             ('redetect_fine', ('off', 'refine', 'require'))):
            if getattr(cfg, key) not in allowed:
                raise ValueError(f"adaptive.{key} must be one of {allowed}, got {getattr(cfg, key)!r}")

        self.memory = FeatureMemory(weights=tuple(cfg.memory_weights), long_lr=cfg.long_lr,
                                    short_len=cfg.short_len, stats_window=cfg.stats_window)

        self.kp_memory = None
        if cfg.fine_enabled:
            try:
                from blihat.fine.keyframes import KeypointMemory
                self.kp_memory = KeypointMemory(
                    max_keypoints=cfg.max_keypoints, target_norm_px=cfg.target_norm_px,
                    crop_scale=cfg.fine_crop_scale, num_keyframes=cfg.num_keyframes,
                    min_inliers=cfg.min_inliers, ransac_threshold=cfg.ransac_threshold,
                    device=self.config.device)
            except ImportError as e:
                logger.warning(f"Fine stage disabled: {e}")

        self._bg_remover = None
        if cfg.fg_mask:
            try:
                from blihat.utils.background import BackgroundRemover
                self._bg_remover = BackgroundRemover(fill="zero")
            except Exception as e:  # pragma: no cover
                logger.warning(f"Foreground mask disabled: {e}")

        self.initialized = False
        self.diagnostics: List[AdaptiveDiagnostics] = []
        # Set True to get result['debug'] (template + heatmap at query time), see utils/debug_viz.py
        self.debug = False

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _search_patches(self, search_scale: float) -> int:
        """Search grid side so search and template share the same pixels-per-target."""
        return max(self.kh + 1, int(round(self.cfg.template_patches * search_scale / self.cfg.template_scale)))

    def _extract(self, crops: List[np.ndarray]) -> torch.Tensor:
        batch = torch.cat([self.backbone.preprocess(c, is_bgr=True) for c in crops], 0)
        return self.backbone.extract_features(batch)['deep']

    def _kernel_weights(self, crop: np.ndarray) -> torch.Tensor:
        """Per-patch weights: context_weight outside the target, (masked) 1 inside."""
        cfg = self.cfg
        n_t = cfg.template_patches
        half = n_t / (2 * cfg.template_scale)
        lo, hi = n_t / 2 - half, n_t / 2 + half
        idx = np.arange(self.k0, self.k1)
        ov = np.clip(np.minimum(idx + 1, hi) - np.maximum(idx, lo), 0, 1)   # overlap with target, per axis
        overlap = np.outer(ov, ov)                                           # (kh, kw)

        inside = np.ones_like(overlap)
        if self._bg_remover is not None:
            try:
                _, alpha = self._bg_remover.remove_background(crop)
                grid = cv2.resize(alpha, (n_t, n_t), interpolation=cv2.INTER_AREA)[self.k0:self.k1, self.k0:self.k1]
                if (grid * overlap).sum() / max(overlap.sum(), 1e-6) > 0.05:
                    inside = cfg.fg_mask_floor + (1 - cfg.fg_mask_floor) * grid
                else:
                    logger.info("Foreground mask empty on the target; using box prior only.")
            except Exception as e:
                logger.warning(f"Foreground mask failed ({e}); using box prior only.")

        w = cfg.context_weight * (1 - overlap) + overlap * inside
        return torch.from_numpy(w.astype(np.float32))[None, None].to(self.backbone.device)

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def init(self, frame: np.ndarray, bbox: tuple):
        cfg = self.cfg
        x, y, w, h = [float(v) for v in bbox]
        self.target_pos = (x + w / 2, y + h / 2)
        self.target_size = (w, h)
        self.velocity = deque(maxlen=cfg.velocity_smoothing)
        self.search_scale = cfg.search_scale
        self.low_count = 0
        self.frame_idx = 0
        self.last_keyframe = 0
        self.diagnostics = []
        self.last_good_pos = self.target_pos
        self.healthy_peak = RunningMedian(cfg.stats_window)
        self.healthy_id = RunningMedian(cfg.stats_window)
        self.redetect_attempts = 0
        self.frames_since_fine = 0

        n_t = cfg.template_patches
        half = n_t / (2 * cfg.template_scale)
        self.k0 = max(0, int(np.floor(n_t / 2 - half)) - cfg.kernel_margin)
        self.k1 = n_t - self.k0
        self.kh = self.k1 - self.k0
        self.offset = n_t / 2 - self.k0   # target center relative to kernel top-left, in patches

        t_crop = _crop_resize(frame, self.target_pos,
                              (w * cfg.template_scale, h * cfg.template_scale), n_t * self.patch)
        self.anchor_crop = t_crop
        self.short_crops = deque([t_crop], maxlen=cfg.short_len)   # display only
        feats = self._extract([t_crop])
        kernel = feats[:, :, self.k0:self.k1, self.k0:self.k1]
        self.memory.initialize(kernel, self._kernel_weights(t_crop))

        if self.kp_memory is not None:
            self.kp_memory.initialize(frame, (x, y, w, h))

        self.initialized = True
        self.diagnostics.append(AdaptiveDiagnostics(frame_idx=0, bbox=(x, y, w, h), branch='init',
                                                    overall_confidence=1.0, reason='Initialization'))

    # ------------------------------------------------------------------
    # Coarse stage
    # ------------------------------------------------------------------
    def _coarse(self, frame: np.ndarray, pred_center) -> Dict[str, Any]:
        cfg = self.cfg
        tw, th = self.target_size
        n_s = self._search_patches(self.search_scale)
        s_px = n_s * self.patch
        widened = self.search_scale > cfg.search_scale + 1e-6
        scales = [1.0] if (cfg.widen_single_scale and widened) else list(cfg.scales)

        crops = [_crop_resize(frame, pred_center,
                              (self.search_scale * tw * s, self.search_scale * th * s), s_px)
                 for s in scales]
        feats = self._extract(crops)                       # (S, D, n_s, n_s)
        raw = self.memory.correlate(feats)                 # (S, 1, Ho, Wo)
        S, _, Ho, Wo = raw.shape

        penalty = torch.tensor([1.0 if abs(s - 1.0) < 1e-6 else cfg.scale_penalty for s in scales],
                               device=raw.device).view(S, 1, 1, 1)
        win = (torch.hann_window(Ho, periodic=False, device=raw.device).view(-1, 1)
               * torch.hann_window(Wo, periodic=False, device=raw.device).view(1, -1))
        win = (1 - cfg.window_weight) + cfg.window_weight * win
        scored = raw * penalty * win

        flat_idx = int(torch.argmax(scored))
        si, rem = divmod(flat_idx, Ho * Wo)
        r, c = divmod(rem, Wo)
        resp = scored[si, 0]
        peak = float(raw[si, 0, r, c])

        # Sub-patch refinement (only with a real local maximum)
        sub_r, sub_c = float(r), float(c)
        if 0 < r < Ho - 1:
            d2 = float(resp[r + 1, c] + resp[r - 1, c] - 2 * resp[r, c])
            if d2 < -1e-6:
                sub_r += float(np.clip(-0.5 * float(resp[r + 1, c] - resp[r - 1, c]) / d2, -0.5, 0.5))
        if 0 < c < Wo - 1:
            d2 = float(resp[r, c + 1] + resp[r, c - 1] - 2 * resp[r, c])
            if d2 < -1e-6:
                sub_c += float(np.clip(-0.5 * float(resp[r, c + 1] - resp[r, c - 1]) / d2, -0.5, 0.5))

        # PSR with the peak neighbourhood excluded
        m = raw[si, 0].clone()
        m[max(0, r - 2):r + 3, max(0, c - 2):c + 3] = float('nan')
        side = m[~torch.isnan(m)]
        psr = float((peak - side.mean()) / (side.std() + 1e-6)) if side.numel() > 1 else 0.0

        s_best = scales[si]
        sw, sh = self.search_scale * tw * s_best, self.search_scale * th * s_best
        sx, sy = pred_center[0] - sw / 2, pred_center[1] - sh / 2
        cx = sx + (sub_c + self.offset) / n_s * sw
        cy = sy + (sub_r + self.offset) / n_s * sh

        observation = feats[si:si + 1, :, r:r + self.kh, c:c + self.kh]

        debug = None
        if self.debug:
            ks = self.memory.kernels()
            debug = {
                'template_images': [('anchor crop (frame 1)', self.anchor_crop),
                                    ('last short-term crop', self.short_crops[-1])],
                'kernels': [(k, v[0].float().cpu().numpy()) for k, v in ks.items()],
                'patch_weights': self.memory.patch_weights[0, 0].cpu().numpy(),
                'search_crops': crops,
                'scale_labels': [f"{s:.3f}" for s in scales],
                'responses': raw[:, 0].float().cpu().numpy(),
                'scored': scored[:, 0].float().cpu().numpy(),
                'best': (si, r, c), 'sub': (sub_r, sub_c),
                'offset_patches': float(self.offset), 'patch_px': int(self.patch),
                'target_wh_crop': (s_px / self.search_scale / s_best, s_px / self.search_scale / s_best),
            }
        heat_bbox = (sx + self.offset / n_s * sw, sy + self.offset / n_s * sh, Wo / n_s * sw, Ho / n_s * sh)
        return {
            'center': (cx, cy), 'peak': peak, 'psr': psr, 'scale': s_best,
            'patch_px': (sw / n_s, sh / n_s), 'observation': observation,
            'feats_best': feats[si:si + 1], 'grid': (sx, sy, sw, sh, n_s),
            'heatmap': scored[si:si + 1].detach(), 'heatmap_bbox': heat_bbox,
            'debug': debug,
        }

    # ------------------------------------------------------------------
    # Re-detection
    # ------------------------------------------------------------------
    def _redetect(self, frame: np.ndarray, current_peak: float) -> Dict[str, Any]:
        """Search a grid of base-size tiles for the target. Returns a dict with
        'accepted' and the attempt's statistics ('reject' says why it was refused).
        A match is accepted only if it looks like the target did while it was
        tracked well and beats the current local peak.

        Tiles use the base search scale so the template/search pixels-per-patch match,
        and are spaced by the span of centers one tile can reach, so together they cover
        grid x span per axis without gaps. Attempts alternate between the last confident
        position and the current (possibly drifted) estimate.
        """
        cfg = self.cfg
        if len(self.healthy_peak.values) == 0:
            return {'accepted': False, 'reject': 'no_history', 'peak': 0.0, 'peak_thr': 0.0,
                    'id_sim': 0.0, 'id_thr': 0.0, 'tile': -1}
        peak_ref = self.healthy_peak.median(default=0.0)
        id_ref = self.healthy_id.median(default=0.0)
        peak_thr = cfg.redetect_q_min * peak_ref
        id_thr = cfg.redetect_id_min * id_ref

        tw, th = self.target_size
        S = cfg.search_scale
        n_s = self._search_patches(S)
        s_px = n_s * self.patch
        sw, sh = S * tw, S * th
        Ho = n_s - self.kh + 1
        span_x, span_y = (Ho - 1) / n_s * sw, (Ho - 1) / n_s * sh

        base = self.last_good_pos if self.redetect_attempts % 2 == 0 else self.target_pos
        self.redetect_attempts += 1
        g = max(1, int(cfg.redetect_grid))
        centers = [(base[0] + (j - (g - 1) / 2) * span_x, base[1] + (i - (g - 1) / 2) * span_y)
                   for i in range(g) for j in range(g)]
        crops = [_crop_resize(frame, c, (sw, sh), s_px) for c in centers]

        best = None
        bs = max(1, int(cfg.redetect_batch))
        for b0 in range(0, len(crops), bs):
            feats = self._extract(crops[b0:b0 + bs])
            raw = self.memory.correlate(feats)                 # (B, 1, Ho, Wo)
            B, _, Ho_, Wo_ = raw.shape
            flat = int(torch.argmax(raw))
            bi, rem = divmod(flat, Ho_ * Wo_)
            r, c = divmod(rem, Wo_)
            peak = float(raw[bi, 0, r, c])
            if best is None or peak > best['peak']:
                obs = feats[bi:bi + 1, :, r:r + self.kh, c:c + self.kh]
                best = {'peak': peak, 'tile': b0 + bi, 'r': r, 'c': c, 'obs': obs}

        id_sim = self.memory.similarity(best['obs'], self.memory.anchor)
        tc = centers[best['tile']]
        sx, sy = tc[0] - sw / 2, tc[1] - sh / 2
        cx = sx + (best['c'] + self.offset) / n_s * sw
        cy = sy + (best['r'] + self.offset) / n_s * sh
        if best['peak'] < peak_thr:
            reject = 'peak'
        elif id_sim < id_thr:
            reject = 'identity'
        elif best['peak'] <= current_peak:
            reject = 'not_better'
        else:
            reject = ''
        return {'accepted': reject == '', 'reject': reject, 'center': (cx, cy),
                'peak': best['peak'], 'peak_thr': peak_thr, 'id_sim': id_sim, 'id_thr': id_thr,
                'tile': best['tile']}

    # ------------------------------------------------------------------
    # Fine-box reliability ("fine" / "dino" box sources)
    # ------------------------------------------------------------------
    def _fine_reliable(self, fr: Dict[str, Any], ref_center) -> Tuple[bool, str]:
        """Is this LightGlue fit good enough to be the output box? A check on the fit
        itself (support, consistency, plausibility), not an agreement test with DINO."""
        cfg = self.cfg
        if not fr['success']:
            return False, fr.get('fail_reason', '') or 'no_fit'
        tw, th = self.target_size
        if cfg.fine_min_target_px > 0 and np.sqrt(tw * th) < cfg.fine_min_target_px:
            return False, 'small'
        if fr['num_inliers'] < cfg.fine_min_inliers:
            return False, 'inliers'
        if fr['inlier_ratio'] < cfg.fine_min_inlier_ratio:
            return False, 'ratio'
        ratio = np.sqrt(fr['size'][0] * fr['size'][1] / max(tw * th, 1e-6))
        # The allowed step compounds while no fine box was used; after fine_scale_step_frames
        # such frames the previous (DINO) size is no reference any more and the check is off,
        # so a stale DINO size can never lock LightGlue out for good.
        # With fine_scale_lock the check never switches off: the step is capped at
        # fine_max_scale_step ** fine_scale_step_frames (a weak fit after a fallback stretch
        # cannot rescale the box in one go).
        k = max(1, cfg.fine_scale_step_frames)
        if cfg.fine_scale_lock or self.frames_since_fine < k:
            step = cfg.fine_max_scale_step ** min(1 + self.frames_since_fine, k)
            if not (1 / step <= ratio <= step):
                return False, 'scale_jump'
        off = np.hypot((fr['center'][0] - ref_center[0]) / tw, (fr['center'][1] - ref_center[1]) / th)
        if off > cfg.fine_max_offset:
            return False, 'offset'
        return True, ''

    def _fine_observation(self, frame: np.ndarray, co: Dict[str, Any], center, size):
        """DINO memory observation at the fine box (see cfg.fine_memory_obs)."""
        mode = self.cfg.fine_memory_obs
        if mode == 'template':
            ts = self.cfg.template_scale
            crop = _crop_resize(frame, center, (size[0] * ts, size[1] * ts),
                                self.cfg.template_patches * self.patch)
            return self._extract([crop])[:, :, self.k0:self.k1, self.k0:self.k1]
        if mode == 'search':
            sx, sy, sw, sh, n_s = co['grid']
            c = int(round((center[0] - sx) / sw * n_s - self.offset))
            r = int(round((center[1] - sy) / sh * n_s - self.offset))
            hi = n_s - self.kh
            c, r = int(np.clip(c, 0, hi)), int(np.clip(r, 0, hi))
            return co['feats_best'][:, :, r:r + self.kh, c:c + self.kh]
        return co['observation']

    # ------------------------------------------------------------------
    # Track
    # ------------------------------------------------------------------
    def track(self, frame: np.ndarray) -> Dict[str, Any]:
        if not self.initialized:
            raise RuntimeError("Tracker not initialized. Call init() first.")
        cfg = self.cfg
        t0 = time.perf_counter()
        self.frame_idx += 1
        img_h, img_w = frame.shape[:2]

        # 1. Motion prior (only while tracking is healthy)
        pred = self.target_pos
        coasting_ok = self.low_count == 0 or self.low_count <= cfg.motion_coast_frames
        if coasting_ok and len(self.velocity) >= max(1, cfg.motion_min_history):
            v = np.mean(self.velocity, axis=0) * (cfg.velocity_decay ** self.low_count)
            pred = (pred[0] + v[0], pred[1] + v[1])

        # 2-3. Coarse stage + relative quality
        co = self._coarse(frame, pred)
        qa = self.memory.assess(co['peak'], co['observation'])
        q_rel, id_rel = qa['q_rel'], qa['id_rel']
        dino_good = q_rel >= cfg.q_min and id_rel >= cfg.id_min

        tw, th = self.target_size
        scale_ok = dino_good if cfg.scale_q_min < 0 else q_rel >= cfg.scale_q_min
        if scale_ok:
            k = 1 + cfg.scale_lr * (co['scale'] - 1)
            dino_size = (tw * k, th * k)
        else:
            dino_size = (tw, th)
        dino_center = co['center']

        # 4-5. Fine stage + box source
        mode = cfg.box_source
        fr = {'success': False, 'num_matches': 0, 'num_inliers': 0}
        agree = False
        fine_ok, fine_reject = False, 'disabled'
        d_dist = d_ratio = float('nan')
        if self.kp_memory is not None:
            fr = self.kp_memory.match(frame, dino_center, dino_size)
            fine_reject = fr.get('fail_reason', '') or ('' if fr['success'] else 'no_fit')
            if fr['success']:
                d_dist = float(np.hypot((fr['center'][0] - dino_center[0]) / dino_size[0],
                                        (fr['center'][1] - dino_center[1]) / dino_size[1]))
                d_ratio = float(np.sqrt(fr['size'][0] * fr['size'][1] / max(dino_size[0] * dino_size[1], 1e-6)))
            if mode == 'agree':
                if fr['success']:
                    pw, ph = co['patch_px']
                    dist = np.hypot((fr['center'][0] - dino_center[0]) / pw,
                                    (fr['center'][1] - dino_center[1]) / ph)
                    ratio = np.sqrt(fr['size'][0] * fr['size'][1] / max(dino_size[0] * dino_size[1], 1e-6))
                    agree = dist <= cfg.agree_radius and (1 / cfg.agree_scale_tol) <= ratio <= cfg.agree_scale_tol
                    fine_reject = '' if agree else 'disagree'
            else:
                fine_ok, fine_reject = self._fine_reliable(fr, dino_center)
        use_fine = agree if mode == 'agree' else (fine_ok and mode == 'fine')

        if mode == 'agree' and agree:
            center = fr['center']
            a = cfg.fine_size_lr
            size = (tw * (fr['size'][0] / tw) ** a, th * (fr['size'][1] / th) ** a)
            branch = 'fine'
        elif use_fine:
            center = fr['center']
            a = cfg.fine_box_size_lr
            size = (tw * (fr['size'][0] / tw) ** a, th * (fr['size'][1] / th) ** a)
            branch = 'fine'
        else:
            center, size = dino_center, dino_size
            branch = 'coarse'

        # 6. Memory updates
        q_strength = float(np.clip((q_rel - cfg.q_min) / max(1 - cfg.q_min, 1e-6), 0, 1))
        obs = co['observation']
        if mode == 'agree' and fr['success'] and not agree:
            strength, push_short, reason = 0.0, False, 'Fine/DINO disagree: memory frozen'
        elif mode == 'agree' and agree:
            # Geometry confirms the location, so DINO may adapt even when its own
            # score dipped (appearance change) - but not during heavy occlusion.
            ok = id_rel >= 0.9 * cfg.id_min
            strength = float(np.clip((q_rel - 0.5) / 0.5, 0, 1)) if ok else 0.0
            push_short = ok and strength >= 0.8
            reason = 'Fine and DINO agree' + ('' if ok else ' (identity check failed)')
        elif use_fine:
            # The LightGlue box is the best estimate of where the target is: learn the
            # DINO appearance there (same strength rule as "agree"), with the identity
            # check on that observation so occluders are not learnt.
            strength = float(np.clip((q_rel - 0.5) / 0.5, 0, 1))
            ok = True
            if strength > 0:
                obs = self._fine_observation(frame, co, center, size)
                ok = self.memory.assess(co['peak'], obs)['id_rel'] >= 0.9 * cfg.id_min
            strength = strength if ok else 0.0
            push_short = ok and strength >= 0.8
            reason = 'Fine box' + ('' if ok else ' (identity check failed)')
        elif dino_good:
            strength = cfg.dino_only_strength * q_strength
            push_short = q_rel >= 0.9
            reason = 'DINO-only update'
        else:
            strength, push_short, reason = 0.0, False, 'Low quality: memory frozen'
        if mode == 'fine' and not use_fine and self.kp_memory is not None:
            reason += f' (fine fallback: {fine_reject})'
        self.memory.update(obs, strength, push_short)
        if push_short and strength > 0:
            self.short_crops.append(_crop_resize(
                frame, center, (size[0] * cfg.template_scale, size[1] * cfg.template_scale),
                cfg.template_patches * self.patch))

        # Keyframes need a fit (its transform anchors the keyframe to frame 1), so DINO-only
        # frames never add one; DINO must also be healthy, so a bad keyframe needs two failures.
        kf_ok = agree if mode == 'agree' else fine_ok
        kf_added = False
        if (kf_ok and dino_good and self.kp_memory is not None
                and self.frame_idx - self.last_keyframe >= cfg.keyframe_interval):
            kf_added = self.kp_memory.add_keyframe(fr, self.frame_idx)
            if kf_added:
                self.last_keyframe = self.frame_idx

        # 7. Never lost: widen search on sustained low quality, shrink back when healthy
        healthy = dino_good or use_fine
        if healthy:
            self.low_count = 0
            self.search_scale = max(cfg.search_scale, self.search_scale / cfg.widen_factor)
            self.velocity.append((center[0] - self.target_pos[0], center[1] - self.target_pos[1]))
        else:
            self.low_count += 1
            if self.low_count >= cfg.widen_patience:
                self.search_scale = min(cfg.max_search_scale, self.search_scale * cfg.widen_factor)
            if self.low_count > cfg.motion_coast_frames:
                self.velocity.clear()

        # Running statistics for q_rel / id_rel. "all" absorbs every frame (original);
        # "healthy" stops a lost tracker from re-normalising itself onto a distractor.
        if (cfg.stats_mode != 'healthy' or healthy
                or self.low_count % max(1, cfg.stats_leak_every) == 0):
            self.memory.record(co['peak'], qa['id_sim'])
        if healthy:
            self.healthy_peak.push(co['peak'])
            self.healthy_id.push(qa['id_sim'])
            self.last_good_pos = (float(center[0]), float(center[1]))

        box_source = 'fine' if branch == 'fine' else 'dino'
        fine_used = branch == 'fine'

        # 8. Re-detection while lost
        rd = None
        if (cfg.redetect and self.low_count >= cfg.redetect_patience
                and (self.low_count - cfg.redetect_patience) % max(1, cfg.redetect_interval) == 0):
            rd = self._redetect(frame, co['peak'])
            rd_box = None
            if (rd['accepted'] and self.kp_memory is not None and cfg.redetect_fine != 'off'
                    and mode != 'dino'):   # the dino probe must track exactly like coarse-only
                fr2 = self.kp_memory.match(frame, rd['center'], (tw, th))
                ok2, _ = self._fine_reliable(fr2, rd['center'])
                if ok2:
                    rd_box = (fr2['center'], fr2['size'])
                elif cfg.redetect_fine == 'require':
                    rd['accepted'], rd['reject'] = False, 'fine'
            if rd['accepted']:
                center, size = rd_box if rd_box is not None else (rd['center'], (tw, th))
                branch, strength = 'redetect', 0.0
                box_source = 'redetect_fine' if rd_box is not None else 'redetect'
                fine_used = rd_box is not None
                reason = (f"Re-detected (peak {rd['peak']:.3f} >= {rd['peak_thr']:.3f}, "
                          f"id {rd['id_sim']:.3f}, tile {rd['tile']})")
                self.low_count = 0
                self.search_scale = cfg.search_scale
                self.velocity.clear()
                self.last_good_pos = (float(center[0]), float(center[1]))

        self.frames_since_fine = 0 if (fine_ok or fine_used) else self.frames_since_fine + 1

        cx = float(np.clip(center[0], 0, img_w - 1))
        cy = float(np.clip(center[1], 0, img_h - 1))
        w = float(np.clip(size[0], 4.0, img_w))
        h = float(np.clip(size[1], 4.0, img_h))
        self.target_pos, self.target_size = (cx, cy), (w, h)

        out = self._clip_bbox((cx - w / 2, cy - h / 2, w, h), img_w, img_h)
        conf = float(np.clip(q_rel, 0, 1))
        if agree or (use_fine and branch == 'fine'):
            conf = max(conf, float(fr.get('inlier_ratio', 0.0)))

        dino_box = (dino_center[0] - dino_size[0] / 2, dino_center[1] - dino_size[1] / 2,
                    dino_size[0], dino_size[1])
        fine_box = ((fr['center'][0] - fr['size'][0] / 2, fr['center'][1] - fr['size'][1] / 2,
                     fr['size'][0], fr['size'][1]) if fr['success'] else (0.0, 0.0, 0.0, 0.0))
        diag = AdaptiveDiagnostics(
            frame_idx=self.frame_idx, bbox=out, branch=branch, peak=co['peak'], q_rel=q_rel,
            id_rel=id_rel, psr=co['psr'], scale=co['scale'], search_scale=self.search_scale,
            fine_success=bool(fr['success']), fine_agree=bool(agree),
            fine_num_matches=int(fr.get('num_matches', 0)), fine_num_inliers=int(fr.get('num_inliers', 0)),
            memory_strength=strength, keyframe_added=kf_added, overall_confidence=conf,
            reason=reason, elapsed_ms=(time.perf_counter() - t0) * 1000,
            box_source=box_source, fine_inlier_ratio=float(fr.get('inlier_ratio', 0.0)),
            fine_num_keypoints=int(fr.get('num_keypoints', 0)), fine_reject=fine_reject,
            dino_fine_dist=d_dist, dino_fine_size_ratio=d_ratio,
            dino_box=tuple(float(v) for v in dino_box), fine_box=tuple(float(v) for v in fine_box))
        if rd is not None:
            diag.redetect_tried, diag.redetect_accepted = True, bool(rd['accepted'])
            diag.redetect_peak, diag.redetect_peak_thr = float(rd['peak']), float(rd['peak_thr'])
            diag.redetect_id, diag.redetect_id_thr = float(rd['id_sim']), float(rd['id_thr'])
            diag.redetect_reject = rd['reject']
        self.diagnostics.append(diag)

        dbg = co['debug']
        if dbg is not None:
            dbg['frame_idx'] = self.frame_idx
            dbg['info'] = {'branch': branch, 'peak': co['peak'], 'q_rel': q_rel, 'id_rel': id_rel,
                           'psr': co['psr'], 'scale': co['scale'], 'search_scale': self.search_scale,
                           'fine inliers': int(fr.get('num_inliers', 0)), 'fine agree': bool(agree),
                           'box source': box_source, 'fine reject': fine_reject,
                           'mem strength': strength, 'short push': bool(push_short),
                           'keyframe added': kf_added, 'reason': reason[:40]}

        return {'bbox': out, 'confidence': conf, 'branch': branch, 'diagnostics': diag,
                'heatmap': co['heatmap'], 'search_bbox': co['heatmap_bbox'], 'debug': dbg}

    # ------------------------------------------------------------------
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
        counts: Dict[str, int] = {}
        for x in self.diagnostics:
            counts[x.branch] = counts.get(x.branch, 0) + 1
        n = max(1, len(d))
        return {
            'total_frames': len(self.diagnostics),
            'branch_counts': counts,
            'avg_fps': 1000.0 / np.mean(times) if times else 0,
            'avg_ms': float(np.mean(times)) if times else 0,
            'median_ms': float(np.median(times)) if times else 0,
            'p95_ms': float(np.percentile(times, 95)) if times else 0,
            'lost_frames': 0,
            'fine_ratio': counts.get('fine', 0) / n,
            'fine_success_ratio': sum(x.fine_success for x in d) / n,
            'memory_update_ratio': sum(x.memory_strength > 0 for x in d) / n,
            'keyframes_added': sum(x.keyframe_added for x in d),
            'avg_confidence': float(np.mean([x.overall_confidence for x in self.diagnostics])),
            'box_source': self.cfg.box_source,
            'fine_box_ratio': sum(x.box_source in ('fine', 'redetect_fine') for x in d) / n,
            'fine_reject_counts': dict(Counter(x.fine_reject for x in d if x.box_source == 'dino')),
            'fine_anchor_keypoints': int(getattr(self.kp_memory, 'anchor_keypoints', 0) or 0),
            'fine_anchor_norm_px': float(getattr(self.kp_memory, 'anchor_norm_px', 0.0) or 0.0),
        }
