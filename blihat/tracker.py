# Copyright 2026 Dagimawi Moges Wakumo
# SPDX-License-Identifier: Apache-2.0
"""
Blihat Tracker: Training-free coarse-to-fine UAV object tracker.

Cascades DINOv2/v3 semantic features (coarse stage) with LightGlue/SuperPoint
geometric matching (fine stage) for zero-shot tracking without any training.
"""

import numpy as np
import cv2
import time
import logging
from typing import Optional, Tuple, Dict, Any, List
from collections import deque
from dataclasses import dataclass

from blihat.config import Config
from blihat.backbone.dinov2 import DINOv2Backbone
from blihat.coarse.correlation import DepthwiseCorrelation
from blihat.coarse.heatmap import HeatmapProcessor
from blihat.coarse.template import TemplateManager
from blihat.fine.lightglue_matcher import LightGlueMatcher
from blihat.fine.geometry import GeometricEstimator
from blihat.fine.confidence import ConfidenceGate

logger = logging.getLogger(__name__)


@dataclass
class FrameDiagnostics:
    """Per-frame diagnostic information for debugging and analysis."""
    frame_idx: int = 0
    bbox: tuple = (0, 0, 0, 0)
    branch: str = "init"            # 'fine', 'coarse', 'blend', 'lost', 'redetect'
    coarse_confidence: float = 0.0
    coarse_peak_value: float = 0.0
    coarse_sharpness: float = 0.0
    fine_num_matches: int = 0
    fine_num_inliers: int = 0
    fine_inlier_ratio: float = 0.0
    fine_reproj_error: float = 0.0
    fine_weight: float = 0.0
    overall_confidence: float = 0.0
    reason: str = ""
    elapsed_ms: float = 0.0


class CascadeTracker:
    """Training-free coarse-to-fine tracker for UAV video.
    
    Pipeline per frame:
    1. Crop search region (4× target size, centered on last position)
    2. Extract DINOv2 features (search region)
    3. Depthwise correlation with template → score map
    4. Peak detection → coarse position + confidence
    5. Crop fine region (2× target size, centered on coarse peak)
    6. Run LightGlue matcher on (template_crop, fine_crop)
    7. Confidence gate decides: use fine bbox, coarse bbox, or blend
    8. Smooth update (EMA on position and size)
    9. Template update (if confidence high, every N frames)
    10. Log diagnostics
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize tracker with configuration.
        
        Args:
            config: Config object. If None, uses default config.
        """
        if config is None:
            config = Config.default()
        self.config = config
        
        # Initialize backbone
        logger.info(f"Initializing backbone: {config.backbone.name}")
        self.backbone = DINOv2Backbone(
            model_name=config.backbone.name,
            device=config.device,
            final_norm=config.backbone.final_norm,
            dtype=config.backbone.dtype,
        )
        
        # Initialize coarse stage components
        self.correlator = DepthwiseCorrelation()
        self.heatmap_processor = HeatmapProcessor(
            cosine_window_weight=config.coarse.cosine_window_weight,
            patch_size=config.backbone.patch_size
        )
        self.template_manager = TemplateManager(self.backbone, config)
        
        # Initialize fine stage components
        self.matcher = LightGlueMatcher(
            max_keypoints=config.fine.max_keypoints,
            device=config.device
        )
        self.geometric_estimator = GeometricEstimator(
            ransac_threshold=config.fine.ransac_threshold,
            ransac_method=config.fine.ransac_method,
            area_ratio_bounds=config.fine.area_ratio_bounds
        )
        self.confidence_gate = ConfidenceGate(
            inlier_threshold_high=config.fine.inlier_threshold_high,
            inlier_threshold_low=config.fine.inlier_threshold_low,
            min_matches=config.fine.min_matches,
            lost_threshold=config.tracking.lost_threshold,
            psr_threshold=config.coarse.psr_threshold,
            sharpness_threshold=config.coarse.sharpness_threshold
        )
        
        # Tracking state
        self.target_pos = None      # (cx, cy) center of target
        self.target_size = None     # (w, h) target size
        self.velocity = deque(maxlen=config.tracking.velocity_smoothing)
        self.lost_count = 0
        self.frame_idx = 0
        self.initialized = False
        
        # Diagnostics log
        self.diagnostics: List[FrameDiagnostics] = []
        # Set True to get result['debug'] (template + heatmap at query time), see utils/debug_viz.py
        self.debug = False
        
        # Template bbox in template crop coordinates
        # When we crop with template_scale, the target sits in the center
        self._template_bbox_in_crop = None
    
    def init(self, frame: np.ndarray, bbox: tuple):
        """Initialize tracker on first frame.
        
        Args:
            frame: first frame (H, W, 3) BGR
            bbox: (x, y, w, h) initial bounding box
        """
        x, y, w, h = bbox
        self.target_pos = (x + w / 2, y + h / 2)
        self.target_size = (w, h)
        self.velocity.clear()
        self.lost_count = 0
        self.frame_idx = 0
        self.diagnostics = []
        
        # Initialize template
        self.template_manager.initialize(frame, bbox)
        
        # The target's box inside the template crop is computed per frame in the fine stage.

        self.initialized = True
        
        diag = FrameDiagnostics(
            frame_idx=0, bbox=bbox, branch='init',
            overall_confidence=1.0, reason='Initialization'
        )
        self.diagnostics.append(diag)
        
        logger.info(f"Tracker initialized: pos={self.target_pos}, size={self.target_size}")
    
    def track(self, frame: np.ndarray) -> Dict[str, Any]:
        """Track target in a new frame.
        
        Args:
            frame: current frame (H, W, 3) BGR
            
        Returns:
            dict with:
                'bbox': (x, y, w, h) predicted bounding box
                'confidence': float, tracking confidence
                'branch': str, which stage produced the result
                'diagnostics': FrameDiagnostics for this frame
        """
        if not self.initialized:
            raise RuntimeError("Tracker not initialized. Call init() first.")
        
        t_start = time.perf_counter()
        self.frame_idx += 1
        
        img_h, img_w = frame.shape[:2]
        
        # ── Apply motion prior ──
        pred_center = self._apply_motion_prior()
        
        # ── Re-detection mode if lost ──
        if self.lost_count >= self.config.tracking.lost_patience:
            return self._redetect(frame, t_start)
        
        # ══════════════════════════════════════
        # STAGE 1: COARSE — Semantic Correlation
        # ══════════════════════════════════════
        
        # Extract search region features
        search_feats, search_crop, search_bbox = self.template_manager.get_search_features(
            frame, pred_center, self.target_size
        )
        
        # Get template features
        template_feats_list = self.template_manager.get_template_features(multi_scale=True)
        # References to the crops that produced these features (template update replaces them later)
        _dbg_tpl_crop = self.template_manager.template_crop
        _dbg_coarse_crop = getattr(self.template_manager, 'coarse_crop', _dbg_tpl_crop)
        
        # Multi-scale correlation
        if len(template_feats_list) > 1:
            response_map, best_scale_idx = self.correlator.multi_scale_correlate(
                search_feats, template_feats_list
            )
        else:
            response_map = self.correlator(search_feats, template_feats_list[0])
            best_scale_idx = None
        
        # Find peak
        coarse_result = self.heatmap_processor.process(response_map, search_bbox)
        coarse_peak_pos = coarse_result['peak_pos']
        coarse_confidence = coarse_result['confidence']
        
        # ── Apply scale update ──
        tw, th = self.target_size
        if best_scale_idx is not None:
            # Get the scale index at the peak position
            peak_scale_idx = best_scale_idx[0, 0, coarse_result['peak_row'], coarse_result['peak_col']].item()
            scale_factor = self.config.coarse.multi_scale_factors[int(peak_scale_idx)]
            
            # scale_factor < 1.0 means target got bigger relative to the template
            target_scale_change = 1.0 / scale_factor
            
            # EMA smooth the size update
            ema_sz = self.config.tracking.ema_size
            tw = tw * (ema_sz * 1.0 + (1 - ema_sz) * target_scale_change)
            th = th * (ema_sz * 1.0 + (1 - ema_sz) * target_scale_change)
            
            # Clip to image size to prevent exponential runaway memory explosion
            tw = min(max(10.0, tw), float(img_w))
            th = min(max(10.0, th), float(img_h))
            self.target_size = (tw, th)
        
        # Build coarse bounding box (centered on peak)
        coarse_bbox = (
            coarse_peak_pos[0] - tw / 2,
            coarse_peak_pos[1] - th / 2,
            tw, th
        )
        coarse_result['bbox'] = coarse_bbox
        
        # ══════════════════════════════════════
        # STAGE 2: FINE — Geometric Matching
        # ══════════════════════════════════════
        
        fine_result = {'success': False, 'num_matches': 0, 'num_inliers': 0,
                       'inlier_ratio': 0.0, 'reprojection_error': float('inf')}
        
        run_fine = (
            self.frame_idx % self.config.tracking.matcher_every_n == 0
            and (
                coarse_confidence > self.config.coarse.peak_threshold
                or coarse_result.get('psr', 0.0) >= self.config.coarse.psr_threshold
                or coarse_result.get('sharpness', 1.0) >= self.config.coarse.sharpness_threshold
            )
        )
        
        if run_fine:
            # Crop fine region centered on coarse peak
            fine_crop_bbox = self.heatmap_processor.get_crop_region(
                coarse_peak_pos, self.target_size,
                self.config.fine.crop_scale, (img_h, img_w)
            )
            fine_crop = self._crop_region(frame, fine_crop_bbox)
            
            # Get UNDISTORTED template crop for matching
            template_crop, initial_bbox = self.template_manager.get_fine_template()
            
            # Run LightGlue matcher
            match_result = self.matcher.match(template_crop, fine_crop)
            
            if match_result['num_matches'] >= self.config.fine.min_matches:
                # Compute template bbox dynamically for undistorted crop
                ts = self.config.coarse.template_scale
                tw_i, th_i = initial_bbox[2], initial_bbox[3]
                template_bbox_in_crop = (
                    tw_i * (ts - 1) / 2,
                    th_i * (ts - 1) / 2,
                    tw_i,
                    th_i
                )
                
                # Estimate bbox from matches
                fine_result = self.geometric_estimator.estimate_bbox(
                    match_result['kpts_template'],
                    match_result['kpts_search'],
                    template_bbox_in_crop
                )
                fine_result['num_matches'] = match_result['num_matches']
                
                # If fine stage succeeded, convert bbox from fine crop coords to image coords
                if fine_result['success'] and fine_result['bbox'] is not None:
                    fine_result['bbox'] = self._crop_bbox_to_image(
                        fine_result['bbox'], fine_crop_bbox, fine_crop.shape
                    )
            else:
                fine_result['num_matches'] = match_result['num_matches']
        
        # ══════════════════════════════════════
        # DECISION: Confidence Gate
        # ══════════════════════════════════════
        
        decision = self.confidence_gate.decide(
            coarse_result, fine_result, self.target_size
        )
        
        final_bbox = decision['bbox']
        branch = decision['branch']
        confidence = decision['confidence']
        
        # ── Handle lost state ──
        if branch == 'lost':
            self.lost_count += 1
        else:
            self.lost_count = 0
        
        # ── Smooth update ──
        if final_bbox is not None and branch != 'lost':
            self._smooth_update(final_bbox, img_w, img_h)
            
            # Clip internal position to image bounds to prevent infinite drift
            cx, cy = self.target_pos
            cx = max(0.0, min(cx, float(img_w - 1)))
            cy = max(0.0, min(cy, float(img_h - 1)))
            self.target_pos = (cx, cy)
        
        # ── Template update ──
        if final_bbox is not None:
            self.template_manager.update(frame, self._get_bbox(), confidence)
        
        # ── Build output bbox ──
        output_bbox = self._get_bbox()
        # Clip to image bounds
        output_bbox = self._clip_bbox(output_bbox, img_w, img_h)
        
        # ── Debug dump: template + heatmap exactly as queried this frame ──
        debug = None
        if self.debug:
            tm = self.template_manager
            used = template_feats_list[0] if len(template_feats_list) == 1 else template_feats_list[len(template_feats_list) // 2]
            ps = self.config.backbone.patch_size
            s_px = search_crop.shape[1]
            debug = {
                'frame_idx': self.frame_idx,
                'template_images': [('template crop (queried)', _dbg_tpl_crop),
                                    ('DINO input (bg removed)', _dbg_coarse_crop),
                                    ('fine template (frame 1)', tm.initial_undistorted_crop)],
                'kernels': [('initial', tm.initial_template_feats[0].float().cpu().numpy()),
                            ('queried', used[0].float().cpu().numpy())],
                'patch_weights': None,
                'search_crops': [search_crop],
                'scale_labels': ['1.000'],
                'responses': response_map[:, 0].float().cpu().numpy(),
                'scored': coarse_result['response_map'][:, 0].float().cpu().numpy(),
                'best': (0, coarse_result['peak_row'], coarse_result['peak_col']),
                'sub': (coarse_result['sub_row'], coarse_result['sub_col']),
                'offset_patches': self.heatmap_processor.template_patches / 2,
                'patch_px': ps,
                'target_wh_crop': (s_px / self.config.coarse.search_scale,) * 2,
                'info': {'branch': branch, 'peak': coarse_result['peak_value'],
                         'psr': coarse_result['psr'], 'sharpness': coarse_result['sharpness'],
                         'confidence': float(confidence),
                         'fine inliers': int(fine_result.get('num_inliers', 0)),
                         'inlier ratio': float(fine_result.get('inlier_ratio', 0.0)),
                         'reason': decision.get('reason', '')[:40]},
            }

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        
        # ── Log diagnostics ──
        diag = FrameDiagnostics(
            frame_idx=self.frame_idx,
            bbox=output_bbox,
            branch=branch,
            coarse_confidence=coarse_confidence,
            coarse_peak_value=coarse_result['peak_value'],
            coarse_sharpness=coarse_result['sharpness'],
            fine_num_matches=fine_result.get('num_matches', 0),
            fine_num_inliers=fine_result.get('num_inliers', 0),
            fine_inlier_ratio=fine_result.get('inlier_ratio', 0.0),
            fine_reproj_error=fine_result.get('reprojection_error', 0.0),
            fine_weight=decision.get('fine_weight', 0.0),
            overall_confidence=confidence,
            reason=decision.get('reason', ''),
            elapsed_ms=elapsed_ms
        )
        self.diagnostics.append(diag)
        
        if self.frame_idx % 50 == 0:
            logger.info(
                f"Frame {self.frame_idx}: branch={branch}, conf={confidence:.3f}, "
                f"inliers={fine_result.get('num_inliers', 0)}, "
                f"elapsed={elapsed_ms:.1f}ms"
            )
        
        return {
            'bbox': output_bbox,
            'confidence': confidence,
            'branch': branch,
            'diagnostics': diag,
            'heatmap': coarse_result.get('response_map', None) if 'coarse_result' in locals() else None,
            'search_bbox': search_bbox if 'search_bbox' in locals() else None,
            'debug': debug,
        }
    
    def _apply_motion_prior(self) -> Tuple[float, float]:
        """Apply simple velocity-based motion prior to predict next center.
        
        Returns:
            Predicted center (cx, cy)
        """
        cx, cy = self.target_pos
        
        if len(self.velocity) >= 2:
            # Average velocity over recent frames
            vx = np.mean([v[0] for v in self.velocity])
            vy = np.mean([v[1] for v in self.velocity])
            cx += vx
            cy += vy
        
        return (cx, cy)
    
    def _smooth_update(self, new_bbox: tuple, img_w: int, img_h: int):
        """Smoothly update target position and size using EMA.
        
        Args:
            new_bbox: (x, y, w, h) new bounding box
            img_w: integer image width for clipping
            img_h: integer image height for clipping
        """
        x, y, w, h = new_bbox
        new_cx, new_cy = x + w / 2, y + h / 2
        
        ema_pos = self.config.tracking.ema_position
        ema_sz = self.config.tracking.ema_size
        
        old_cx, old_cy = self.target_pos
        old_w, old_h = self.target_size
        
        # Update velocity before smoothing
        self.velocity.append((new_cx - old_cx, new_cy - old_cy))
        
        # EMA smooth
        smooth_cx = ema_pos * old_cx + (1 - ema_pos) * new_cx
        smooth_cy = ema_pos * old_cy + (1 - ema_pos) * new_cy
        smooth_w = ema_sz * old_w + (1 - ema_sz) * w
        smooth_h = ema_sz * old_h + (1 - ema_sz) * h
        
        # Clip size to image bounds
        smooth_w = min(max(10.0, smooth_w), float(img_w))
        smooth_h = min(max(10.0, smooth_h), float(img_h))
        
        self.target_pos = (smooth_cx, smooth_cy)
        self.target_size = (smooth_w, smooth_h)
    
    def _get_bbox(self) -> tuple:
        """Get current bounding box from target_pos and target_size."""
        cx, cy = self.target_pos
        w, h = self.target_size
        return (cx - w / 2, cy - h / 2, w, h)
    
    def _clip_bbox(self, bbox: tuple, img_w: int, img_h: int) -> tuple:
        """Clip bounding box to image bounds."""
        x, y, w, h = bbox
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = max(1, min(w, img_w - x))
        h = max(1, min(h, img_h - y))
        return (x, y, w, h)
    
    def _crop_region(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """Crop a region from the frame, handling boundary padding.
        
        Args:
            frame: (H, W, 3) image
            bbox: (x, y, w, h) region to crop
            
        Returns:
            Cropped image
        """
        x, y, w, h = [int(v) for v in bbox]
        img_h, img_w = frame.shape[:2]
        
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)
        
        crop = frame[y1:y2, x1:x2]
        
        # Pad if needed
        pad_left = max(0, -x)
        pad_top = max(0, -y)
        pad_right = max(0, (x + w) - img_w)
        pad_bottom = max(0, (y + h) - img_h)
        
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            crop = cv2.copyMakeBorder(
                crop, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_REPLICATE
            )
        
        return crop
    
    def _crop_bbox_to_image(self, bbox_in_crop: tuple, crop_bbox: tuple,
                             crop_shape: tuple) -> tuple:
        """Convert a bounding box from crop coordinates to image coordinates.
        
        Args:
            bbox_in_crop: (x, y, w, h) in crop pixel coordinates
            crop_bbox: (x, y, w, h) of the crop region in image coordinates
            crop_shape: (H, W, ...) shape of the crop image
            
        Returns:
            (x, y, w, h) in image coordinates
        """
        bx, by, bw, bh = bbox_in_crop
        cx, cy, cw, ch = crop_bbox
        crop_h, crop_w = crop_shape[:2]
        
        # Scale from crop pixel coords to image coords
        scale_x = cw / crop_w
        scale_y = ch / crop_h
        
        img_x = cx + bx * scale_x
        img_y = cy + by * scale_y
        img_w = bw * scale_x
        img_h = bh * scale_y
        
        return (img_x, img_y, img_w, img_h)
    
    def _redetect(self, frame: np.ndarray, t_start: float) -> Dict[str, Any]:
        """Attempt re-detection using initial template on full frame.
        
        Args:
            frame: current frame (H, W, 3) BGR
            t_start: time when track() started
            
        Returns:
            Standard track result dict
        """
        logger.info(f"Frame {self.frame_idx}: Re-detection triggered (lost for {self.lost_count} frames)")
        
        img_h, img_w = frame.shape[:2]
        
        # Use initial template features for re-detection
        initial_feats = self.template_manager.get_initial_template_features()
        
        # Extract features from full frame (resized to manageable size)
        full_frame_tensor = self.backbone.preprocess(frame, is_bgr=True)
        full_frame_feats = self.backbone.extract_features(full_frame_tensor)
        
        # Correlate
        response_map = self.correlator(full_frame_feats['deep'], initial_feats)
        
        # Find peak  
        full_frame_bbox = (0, 0, img_w, img_h)
        coarse_result = self.heatmap_processor.process(
            response_map, full_frame_bbox, apply_window=False
        )
        
        peak_pos = coarse_result['peak_pos']
        confidence = coarse_result['confidence']
        
        if confidence > self.config.coarse.peak_threshold:
            # Re-detected — reset tracker state
            tw, th = self.target_size
            new_bbox = (peak_pos[0] - tw / 2, peak_pos[1] - th / 2, tw, th)
            self.target_pos = peak_pos
            self.lost_count = 0
            self.velocity.clear()
            
            # Rebuild template
            self.template_manager.initialize(frame, new_bbox)
            
            branch = 'redetect'
            logger.info(f"Re-detection successful at {peak_pos}")
        else:
            new_bbox = self._get_bbox()
            branch = 'lost'
            self.lost_count += 1
        
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        
        output_bbox = self._clip_bbox(new_bbox, img_w, img_h)
        
        diag = FrameDiagnostics(
            frame_idx=self.frame_idx,
            bbox=output_bbox,
            branch=branch,
            coarse_confidence=confidence,
            overall_confidence=confidence,
            reason=f"Re-detection {'success' if branch == 'redetect' else 'failed'}",
            elapsed_ms=elapsed_ms
        )
        self.diagnostics.append(diag)
        
        return {
            'bbox': output_bbox,
            'confidence': confidence,
            'branch': branch,
            'diagnostics': diag
        }
    
    def get_diagnostics_summary(self) -> Dict[str, Any]:
        """Get summary statistics of tracking diagnostics.
        
        Returns:
            Dict with summary statistics.
        """
        if not self.diagnostics:
            return {}
        
        branches = [d.branch for d in self.diagnostics]
        times = [d.elapsed_ms for d in self.diagnostics[1:]]  # Skip init frame
        
        branch_counts = {}
        for b in branches:
            branch_counts[b] = branch_counts.get(b, 0) + 1
        
        return {
            'total_frames': len(self.diagnostics),
            'branch_counts': branch_counts,
            'avg_fps': 1000.0 / np.mean(times) if times else 0,
            'avg_ms': np.mean(times) if times else 0,
            'median_ms': np.median(times) if times else 0,
            'p95_ms': np.percentile(times, 95) if times else 0,
            'lost_frames': branch_counts.get('lost', 0),
            'redetect_frames': branch_counts.get('redetect', 0),
            'fine_ratio': branch_counts.get('fine', 0) / max(1, len(self.diagnostics)),
            'avg_confidence': np.mean([d.overall_confidence for d in self.diagnostics]),
        }
