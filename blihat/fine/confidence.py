# Copyright 2026 Dagimawi Moges Wakumo
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ConfidenceGate:
    """Decides whether to use the fine-stage bbox or fall back to coarse.
    
    The core decision logic of the coarse-to-fine cascade. Uses multiple
    signals (PSR, peak sharpness, inlier ratio, geometric consistency, area ratio)
    to determine whether the geometric matcher produced a reliable result, or
    if tracking should smoothly rely on DINO coarse features or mark lost.
    """
    
    def __init__(self, inlier_threshold_high: float = 0.5,
                 inlier_threshold_low: float = 0.15,
                 min_matches: int = 4,
                 area_ratio_bounds: tuple = (0.5, 2.0),
                 max_reprojection_error: float = 5.0,
                 lost_threshold: float = 0.1,
                 psr_threshold: float = 2.0,
                 sharpness_threshold: float = 1.15):
        self.inlier_threshold_high = inlier_threshold_high
        self.inlier_threshold_low = inlier_threshold_low
        self.min_matches = min_matches
        self.area_ratio_bounds = area_ratio_bounds
        self.max_reprojection_error = max_reprojection_error
        self.lost_threshold = lost_threshold
        self.psr_threshold = psr_threshold
        self.sharpness_threshold = sharpness_threshold

    def _is_coarse_valid(self, coarse_result: Dict[str, Any]) -> bool:
        """Check if coarse peak is valid based on relative prominence (PSR/sharpness)."""
        psr = coarse_result.get('psr', 0.0)
        sharpness = coarse_result.get('sharpness', 1.0)
        coarse_conf = coarse_result.get('confidence', 0.0)
        
        # Primary check: Relative peak prominence
        if psr >= self.psr_threshold or sharpness >= self.sharpness_threshold:
            return True
            
        # Fallback check: Absolute confidence if PSR/sharpness are uncomputed
        if psr == 0.0 and sharpness == 1.0 and coarse_conf >= self.lost_threshold:
            return True
            
        return False

    def decide(self, coarse_result: Dict[str, Any], fine_result: Dict[str, Any], 
               expected_size: Tuple[float, float]) -> Dict[str, Any]:
        """
        Decide which bbox to use: fine, coarse, or blend.
        
        Args:
            coarse_result: dict from coarse stage with 'peak_pos', 'confidence', 'psr', 'sharpness', 'bbox'
            fine_result: dict from GeometricEstimator.estimate_bbox()
            expected_size: (w, h) expected target size from previous frame
        
        Returns:
            dict with:
                'bbox': (x, y, w, h) final bounding box
                'branch': str, one of 'fine', 'coarse', 'blend', 'lost'
                'confidence': float, overall tracking confidence
                'fine_weight': float, weight given to fine result (0-1)
                'reason': str, human-readable explanation of the decision
        """
        coarse_bbox = coarse_result.get('bbox')
        coarse_conf = coarse_result.get('confidence', 0.0)
        
        # 1. If fine_result['success'] is False → always use coarse prediction (never lost)
        if not fine_result.get('success', False):
            return {
                'bbox': coarse_bbox,
                'branch': 'coarse',
                'confidence': coarse_conf,
                'fine_weight': 0.0,
                'reason': 'Fine matching failed; using DINO coarse feature peak (never-lost policy)'
            }
            
        fine_bbox = fine_result['bbox']
        inlier_ratio = fine_result['inlier_ratio']
        num_inliers = fine_result['num_inliers']
        repr_error = fine_result['reprojection_error']
        
        area_ratio = self._compute_area_ratio(fine_bbox, expected_size)
        area_valid = self.area_ratio_bounds[0] <= area_ratio <= self.area_ratio_bounds[1]
        
        # 2. If inlier_ratio > high_threshold AND area_ratio in bounds → use fine
        if inlier_ratio >= self.inlier_threshold_high and area_valid and repr_error <= self.max_reprojection_error:
            return {
                'bbox': fine_bbox,
                'branch': 'fine',
                'confidence': float(inlier_ratio),
                'fine_weight': 1.0,
                'reason': 'High inlier ratio and valid geometry'
            }
            
        # 3. If inlier_ratio < low_threshold OR num_matches < min → use coarse (never lost)
        if inlier_ratio < self.inlier_threshold_low or num_inliers < self.min_matches:
            return {
                'bbox': coarse_bbox,
                'branch': 'coarse',
                'confidence': coarse_conf,
                'fine_weight': 0.0,
                'reason': 'Low keypoint inliers; using DINO coarse feature peak (never-lost policy)'
            }
            
        # 4. Otherwise → weighted blend based on inlier_ratio
        weight = (inlier_ratio - self.inlier_threshold_low) / (self.inlier_threshold_high - self.inlier_threshold_low)
        weight = float(np.clip(weight, 0.0, 1.0))
        
        blended_bbox = self._blend_bboxes(fine_bbox, coarse_bbox, weight)
        
        return {
            'bbox': blended_bbox,
            'branch': 'blend',
            'confidence': coarse_conf * (1 - weight) + inlier_ratio * weight,
            'fine_weight': weight,
            'reason': 'Intermediate confidence, blending coarse and fine'
        }
    
    def _blend_bboxes(self, bbox1: Tuple[float, float, float, float], 
                      bbox2: Tuple[float, float, float, float], 
                      weight: float) -> Tuple[float, float, float, float]:
        """Weighted average of two bboxes (in xywh format). weight applies to bbox1."""
        # Convert both to center format
        x1, y1, w1, h1 = bbox1
        cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
        
        x2, y2, w2, h2 = bbox2
        cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2
        
        # Weighted average
        cx = cx1 * weight + cx2 * (1 - weight)
        cy = cy1 * weight + cy2 * (1 - weight)
        w = w1 * weight + w2 * (1 - weight)
        h = h1 * weight + h2 * (1 - weight)
        
        # Convert back
        return (cx - w / 2, cy - h / 2, w, h)
    
    def _compute_area_ratio(self, bbox: Tuple[float, float, float, float], 
                            expected_size: Tuple[float, float]) -> float:
        """Ratio of bbox area to expected area."""
        w, h = bbox[2], bbox[3]
        ew, eh = expected_size
        
        expected_area = ew * eh
        if expected_area <= 0:
            return 1.0
            
        return (w * h) / expected_area
