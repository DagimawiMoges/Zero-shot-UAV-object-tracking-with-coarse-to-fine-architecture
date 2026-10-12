# Copyright 2026 Dagimawi Moges Wakumo
# SPDX-License-Identifier: Apache-2.0
import torch
import numpy as np
import cv2
import logging
from typing import Optional, Tuple

from blihat.utils.background import BackgroundRemover

logger = logging.getLogger(__name__)


class TemplateManager:
    """Manages template features for the coarse tracking stage.
    
    Handles:
    - Initial template extraction with context padding
    - Multi-scale template storage (features at multiple scales)
    - EMA template update with confidence gating
    - Re-detection support (full-frame search when target is lost)
    """
    
    def __init__(self, backbone, config):
        """
        Args:
            backbone: DINOv2Backbone instance
            config: Config object with .coarse and .tracking sub-configs
        """
        self.backbone = backbone
        self.config = config
        
        self.template_feats = None          # Current template features (deep layer)
        self.template_feats_multiscale = []  # Multi-scale versions
        self.initial_template_feats = None   # Never updated (for re-detection)
        self.template_bbox = None            # (x, y, w, h) in original template crop coordinates
        self.template_crop = None            # Raw template crop (for fine stage)
        self.frame_count = 0
        
        # Extract config values
        self.template_scale = config.coarse.template_scale
        self.search_scale = config.coarse.search_scale
        self.multi_scale_factors = config.coarse.multi_scale_factors
        self.update_interval = config.tracking.template_update_interval
        self.update_threshold = config.tracking.template_update_threshold
        self.template_update_alpha = getattr(config.tracking, 'template_update_alpha', 0.90)
        self.patch_size = config.backbone.patch_size

        # Background removal (coarse stage only, initialization only)
        self.bg_removal_enabled = config.coarse.bg_removal
        self._bg_remover = None  # lazy
        if self.bg_removal_enabled:
            self._bg_remover = BackgroundRemover(fill=config.coarse.bg_removal_fill)
        
    def _crop_and_pad(self, frame: np.ndarray, bbox: tuple, scale: float, 
                       output_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Crop region from frame with context padding for out-of-bounds.
        
        Args:
            frame: (H, W, 3) BGR image
            bbox: (x, y, w, h) region to crop around
            scale: context scale factor (crop_size = bbox_size * scale)
            output_size: (W, H) resize target, or None to keep original size
            
        Returns:
            Cropped image, padded if necessary
        """
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        
        crop_w, crop_h = w * scale, h * scale
        
        x1 = int(cx - crop_w / 2)
        y1 = int(cy - crop_h / 2)
        x2 = int(cx + crop_w / 2)
        y2 = int(cy + crop_h / 2)
        
        img_h, img_w = frame.shape[:2]
        
        # Valid intersection with image bounds
        valid_x1 = max(0, x1)
        valid_y1 = max(0, y1)
        valid_x2 = min(img_w, x2)
        valid_y2 = min(img_h, y2)
        
        # If the crop is completely outside the image, just replicate the nearest edge pixel
        if valid_x1 >= valid_x2 or valid_y1 >= valid_y2:
            closest_x = max(0, min(x1, img_w - 1))
            closest_y = max(0, min(y1, img_h - 1))
            pixel = frame[closest_y:closest_y+1, closest_x:closest_x+1]
            crop = np.tile(pixel, (y2 - y1, x2 - x1, 1))
        else:
            # Crop the valid region
            crop = frame[valid_y1:valid_y2, valid_x1:valid_x2]
            
            # Pad the crop if it went out of bounds
            pad_left = max(0, valid_x1 - x1)
            pad_top = max(0, valid_y1 - y1)
            pad_right = max(0, x2 - valid_x2)
            pad_bottom = max(0, y2 - valid_y2)
            
            if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                crop = cv2.copyMakeBorder(
                    crop, pad_top, pad_bottom, pad_left, pad_right, 
                    cv2.BORDER_REPLICATE
                )
        
        if output_size is not None:
            crop = cv2.resize(crop, output_size)
            
        return crop
    
    def _get_template_size(self) -> Tuple[int, int]:
        """Get template crop size (multiple of patch_size)."""
        ps = self.patch_size
        # Template: 8 patches x 8 patches = 112x112 for patch_size=14
        size = ps * 8
        return (size, size)
    
    def _get_search_size(self) -> Tuple[int, int]:
        """Get search region crop size (multiple of patch_size)."""
        ps = self.patch_size
        # Search: 16 patches x 16 patches = 224x224 for patch_size=14
        size = ps * 16
        return (size, size)
    
    def _extract_deep_features(self, crop: np.ndarray) -> torch.Tensor:
        """Extract deep-layer features from a crop using the backbone.
        
        Args:
            crop: (H, W, 3) BGR image
            
        Returns:
            Deep features tensor (B, D, h, w)
        """
        tensor = self.backbone.preprocess(crop, is_bgr=True)
        feat_dict = self.backbone.extract_features(tensor)
        return feat_dict['deep']
        
    def _apply_bg_removal(self, crop: np.ndarray) -> np.ndarray:
        """Remove background from a template crop if enabled.

        Args:
            crop: (H, W, 3) BGR template crop.

        Returns:
            The same crop with background pixels replaced (or unchanged if
            background removal is disabled).
        """
        if not self.bg_removal_enabled or self._bg_remover is None:
            return crop
        masked, _mask = self._bg_remover.remove_background(crop)
        return masked

    def initialize(self, frame: np.ndarray, bbox: tuple):
        """Initialize template from first frame.
        
        Args:
            frame: full frame (H, W, 3) BGR
            bbox: (x, y, w, h) ground truth bounding box
        """
        self.template_bbox = bbox
        template_size = self._get_template_size()
        
        # Crop template with context padding (distorted for backbone)
        self.template_crop = self._crop_and_pad(frame, bbox, self.template_scale, template_size)
        
        # Remove background from coarse template (keep original for display)
        coarse_crop = self._apply_bg_removal(self.template_crop)
        self.coarse_crop = coarse_crop  # kept for debug visualisation
        
        # Undistorted crop for fine stage geometric matching (NO bg removal)
        self.initial_undistorted_crop = self._crop_and_pad(frame, bbox, self.template_scale, None)
        self.initial_bbox = bbox
        
        # Extract features from background-removed crop
        self.initial_template_feats = self._extract_deep_features(coarse_crop)
        self.template_feats = self.initial_template_feats.clone()
        
        # Generate multi-scale templates (bg removal applied inside)
        self._build_multiscale_templates(frame, bbox)
        
        self.frame_count = 1
        logger.info(f"Template initialized: bbox={bbox}, feat_shape={self.template_feats.shape}, bg_removal={self.bg_removal_enabled}")
    
    def _build_multiscale_templates(self, frame: np.ndarray, bbox: tuple):
        """Build template features at multiple scales (with bg removal)."""
        self.template_feats_multiscale = []
        self.template_crops_multiscale = []
        template_size = self._get_template_size()
        
        for scale_factor in self.multi_scale_factors:
            adjusted_scale = self.template_scale * scale_factor
            crop = self._crop_and_pad(frame, bbox, adjusted_scale, template_size)
            self.template_crops_multiscale.append(crop)
            coarse_crop = self._apply_bg_removal(crop)
            feats = self._extract_deep_features(coarse_crop)
            self.template_feats_multiscale.append(feats)
    
    def update(self, frame: np.ndarray, bbox: tuple, confidence: float):
        """Update template crop using image-space EMA continuously every N frames,
        and re-extract deep features from the updated crop.
        
        Args:
            frame: current frame (H, W, 3) BGR
            bbox: (x, y, w, h) current predicted bounding box
            confidence: tracking confidence from the confidence gate
        """
        self.frame_count += 1
        
        if self.frame_count % self.update_interval == 0:
            template_size = self._get_template_size()
            new_crop = self._crop_and_pad(frame, bbox, self.template_scale, template_size)
            
            # 1. Image-space EMA update for main template crop (prevents feature manifold degradation)
            alpha = self.template_update_alpha
            blended_crop = alpha * self.template_crop.astype(np.float32) + (1.0 - alpha) * new_crop.astype(np.float32)
            self.template_crop = np.clip(blended_crop, 0, 255).astype(np.uint8)
            
            # 2. Extract fresh deep features from the updated crop
            coarse_crop = self._apply_bg_removal(self.template_crop)
            self.coarse_crop = coarse_crop  # kept for debug visualisation
            self.template_feats = self._extract_deep_features(coarse_crop)
            
            # Synchronize multi-scale template list so correlation receives updated features
            if len(self.multi_scale_factors) == 1:
                self.template_feats_multiscale = [self.template_feats]
                self.template_crops_multiscale = [self.template_crop]
            else:
                self._update_multiscale_templates(frame, bbox)
            
            logger.debug(f"Template updated (image-space EMA) at frame {self.frame_count}, confidence={confidence:.3f}")

    def _update_multiscale_templates(self, frame: np.ndarray, bbox: tuple):
        """Update multi-scale template crops via image-space EMA and extract updated features."""
        template_size = self._get_template_size()
        alpha = self.template_update_alpha
        new_multiscale_feats = []
        new_multiscale_crops = []
        
        for i, scale_factor in enumerate(self.multi_scale_factors):
            adjusted_scale = self.template_scale * scale_factor
            new_crop = self._crop_and_pad(frame, bbox, adjusted_scale, template_size)
            if hasattr(self, 'template_crops_multiscale') and i < len(self.template_crops_multiscale):
                old_crop = self.template_crops_multiscale[i]
                blended = alpha * old_crop.astype(np.float32) + (1.0 - alpha) * new_crop.astype(np.float32)
                updated_crop = np.clip(blended, 0, 255).astype(np.uint8)
            else:
                updated_crop = new_crop
            
            new_multiscale_crops.append(updated_crop)
            coarse_crop = self._apply_bg_removal(updated_crop)
            feats = self._extract_deep_features(coarse_crop)
            new_multiscale_feats.append(feats)
            
        self.template_crops_multiscale = new_multiscale_crops
        self.template_feats_multiscale = new_multiscale_feats
    
    def get_template_features(self, multi_scale: bool = True) -> list | torch.Tensor:
        """Get current template features.
        
        Args:
            multi_scale: if True, return list of features at multiple scales
            
        Returns:
            List of (B, D, h, w) tensors if multi_scale, else single tensor
        """
        if multi_scale and self.template_feats_multiscale:
            return self.template_feats_multiscale
        return [self.template_feats]
    
    def get_template_crop(self) -> np.ndarray:
        """Get the raw template crop image (for the fine stage matcher)."""
        return self.template_crop
    
    def get_search_features(self, frame: np.ndarray, center: tuple, 
                            target_size: tuple, scale: Optional[float] = None) -> tuple:
        """Extract features from search region.
        
        Args:
            frame: full frame (H, W, 3) BGR
            center: (cx, cy) search center
            target_size: (w, h) current target size
            scale: search region scale factor (default from config)
        
        Returns:
            (features, search_crop, search_bbox):
                features: deep features tensor (B, D, h, w)
                search_crop: raw search crop image
                search_bbox: (x, y, w, h) in image coordinates
        """
        if scale is None:
            scale = self.search_scale
            
        cx, cy = center
        tw, th = target_size
        
        sw, sh = tw * scale, th * scale
        search_bbox = (cx - sw / 2, cy - sh / 2, sw, sh)
        
        search_size = self._get_search_size()
        search_crop = self._crop_and_pad(frame, search_bbox, 1.0, search_size)
        features = self._extract_deep_features(search_crop)
        
        return features, search_crop, search_bbox

    def get_initial_template_features(self) -> torch.Tensor:
        """Get the initial (never-updated) template features for re-detection."""
        return self.initial_template_feats

    def get_fine_template(self) -> Tuple[np.ndarray, tuple]:
        """Get the initial undistorted template crop and bbox for geometric matching."""
        return self.initial_undistorted_crop, self.initial_bbox
