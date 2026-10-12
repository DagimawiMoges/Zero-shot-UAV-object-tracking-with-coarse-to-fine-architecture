# Copyright 2026 Dagimawi Moges Wakumo
# SPDX-License-Identifier: Apache-2.0
"""
Backbone feature extraction supporting DINOv3 (primary) and DINOv2 (fallback).

DINOv3 is loaded via HuggingFace transformers.
DINOv2 is loaded via torch.hub as a fallback for ablations or if DINOv3 is unavailable.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import cv2
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────

DINOV3_MODELS = {
    'dinov3_vits16': {
        'hf_id': 'facebook/dinov3-vits16-pretrain-lvd1689m',
        'dim': 384, 'num_layers': 12, 'patch_size': 16, 'num_register_tokens': 4,
    },
    'dinov3_vitb16': {
        'hf_id': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
        'dim': 768, 'num_layers': 12, 'patch_size': 16, 'num_register_tokens': 4,
    },
    'dinov3_vitl16': {
        'hf_id': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
        'dim': 1024, 'num_layers': 24, 'patch_size': 16, 'num_register_tokens': 4,
    },
    # dim / num_layers / registers below are only defaults: after loading, the
    # values from the checkpoint's own config replace them (see _load_dinov3).
    'dinov3_vits16plus': {
        'hf_id': 'facebook/dinov3-vits16plus-pretrain-lvd1689m',
        'dim': 384, 'num_layers': 12, 'patch_size': 16, 'num_register_tokens': 4,
    },
    'dinov3_vith16plus': {   # ViT-H+/16, 840M params
        'hf_id': 'facebook/dinov3-vith16plus-pretrain-lvd1689m',
        'dim': 1280, 'num_layers': 32, 'patch_size': 16, 'num_register_tokens': 4,
    },
    'dinov3_vit7b16': {      # ViT-7B/16, 6.7B params (largest); use dtype bfloat16
        'hf_id': 'facebook/dinov3-vit7b16-pretrain-lvd1689m',
        'dim': 4096, 'num_layers': 40, 'patch_size': 16, 'num_register_tokens': 4,
    },
}
# The old 'dinov3_vith16' name pointed at a repo that does not exist; keep it as an alias.
DINOV3_MODELS['dinov3_vith16'] = DINOV3_MODELS['dinov3_vith16plus']

DINOV2_MODELS = {
    'dinov2_vits14': {'dim': 384, 'num_layers': 12, 'patch_size': 14},
    'dinov2_vitb14': {'dim': 768, 'num_layers': 12, 'patch_size': 14},
    'dinov2_vitl14': {'dim': 1024, 'num_layers': 24, 'patch_size': 14},
    'dinov2_vitg14': {'dim': 1536, 'num_layers': 40, 'patch_size': 14},
}

ALL_MODELS = {**DINOV3_MODELS, **DINOV2_MODELS}


class DINOBackbone:
    """Frozen DINOv3/DINOv2 feature extractor for tracking.
    
    DINOv3 (default):
        - Loaded via HuggingFace transformers (requires gated access + huggingface-cli login)
        - Patch size 16, includes 4 register tokens + 1 CLS token
        - Uses AutoModel.from_pretrained() and AutoImageProcessor
        
    DINOv2 (fallback):
        - Loaded via torch.hub from facebookresearch/dinov2
        - Patch size 14, uses get_intermediate_layers() API
    """
    
    def __init__(self, model_name: str = 'dinov3_vitl16', device: str = 'cuda', final_norm: bool = True,
                 dtype: str = 'float32'):
        """
        Args:
            dtype: weights/compute dtype for DINOv3: 'float32', 'bfloat16' or 'float16'.
                ViT-7B needs ~27 GB in float32, ~14 GB in bfloat16. Features are
                returned as float32 either way.
            final_norm: DINOv3 only. Apply the model's final LayerNorm to the
                intermediate hidden states. HuggingFace returns hidden_states
                *before* that norm (only last_hidden_state is normed); the raw
                residual stream is dominated by a few huge shared channels, so
                cosine similarity between any two patches is ~0.9+ and the
                correlation map is nearly flat. DINOv2's get_intermediate_layers
                applies the norm by default, so this keeps the two consistent.
                Set False only to reproduce results produced before this fix.
        """
        if model_name not in ALL_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. Available:\n"
                f"  DINOv3: {list(DINOV3_MODELS.keys())}\n"
                f"  DINOv2: {list(DINOV2_MODELS.keys())}"
            )
        
        self.model_name = model_name
        self.final_norm = final_norm
        self.device = torch.device(device)
        self.dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                      'float16': torch.float16}[dtype]
        self.is_dinov3 = model_name in DINOV3_MODELS
        self.info = ALL_MODELS[model_name]
        self.dim = self.info['dim']
        self.num_layers = self.info['num_layers']
        self.patch_size = self.info['patch_size']
        
        if self.is_dinov3:
            self._load_dinov3()
        else:
            self._load_dinov2()
        
        # ImageNet normalization (used for DINOv2; DINOv3 uses its own processor)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def _load_dinov3(self):
        """Load DINOv3 model from HuggingFace."""
        hf_id = self.info['hf_id']
        logger.info(f"Loading DINOv3 model '{hf_id}' from HuggingFace...")
        
        try:
            from transformers import AutoModel, AutoImageProcessor
        except ImportError:
            raise ImportError(
                "HuggingFace transformers is required for DINOv3. Install with:\n"
                "  pip install --upgrade git+https://github.com/huggingface/transformers.git"
            )
        
        try:
            self.processor = AutoImageProcessor.from_pretrained(hf_id)
            self.model = AutoModel.from_pretrained(hf_id, torch_dtype=self.dtype).to(self.device)
        except Exception as e:
            logger.error(
                f"Failed to load DINOv3 model. Make sure you have:\n"
                f"  1. Requested access at https://huggingface.co/{hf_id}\n"
                f"  2. Logged in via: huggingface-cli login\n"
                f"  3. Installed transformers >= 4.56.0\n"
                f"Error: {e}"
            )
            raise
        
        self.model.eval()
        self.model.requires_grad_(False)
        # Trust the checkpoint's own config over the registry defaults
        mcfg = self.model.config
        self.dim = getattr(mcfg, 'hidden_size', self.dim)
        self.num_layers = getattr(mcfg, 'num_hidden_layers', self.num_layers)
        self.patch_size = getattr(mcfg, 'patch_size', self.patch_size)
        self.num_register_tokens = getattr(mcfg, 'num_register_tokens',
                                           self.info.get('num_register_tokens', 4))
        self.num_prefix_tokens = 1 + self.num_register_tokens  # CLS + registers
        
        logger.info(f"DINOv3 loaded: dim={self.dim}, layers={self.num_layers}, dtype={self.dtype}, "
                     f"patch_size={self.patch_size}, prefix_tokens={self.num_prefix_tokens}")
    
    def _load_dinov2(self):
        """Load DINOv2 model via torch.hub."""
        logger.info(f"Loading DINOv2 model '{self.model_name}' via torch.hub...")
        
        try:
            self.model = torch.hub.load('facebookresearch/dinov2', self.model_name)
        except Exception as e:
            logger.error(f"Failed to load DINOv2 model via torch.hub: {e}")
            raise
        
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.processor = None
        self.num_prefix_tokens = 0  # DINOv2 get_intermediate_layers strips CLS by default
        
        logger.info(f"DINOv2 loaded: dim={self.dim}, layers={self.num_layers}, "
                     f"patch_size={self.patch_size}")
    
    @torch.no_grad()
    def extract_features(self, image: torch.Tensor, layers: Optional[tuple] = None,
                         l2_norm: bool = True) -> dict:
        """Extract patch-level features from specified layers.
        
        Args:
            image: Preprocessed tensor. For DINOv2: (B, 3, H, W) normalized, on device.
                   For DINOv3: (B, 3, H, W) normalized via processor, on device.
                   H, W must be multiples of patch_size.
            layers: tuple of layer indices to extract. If None, use last 4 layers.
            l2_norm: whether to L2-normalize the spatial features.
        
        Returns:
            dict with keys:
                'features': list of (B, D, H//patch_size, W//patch_size) tensors
                'deep': (B, D, h, w) — deepest layer features (for coarse stage)
                'mid': (B, D, h, w) — mid-level features (for fine stage detail)
        """
        B, C, H, W = image.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Image dimensions ({H}, {W}) must be multiples of patch_size {self.patch_size}"
            )
        
        h_feat = H // self.patch_size
        w_feat = W // self.patch_size
        n_layers = len(layers) if layers else 4
        
        if self.is_dinov3:
            spatial_features = self._extract_dinov3(image, n_layers, h_feat, w_feat)
        else:
            spatial_features = self._extract_dinov2(image, n_layers, h_feat, w_feat)
        
        if l2_norm:
            spatial_features = [F.normalize(f, dim=1) for f in spatial_features]
        
        return {
            'features': spatial_features,
            'deep': spatial_features[-1],
            'mid': spatial_features[len(spatial_features) // 2] if len(spatial_features) > 1 else spatial_features[0],
        }
    
    def _extract_dinov3(self, image: torch.Tensor, n_layers: int,
                        h_feat: int, w_feat: int) -> list[torch.Tensor]:
        """Extract features from DINOv3 via HuggingFace forward pass.
        
        DINOv3 outputs:
            last_hidden_state: (B, 1+num_registers+num_patches, D)
            hidden_states (if output_hidden_states=True): tuple of (B, N, D) per layer
        
        We strip the CLS and register tokens (first num_prefix_tokens) to get
        only patch tokens, then reshape to spatial maps.
        """
        B = image.shape[0]
        
        outputs = self.model(image.to(self.dtype), output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple of (B, N, D) for each layer
        
        # Take last n_layers (skip the embedding layer at index 0)
        selected = hidden_states[-n_layers:]
        
        norm = getattr(self.model, 'norm', None) if self.final_norm else None
        spatial_features = []
        for feat in selected:
            if norm is not None:
                feat = norm(feat)   # same LayerNorm that produces last_hidden_state
            # Strip CLS + register tokens → keep only patch tokens
            patch_tokens = feat[:, self.num_prefix_tokens:, :]  # (B, h*w, D)
            # Reshape to spatial map
            spatial = patch_tokens.transpose(1, 2).reshape(B, self.dim, h_feat, w_feat)
            spatial_features.append(spatial.float())
        
        return spatial_features
    
    def _extract_dinov2(self, image: torch.Tensor, n_layers: int,
                        h_feat: int, w_feat: int) -> list[torch.Tensor]:
        """Extract features from DINOv2 via get_intermediate_layers API.
        
        get_intermediate_layers returns patch tokens already (no CLS) when
        return_class_token=False.
        """
        B = image.shape[0]
        
        features = self.model.get_intermediate_layers(
            image, n=n_layers, return_class_token=False
        )
        
        spatial_features = []
        for feat in features:
            spatial = feat.transpose(1, 2).reshape(B, self.dim, h_feat, w_feat)
            spatial_features.append(spatial)
        
        return spatial_features
    
    def preprocess(self, image: np.ndarray, is_bgr: bool = True) -> torch.Tensor:
        """Preprocess a numpy image for the model.
        
        For DINOv3: uses the HuggingFace AutoImageProcessor.
        For DINOv2: manual ImageNet normalization.
        
        Args:
            image: (H, W, 3) numpy image
            is_bgr: whether the image is in BGR format (cv2 default)
            
        Returns:
            (1, 3, H', W') tensor on device, where H' and W' are multiples of patch_size.
        """
        # Convert to RGB
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and is_bgr:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.is_dinov3:
            return self._preprocess_dinov3(image)
        else:
            return self._preprocess_dinov2(image)
    
    def _preprocess_dinov3(self, image_rgb: np.ndarray) -> torch.Tensor:
        """Preprocess for DINOv3 using HuggingFace processor.
        
        The HF processor handles resizing and normalization, but we need to
        ensure the output is a multiple of patch_size for spatial feature maps.
        """
        # First pad to patch multiple
        H, W = image_rgb.shape[:2]
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            image_rgb = cv2.copyMakeBorder(
                image_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE
            )
        
        # Use the HF processor (handles normalization)
        # do_resize=False because we've already sized it to a patch multiple
        inputs = self.processor(
            images=image_rgb, return_tensors="pt",
            do_resize=False, do_center_crop=False
        )
        pixel_values = inputs['pixel_values'].to(self.device)
        
        return pixel_values
    
    def _preprocess_dinov2(self, image_rgb: np.ndarray) -> torch.Tensor:
        """Preprocess for DINOv2 with manual ImageNet normalization."""
        img_tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1) / 255.0
        img_tensor = self.pad_to_patch_multiple(img_tensor)
        img_tensor = self.normalize(img_tensor)
        return img_tensor.unsqueeze(0).to(self.device)
    
    def pad_to_patch_multiple(self, image: torch.Tensor) -> torch.Tensor:
        """Pad image so H and W are multiples of patch_size."""
        _, H, W = image.shape[-3:]
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        
        if pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h), mode='replicate')
        
        return image


# ──────────────────────────────────────────────
# Backward compatibility alias
# ──────────────────────────────────────────────
DINOv2Backbone = DINOBackbone
