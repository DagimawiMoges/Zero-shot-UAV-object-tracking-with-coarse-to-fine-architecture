# Copyright 2026 Dagimawi Moges Wakumo
# SPDX-License-Identifier: Apache-2.0
import torch
import numpy as np
import cv2
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class LightGlueMatcher:
    """Sparse keypoint matching using SuperPoint + LightGlue.
    
    This is the default fine-stage matcher. It extracts SuperPoint keypoints
    from both the template crop and the search crop, then uses LightGlue
    to find correspondences. The matched points are passed to the geometry
    module for bbox estimation.
    """
    
    def __init__(self, max_keypoints: int = 1024, device: str = 'cuda'):
        self.device = device
        self.max_keypoints = max_keypoints
        try:
            from lightglue import LightGlue, SuperPoint
        except ImportError:
            raise ImportError(
                "LightGlue is not installed. Please install it using: "
                "pip install git+https://github.com/cvg/LightGlue.git"
            )
            
        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
        self.matcher = LightGlue(features='superpoint').eval().to(device)

    def _to_grayscale_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert BGR numpy image to (1, 1, H, W) grayscale tensor.
        
        Args:
            image: (H, W, 3) BGR numpy image.
            
        Returns:
            torch.Tensor: (1, 1, H, W) grayscale float32 tensor with values in [0, 1].
        """
        if image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 2:
            gray = image
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
            
        # Convert to float32 in [0, 1]
        tensor = torch.from_numpy(gray).float() / 255.0
        # Add batch and channel dims: (1, 1, H, W)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def match(self, template_crop: np.ndarray, search_crop: np.ndarray) -> Dict[str, Any]:
        """
        Find keypoint correspondences between template and search crops.
        
        Args:
            template_crop: (H, W, 3) BGR numpy image
            search_crop: (H, W, 3) BGR numpy image
        
        Returns:
            dict with:
                'kpts_template': (N, 2) matched keypoints in template crop coords
                'kpts_search': (N, 2) matched keypoints in search crop coords  
                'confidence': (N,) match confidence scores
                'num_matches': int
                'num_keypoints_template': int, total keypoints found in template
                'num_keypoints_search': int, total keypoints found in search
        """
        # 1. Convert both crops to grayscale float32 tensors (B, 1, H, W) in [0, 1]
        t_tensor = self._to_grayscale_tensor(template_crop)
        s_tensor = self._to_grayscale_tensor(search_crop)
        
        # 2. Extract SuperPoint features from both
        feats0 = self.extractor.extract(t_tensor)
        feats1 = self.extractor.extract(s_tensor)
        
        num_kpts_t = feats0['keypoints'].shape[1] if 'keypoints' in feats0 else 0
        num_kpts_s = feats1['keypoints'].shape[1] if 'keypoints' in feats1 else 0
        
        if num_kpts_t == 0 or num_kpts_s == 0:
            logger.debug("No keypoints found in one of the images.")
            return {
                'kpts_template': np.empty((0, 2), dtype=np.float32),
                'kpts_search': np.empty((0, 2), dtype=np.float32),
                'confidence': np.empty((0,), dtype=np.float32),
                'num_matches': 0,
                'num_keypoints_template': num_kpts_t,
                'num_keypoints_search': num_kpts_s
            }
            
        # 3. Run LightGlue matcher
        result = self.matcher({'image0': feats0, 'image1': feats1})
        
        # 4. Extract matched keypoint coordinates
        matches = result['matches'][0]  # shape (N, 2)
        scores = result['scores'][0]    # shape (N,)
        
        if matches.shape[0] == 0:
            logger.debug("No matches found between template and search.")
            return {
                'kpts_template': np.empty((0, 2), dtype=np.float32),
                'kpts_search': np.empty((0, 2), dtype=np.float32),
                'confidence': np.empty((0,), dtype=np.float32),
                'num_matches': 0,
                'num_keypoints_template': num_kpts_t,
                'num_keypoints_search': num_kpts_s
            }
            
        kpts0 = feats0['keypoints'][0]  # shape (K0, 2)
        kpts1 = feats1['keypoints'][0]  # shape (K1, 2)
        
        m_kpts0 = kpts0[matches[..., 0]].cpu().numpy()
        m_kpts1 = kpts1[matches[..., 1]].cpu().numpy()
        scores_np = scores.cpu().numpy()
        
        return {
            'kpts_template': m_kpts0,
            'kpts_search': m_kpts1,
            'confidence': scores_np,
            'num_matches': matches.shape[0],
            'num_keypoints_template': num_kpts_t,
            'num_keypoints_search': num_kpts_s
        }
