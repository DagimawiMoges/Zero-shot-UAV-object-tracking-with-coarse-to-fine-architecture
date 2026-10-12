# Copyright 2026 Dagimawi Moges Wakumo
# SPDX-License-Identifier: Apache-2.0
import cv2
import numpy as np
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class GeometricEstimator:
    """Estimate target bounding box from point correspondences.
    
    Given matched keypoints between template and search images,
    estimates a geometric transformation (homography or affine)
    and projects the template bounding box into the search image.
    """
    
    def __init__(self, ransac_threshold: float = 3.0, 
                 ransac_method: str = 'USAC_MAGSAC',
                 area_ratio_bounds: tuple = (0.25, 4.0)):
        
        self.ransac_threshold = ransac_threshold
        self.area_ratio_bounds = area_ratio_bounds
        
        # Map ransac_method string to cv2 constant
        method_map = {
            'RANSAC': cv2.RANSAC,
            'LMEDS': cv2.LMEDS,
            'RHO': cv2.RHO,
            'USAC_DEFAULT': cv2.USAC_DEFAULT,
            'USAC_PARALLEL': cv2.USAC_PARALLEL,
            'USAC_FM_8PTS': cv2.USAC_FM_8PTS,
            'USAC_FAST': cv2.USAC_FAST,
            'USAC_ACCURATE': cv2.USAC_ACCURATE,
            'USAC_PROSAC': cv2.USAC_PROSAC,
            'USAC_MAGSAC': cv2.USAC_MAGSAC
        }
        self.ransac_method = method_map.get(ransac_method.upper(), cv2.USAC_MAGSAC)

    def estimate_bbox(self, kpts_template: np.ndarray, kpts_search: np.ndarray,
                      template_bbox: tuple) -> Dict[str, Any]:
        """
        Estimate target bbox in search image from point correspondences.
        
        Args:
            kpts_template: (N, 2) matched keypoints in template coordinates
            kpts_search: (N, 2) matched keypoints in search coordinates
            template_bbox: (x, y, w, h) target bbox within the template crop
        
        Returns:
            dict with:
                'bbox': (x, y, w, h) estimated bbox in search crop coords, or None if failed
                'homography': (3, 3) estimated homography matrix, or None
                'num_inliers': int
                'inlier_ratio': float (num_inliers / total_matches)
                'reprojection_error': float (mean reprojection error of inliers)
                'inlier_mask': (N,) boolean mask
                'projected_corners': (4, 2) projected bbox corners
                'success': bool
        """
        N = len(kpts_template)
        
        # Default failure result
        fail_res = {
            'bbox': None,
            'homography': None,
            'num_inliers': 0,
            'inlier_ratio': 0.0,
            'reprojection_error': float('inf'),
            'inlier_mask': np.zeros((N,), dtype=bool),
            'projected_corners': None,
            'success': False
        }
        
        if N < 3:
            logger.debug(f"Not enough points to estimate transformation: {N}")
            return fail_res
            
        transform = None
        inliers = None
        is_homography = False
        
        # 1. Try Partial Affine (Translation, Rotation, Uniform Scale) - most robust for UAVs
        if N >= 3:
            transform, inliers = cv2.estimateAffinePartial2D(
                kpts_template, kpts_search, method=cv2.RANSAC, 
                ransacReprojThreshold=self.ransac_threshold
            )
            is_homography = False
            
        # 2. Fallback to Homography only if target has intense perspective shifts
        if transform is None or (inliers is not None and inliers.sum() < 3):
            if N >= 4:
                transform, inliers = cv2.findHomography(
                    kpts_template, kpts_search, self.ransac_method, self.ransac_threshold
                )
                is_homography = True
            
        if transform is None or inliers is None or inliers.sum() < 3:
            logger.debug("Failed to estimate a valid transformation.")
            return fail_res
            
        inlier_mask = inliers.ravel().astype(bool)
        num_inliers = int(inlier_mask.sum())
        inlier_ratio = num_inliers / N if N > 0 else 0.0
        
        # 4. Project template bbox corners through transform
        projected_corners = self._project_bbox_corners(template_bbox, transform, is_homography)
        if projected_corners is None:
            return fail_res
            
        # 5. Get axis-aligned bbox from projected corners
        bbox = self._corners_to_bbox(projected_corners)
        
        # 6. Sanity checks:
        x, y, w, h = bbox
        _, _, tw, th = template_bbox
        template_area = tw * th
        bbox_area = w * h
        
        if template_area <= 0 or bbox_area <= 0:
            return fail_res
            
        area_ratio = bbox_area / template_area
        if not (self.area_ratio_bounds[0] <= area_ratio <= self.area_ratio_bounds[1]):
            logger.debug(f"Area ratio {area_ratio:.2f} out of bounds.")
            return fail_res
            
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > 5.0 or aspect_ratio < 0.2:
            logger.debug(f"Aspect ratio {aspect_ratio:.2f} too extreme.")
            return fail_res
            
        # Check if corners are completely invalid (though they can be outside the image, we just check for extreme degeneracy)
        # 7. Compute reprojection error for inliers
        repr_err = self._compute_reprojection_error(
            kpts_template, kpts_search, transform, inlier_mask, is_homography
        )
        
        # Ensure homography is always 3x3 for the output dict
        homography_matrix = transform
        if not is_homography and transform is not None:
            homography_matrix = np.vstack([transform, [0, 0, 1]])
            
        return {
            'bbox': bbox,
            'homography': homography_matrix,
            'num_inliers': num_inliers,
            'inlier_ratio': inlier_ratio,
            'reprojection_error': repr_err,
            'inlier_mask': inlier_mask,
            'projected_corners': projected_corners,
            'success': True
        }

    def _project_bbox_corners(self, bbox: tuple, transform: np.ndarray, 
                               is_homography: bool) -> Optional[np.ndarray]:
        """Project bbox corners through homography or affine transform."""
        x, y, w, h = bbox
        # bbox corners: TL, TR, BR, BL
        corners = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        if is_homography:
            # perspectiveTransform requires (N, 1, 2) and 3x3 transform
            try:
                projected = cv2.perspectiveTransform(corners, transform)
            except cv2.error:
                return None
        else:
            # transform requires (N, 1, 2) and 2x3 transform
            try:
                projected = cv2.transform(corners, transform)
            except cv2.error:
                return None
                
        return projected.reshape(4, 2)

    def _corners_to_bbox(self, corners: np.ndarray) -> tuple:
        """Convert 4 projected corners to axis-aligned (x, y, w, h) bbox."""
        min_x = np.min(corners[:, 0])
        max_x = np.max(corners[:, 0])
        min_y = np.min(corners[:, 1])
        max_y = np.max(corners[:, 1])
        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def _compute_reprojection_error(self, kpts_src: np.ndarray, kpts_dst: np.ndarray, 
                                      transform: np.ndarray, inlier_mask: np.ndarray, 
                                      is_homography: bool) -> float:
        """Compute mean reprojection error for inlier matches."""
        if not inlier_mask.any():
            return float('inf')
            
        src_inliers = kpts_src[inlier_mask].reshape(-1, 1, 2).astype(np.float32)
        dst_inliers = kpts_dst[inlier_mask].reshape(-1, 1, 2).astype(np.float32)
        
        if is_homography:
            proj_src = cv2.perspectiveTransform(src_inliers, transform)
        else:
            proj_src = cv2.transform(src_inliers, transform)
            
        errors = np.linalg.norm(proj_src - dst_inliers, axis=-1)
        return float(np.mean(errors))
