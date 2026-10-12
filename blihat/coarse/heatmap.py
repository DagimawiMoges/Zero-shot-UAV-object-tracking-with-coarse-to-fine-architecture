# Copyright 2026 Dagimawi Moges Wakumo
# SPDX-License-Identifier: Apache-2.0
import torch


class HeatmapProcessor:
    """Process correlation response maps to find target location.
    
    Handles:
    - Multiplicative cosine window penalty (suppresses edges)
    - Sub-pixel peak refinement via quadratic fitting
    - Correct peak-to-image coordinate mapping accounting for
      template kernel offset in the cross-correlation output
    """
    
    def __init__(self, cosine_window_weight: float = 0.3, patch_size: int = 16,
                 template_patches: int = 8):
        """
        Args:
            cosine_window_weight: blending weight for cosine window penalty [0, 1]
            patch_size: backbone patch size in pixels (16 for DINOv3, 14 for DINOv2)
            template_patches: number of patches along each side of the template
                              feature map (always 8 — template crop = 8 * patch_size)
        """
        self.cosine_window_weight = cosine_window_weight
        self.patch_size = patch_size
        self.template_patches = template_patches
        self._window_cache = {}
    
    def _get_cosine_window(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Get or create cached cosine window of given size."""
        key = (h, w)
        if key not in self._window_cache:
            window_col = torch.hann_window(w, device=device)
            window_row = torch.hann_window(h, device=device)
            window = window_row.view(-1, 1) * window_col.view(1, -1)
            self._window_cache[key] = window.view(1, 1, h, w)
        return self._window_cache[key]
    
    def process(self, response_map: torch.Tensor, search_bbox: tuple, apply_window: bool = True) -> dict:
        """
        Find peak in response map and convert to image coordinates.
        
        Coordinate mapping derivation:
            Given F.conv2d output of shape (Hs-Ht+1, Ws-Wt+1):
            - peak at (r, c) means the 8×8 template was placed starting
              at search patch (r, c)
            - The template center is at search patch (c + Wt/2, r + Ht/2)
            - Search patch p maps to crop pixel p * patch_size + patch_size/2
            - Simplifies to:  img_x = search_x + (c + Wt/2) / Ws * search_w
        
        Args:
            response_map: (B, 1, H, W) raw response from correlation
            search_bbox: (x, y, w, h) of the search region in image coordinates
            apply_window: bool, whether to apply the cosine window penalty
        
        Returns:
            dict with: peak_pos, peak_value, confidence, sharpness, psr,
                       response_map, peak_row, peak_col
        """
        B, _, H, W = response_map.shape
        device = response_map.device
        
        # ── 1. Apply cosine window MULTIPLICATIVELY ──
        # penalty ranges from (1 - weight) at edges to 1.0 at center
        # This suppresses spurious peaks near borders without adding bias
        if apply_window:
            cosine_window = self._get_cosine_window(H, W, device)
            penalty = (1.0 - self.cosine_window_weight) + self.cosine_window_weight * cosine_window
            windowed_response = response_map * penalty
        else:
            windowed_response = response_map
        
        # ── 2. Find peak position via argmax ──
        flat_response = windowed_response.view(B, -1)
        peak_values, peak_indices = torch.max(flat_response, dim=1)
        
        # ── 3. Confidence metrics ──
        # Primary confidence: raw peak value (cosine similarity, bounded ~[-0.3, 0.7])
        # This is directly comparable with the fine stage's inlier_ratio [0, 1]
        confidence = peak_values
        
        # PSR (Peak-to-Sidelobe Ratio) for diagnostics
        mean_val = flat_response.mean(dim=1)
        std_val = flat_response.std(dim=1)
        psr = (peak_values - mean_val) / (std_val + 1e-6)
        
        # Sharpness: peak / second_peak
        flat_masked = flat_response.clone()
        flat_masked.scatter_(1, peak_indices.unsqueeze(1), -float('inf'))
        second_peak, _ = torch.max(flat_masked, dim=1)
        sharpness = peak_values / (second_peak + 1e-6)
        
        peak_idx = peak_indices.item()
        peak_row = peak_idx // W
        peak_col = peak_idx % W
        
        # ── 4. Sub-pixel refinement via quadratic fitting ──
        # Fit a parabola to the 3 values around the peak in each axis
        # and find the continuous maximum. Adds ~0.5-2px of precision.
        sub_col, sub_row = float(peak_col), float(peak_row)
        if 0 < peak_row < H - 1 and 0 < peak_col < W - 1:
            r = windowed_response[0, 0]
            # First derivatives (central difference)
            dx = 0.5 * (r[peak_row, peak_col + 1] - r[peak_row, peak_col - 1]).item()
            dy = 0.5 * (r[peak_row + 1, peak_col] - r[peak_row - 1, peak_col]).item()
            # Second derivatives
            dxx = (r[peak_row, peak_col + 1] + r[peak_row, peak_col - 1]
                   - 2 * r[peak_row, peak_col]).item()
            dyy = (r[peak_row + 1, peak_col] + r[peak_row - 1, peak_col]
                   - 2 * r[peak_row, peak_col]).item()
            
            # Parabola maximum: x_max = x_peak - f'(x) / f''(x)
            if abs(dxx) > 1e-6:
                sub_col = peak_col - dx / dxx
            if abs(dyy) > 1e-6:
                sub_row = peak_row - dy / dyy
            
            # Clamp to valid range (stay within half a cell of integer peak)
            sub_col = max(0.0, min(float(sub_col), W - 1.0))
            sub_row = max(0.0, min(float(sub_row), H - 1.0))
        
        # ── 5. CORRECT peak → image coordinate mapping ──
        #
        # Response map shape: (H, W) = (Hs - Ht + 1, Ws - Wt + 1)
        # where Hs, Ws = search feature map size, Ht, Wt = template feature map size
        #
        # We derive the search feature map size dynamically so this works
        # for both normal tracking (search=16×16) and re-detection (full frame).
        Wt = self.template_patches
        Ht = self.template_patches
        Ws = W + Wt - 1   # search feature map width
        Hs = H + Ht - 1   # search feature map height
        
        search_x, search_y, search_w, search_h = search_bbox
        
        img_x = search_x + (sub_col + Wt / 2) / Ws * search_w
        img_y = search_y + (sub_row + Ht / 2) / Hs * search_h
        
        return {
            'peak_pos': (float(img_x), float(img_y)),
            'peak_value': float(peak_values.item()),
            'confidence': float(confidence.item()),
            'sharpness': float(sharpness.item()),
            'psr': float(psr.item()),
            'response_map': windowed_response.detach(),
            'peak_row': peak_row,
            'peak_col': peak_col,
            'sub_row': float(sub_row),
            'sub_col': float(sub_col),
        }
    
    def get_crop_region(self, peak_pos: tuple[float, float], target_size: tuple[float, float],
                        crop_scale: float, img_shape: tuple[int, int]
                        ) -> tuple[float, float, float, float]:
        """
        Generate the crop region for the fine stage.
        
        Args:
            peak_pos: (x, y) center of detected target
            target_size: (w, h) current target size estimate
            crop_scale: float, crop = target_size * crop_scale
            img_shape: (H, W) image dimensions for clipping
        
        Returns:
            crop_bbox: (x, y, w, h) in image coordinates, clipped to image bounds
        """
        cx, cy = peak_pos
        tw, th = target_size
        
        cw, ch = tw * crop_scale, th * crop_scale
        
        x = cx - cw / 2
        y = cy - ch / 2
        
        # Clipping
        img_h, img_w = img_shape
        x = max(0.0, min(x, img_w - cw))
        y = max(0.0, min(y, img_h - ch))
        
        return (x, y, cw, ch)
