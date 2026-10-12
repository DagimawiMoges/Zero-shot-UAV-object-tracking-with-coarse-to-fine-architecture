# Copyright 2026 Dagimawi Moges Wakumo
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F


class DepthwiseCorrelation:
    """Compute depthwise cross-correlation between template and search features.
    
    This is the core operation of Siamese tracking: the template feature map
    acts as a convolution kernel applied to the search region feature map.
    Each channel is correlated independently (depthwise), then aggregated.
    """
    
    def __call__(self, search_feat: torch.Tensor, template_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            search_feat: (B, C, Hs, Ws) search region features
            template_feat: (B, C, Ht, Wt) template features
        
        Returns:
            response_map: (B, 1, Ho, Wo) where Ho = Hs-Ht+1, Wo = Ws-Wt+1
            This is the mean over channels of the depthwise correlation.
        """
        B, C, Hs, Ws = search_feat.shape
        _, _, Ht, Wt = template_feat.shape
        
        # 1. Reshape template to (B*C, 1, Ht, Wt)
        template_kernel = template_feat.view(B * C, 1, Ht, Wt)
        
        # 2. Reshape search to (1, B*C, Hs, Ws)
        search_input = search_feat.view(1, B * C, Hs, Ws)
        
        # 3. F.conv2d with groups=B*C for depthwise correlation
        corr = F.conv2d(search_input, template_kernel, groups=B * C)
        
        Ho, Wo = corr.shape[2:]
        
        # 4. Reshape output to (B, C, Ho, Wo)
        corr = corr.view(B, C, Ho, Wo)
        
        # 5. Calculate average cosine similarity
        # Summing across channels gives the dot product (cosine similarity since features are L2 normalized)
        # Then we divide by the spatial area (Ht * Wt) to get the average similarity in [-1, 1]
        response_map = corr.sum(dim=1, keepdim=True) / (Ht * Wt)
        
        return response_map
        
    def multi_scale_correlate(self, search_feat: torch.Tensor, template_feats_list: list[torch.Tensor], scale_penalty: float = 0.9745) -> tuple[torch.Tensor, torch.Tensor]:
        """Correlate search against multiple template scales, take element-wise max.
        
        Args:
            search_feat: (B, C, Hs, Ws)
            template_feats_list: list of (B, C, Ht_i, Wt_i) at different scales
            scale_penalty: multiplier penalty for non-center scales to prevent scale drift
        
        Returns:
            best_response: (B, 1, Ho, Wo) max response across scales
            best_scale_idx: (B, 1, Ho, Wo) which scale produced the max at each position
        """
        responses = []
        center_idx = len(template_feats_list) // 2
        
        for i, template_feat in enumerate(template_feats_list):
            resp = self.__call__(search_feat, template_feat)
            
            # Apply scale penalty to discourage scale change unless significantly better
            if i != center_idx:
                resp = resp * scale_penalty
                
            responses.append(resp)
            
        # We need to pad responses to the same size (which is the size of the smallest template response)
        # However, typically Ho = Hs - Ht + 1. The smallest template gives the largest response map.
        # Actually, let's pad all to the max size in the list.
        max_h = max(r.shape[2] for r in responses)
        max_w = max(r.shape[3] for r in responses)
        
        padded_responses = []
        for r in responses:
            pad_h = max_h - r.shape[2]
            pad_w = max_w - r.shape[3]
            
            # Padding: (left, right, top, bottom)
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            
            padded = F.pad(r, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=-float('inf'))
            padded_responses.append(padded)
            
        stacked_responses = torch.stack(padded_responses, dim=0) # (num_scales, B, 1, H, W)
        
        best_response, best_scale_idx = torch.max(stacked_responses, dim=0)
        
        return best_response, best_scale_idx
