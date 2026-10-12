# Copyright 2026 Dagimawi Moges Wakumo
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch
from typing import Union

TensorOrArray = Union[np.ndarray, torch.Tensor]

def _is_tensor(x: TensorOrArray) -> bool:
    return isinstance(x, torch.Tensor)

def xywh_to_xyxy(bbox: TensorOrArray) -> TensorOrArray:
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
    x2 = x + w
    y2 = y + h
    if _is_tensor(bbox):
        return torch.stack([x, y, x2, y2], dim=-1)
    else:
        return np.stack([x, y, x2, y2], axis=-1)

def xyxy_to_xywh(bbox: TensorOrArray) -> TensorOrArray:
    """Convert [x1, y1, x2, y2] to [x, y, w, h]."""
    x1, y1, x2, y2 = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
    w = x2 - x1
    h = y2 - y1
    if _is_tensor(bbox):
        return torch.stack([x1, y1, w, h], dim=-1)
    else:
        return np.stack([x1, y1, w, h], axis=-1)

def xywh_to_cxcywh(bbox: TensorOrArray) -> TensorOrArray:
    """Convert [x, y, w, h] to [cx, cy, w, h]."""
    x, y, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
    cx = x + w / 2.0
    cy = y + h / 2.0
    if _is_tensor(bbox):
        return torch.stack([cx, cy, w, h], dim=-1)
    else:
        return np.stack([cx, cy, w, h], axis=-1)

def cxcywh_to_xywh(bbox: TensorOrArray) -> TensorOrArray:
    """Convert [cx, cy, w, h] to [x, y, w, h]."""
    cx, cy, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
    x = cx - w / 2.0
    y = cy - h / 2.0
    if _is_tensor(bbox):
        return torch.stack([x, y, w, h], dim=-1)
    else:
        return np.stack([x, y, w, h], axis=-1)

def compute_iou(bbox1: TensorOrArray, bbox2: TensorOrArray) -> float:
    """Compute IoU between two bboxes in [x, y, w, h] format."""
    b1 = xywh_to_xyxy(bbox1)
    b2 = xywh_to_xyxy(bbox2)
    
    if _is_tensor(bbox1):
        x1 = torch.max(b1[0], b2[0])
        y1 = torch.max(b1[1], b2[1])
        x2 = torch.min(b1[2], b2[2])
        y2 = torch.min(b1[3], b2[3])
        
        inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union_area = area1 + area2 - inter_area
        return float(inter_area / (union_area + 1e-6))
    else:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union_area = area1 + area2 - inter_area
        return float(inter_area / (union_area + 1e-6))

def clip_bbox(bbox: TensorOrArray, img_shape: tuple) -> TensorOrArray:
    """Clip bbox [x, y, w, h] to image bounds (h, w)."""
    h, w = img_shape[:2]
    b = xywh_to_xyxy(bbox)
    if _is_tensor(bbox):
        b[..., 0] = torch.clamp(b[..., 0], min=0, max=w)
        b[..., 1] = torch.clamp(b[..., 1], min=0, max=h)
        b[..., 2] = torch.clamp(b[..., 2], min=0, max=w)
        b[..., 3] = torch.clamp(b[..., 3], min=0, max=h)
    else:
        b[..., 0] = np.clip(b[..., 0], 0, w)
        b[..., 1] = np.clip(b[..., 1], 0, h)
        b[..., 2] = np.clip(b[..., 2], 0, w)
        b[..., 3] = np.clip(b[..., 3], 0, h)
    return xyxy_to_xywh(b)

def bbox_area(bbox: TensorOrArray) -> Union[float, torch.Tensor]:
    """Compute area of bbox [x, y, w, h]."""
    return bbox[..., 2] * bbox[..., 3]

def scale_bbox(bbox: TensorOrArray, scale_factor: float) -> TensorOrArray:
    """Scale bbox [x, y, w, h] from its center."""
    cxcywh = xywh_to_cxcywh(bbox)
    cxcywh[..., 2] = cxcywh[..., 2] * scale_factor
    cxcywh[..., 3] = cxcywh[..., 3] * scale_factor
    return cxcywh_to_xywh(cxcywh)
