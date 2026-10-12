# Copyright 2026 Dagimawi Moges Wakumo
# SPDX-License-Identifier: Apache-2.0
"""
Quick check (a few minutes) of how discriminative the coarse DINO response is,
with and without the DINOv3 final LayerNorm. Run this before a long benchmark.

    python eval/check_features.py --config configs/adaptive_recovery.yaml --dataset-root UAV123 \
        --sequences person7_2,car6_3,uav3,bird1_1,wakeboard9 --offset 10

For each sequence the tracker is initialised on the first frame and the search
crop of frame <offset> is centred on the ground truth. Reported per setting:
  peak     best raw cosine response
  median   median response over the search map (background level)
  contrast peak - median (higher = target stands out)
  psr      peak-to-sidelobe ratio
  err      distance of the argmax from the true center, in target sizes
With the pre-fix features expect peak ~0.94 and contrast near zero.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blihat.adaptive_tracker import AdaptiveTracker  # noqa: E402
from blihat.backbone.dinov2 import DINOBackbone       # noqa: E402
from blihat.config import Config                      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--dataset-root', required=True)
    ap.add_argument('--sequences', default='person7_2,car6_3,uav3,bird1_1,wakeboard9,boat6,group3_4')
    ap.add_argument('--offset', type=int, default=10, help='frame index of the search frame')
    args = ap.parse_args()

    from got10k.datasets import UAV123
    ds = UAV123(args.dataset_root)
    cfg = Config.from_yaml(args.config)
    backbone = DINOBackbone(cfg.backbone.name, device=cfg.device, dtype=cfg.backbone.dtype)

    print(f"{'sequence':12s} {'final_norm':>10s} {'peak':>6s} {'median':>7s} {'contrast':>8s} {'psr':>6s} {'err':>5s}")
    for name in [s.strip() for s in args.sequences.split(',') if s.strip()]:
        img_files, anno = ds[name]
        k = min(args.offset, len(img_files) - 1)
        if np.isnan(anno[k]).any():
            print(f"{name:12s} target absent at frame {k}, skipped")
            continue
        f0, fk = cv2.imread(img_files[0]), cv2.imread(img_files[k])
        g = anno[k]
        center = (g[0] + g[2] / 2, g[1] + g[3] / 2)
        for fn in (False, True):
            backbone.final_norm = fn
            trk = AdaptiveTracker(cfg, backbone=backbone)
            trk.init(f0, tuple(anno[0]))
            trk.target_size = (float(g[2]), float(g[3]))      # isolate appearance from scale error
            with torch.no_grad():
                co = trk._coarse(fk, center)
            resp = co['heatmap'][0, 0].float().cpu().numpy()
            raw_med = float(np.median(resp))
            err = np.hypot(co['center'][0] - center[0], co['center'][1] - center[1]) / np.sqrt(g[2] * g[3])
            print(f"{name:12s} {str(fn):>10s} {co['peak']:6.3f} {raw_med:7.3f} {co['peak'] - raw_med:8.3f} "
                  f"{co['psr']:6.2f} {err:5.2f}")


if __name__ == '__main__':
    main()
