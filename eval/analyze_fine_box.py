# Copyright 2026 Dagimawi Moges Wakumo
# SPDX-License-Identifier: Apache-2.0
"""
Box-quality and box-source analysis for the split DINO (where) / LightGlue (box) design.

    python eval/analyze_fine_box.py --results RESULTS/UAV123/Blihat-finebox --anno UAV123/anno/UAV123
        [--sequences eval/uav123_scale.txt] [--near 0.3]

Needs diagnostics saved by eval/run_benchmark.py from a run with fine_enabled: true
(fields dino_box / fine_box / box_source / fine_reject). Reports:

  1. Box source by target size: how often the output box came from LightGlue vs the
     DINO fallback (and why the fit was not used), binned by GT size sqrt(w*h) in px.
  2. Box quality: on frames where the DINO center is within --near target sizes of
     the GT center (|dc| / sqrt(gt_w*gt_h)) and a LightGlue fit exists, IoU of the DINO
     box vs the LightGlue box against GT; split by reliability verdict, size and inliers.
  3. Size adaptation: how often the output box size changes and the median scale error
     |log sqrt(area_pred / area_gt)| of the output box and of the DINO box.

Cleanest box-quality probe: run with --set adaptive.box_source=dino. Tracking is then
identical to coarse-only and both boxes are logged on every frame. In a box_source=fine
run the DINO box already benefits from LightGlue's size, and the fit is computed at the
fine-tracked size (more favourable to LightGlue).
"""

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np

SIZE_BINS = [0, 12, 20, 32, 64, np.inf]
INL_BINS = [8, 10, 15, 30, 60, np.inf]


def _iou(a, b):
    x1 = np.maximum(a[:, 0], b[:, 0]); y1 = np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2]); y2 = np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / np.maximum(a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter, 1e-12)


def _center(b):
    return b[:, :2] + b[:, 2:] / 2


def _bin_label(edges, k):
    lo, hi = edges[k], edges[k + 1]
    return f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">={lo:g}"


def _load_names(spec):
    if not spec:
        return None
    if os.path.isfile(spec):
        return {l.split('#')[0].strip() for l in open(spec)} - {''}
    return {n.strip() for n in spec.split(',') if n.strip()}


def load(results, anno, names=None):
    """Per-frame table (init frame and NaN-GT frames dropped) over all sequences."""
    files = sorted(glob.glob(os.path.join(results, 'diag', '*.json')))
    if not files:
        raise SystemExit(f"No diagnostics in {os.path.join(results, 'diag')}")
    cols = defaultdict(list)
    for f in files:
        d = json.load(open(f))
        s, fr = d['sequence'], d['frames']
        if names is not None and s not in names:
            continue
        if 'dino_box' not in fr:
            raise SystemExit(f"{f} has no dino_box/fine_box: re-run with the updated tracker")
        g = np.genfromtxt(os.path.join(anno, s + '.txt'), delimiter=',')
        res = os.path.join(results, s + '.txt')
        p = np.loadtxt(res, delimiter=',') if os.path.exists(res) else None
        n = len(fr['branch'])
        m = min(n, len(g), len(p) if p is not None else n)
        v = ~np.isnan(g[:m]).any(1); v[0] = False
        idx = np.where(v)[0]
        cols['seq'] += [s] * len(idx)
        cols['gt'].append(g[idx])
        cols['out'].append(p[idx] if p is not None else np.full((len(idx), 4), np.nan))
        prev = p[idx - 1] if p is not None else np.full((len(idx), 4), np.nan)
        cols['prev_out'].append(prev)
        cols['dino'].append(np.array(fr['dino_box'], float)[idx])
        cols['fine'].append(np.array(fr['fine_box'], float)[idx])
        for k, dflt in (('fine_success', False), ('box_source', 'dino'), ('fine_reject', ''),
                        ('fine_num_inliers', 0), ('fine_inlier_ratio', 0.0), ('fine_num_keypoints', 0)):
            col = fr.get(k, [dflt] * n)
            cols[k] += [col[i] for i in idx]
    if not cols['seq']:
        raise SystemExit("No matching sequences")
    T = {k: (np.concatenate(v) if k in ('gt', 'out', 'prev_out', 'dino', 'fine') else np.array(v))
         for k, v in cols.items()}
    T['size'] = np.sqrt(T['gt'][:, 2] * T['gt'][:, 3])
    T['size_bin'] = np.clip(np.digitize(T['size'], SIZE_BINS) - 1, 0, len(SIZE_BINS) - 2)
    return T


def report_source(T):
    print("1. Box source by GT target size  (fraction of frames)")
    print(f"  {'size px':>8s} {'frames':>7s} {'fine':>6s} {'dino':>6s} {'redet':>6s}   top fallback reasons")
    for k in range(len(SIZE_BINS) - 1):
        sel = T['size_bin'] == k
        if not sel.any():
            continue
        src = T['box_source'][sel]
        fine = np.isin(src, ['fine', 'redetect_fine']).mean()
        red = np.isin(src, ['redetect', 'redetect_fine']).mean()
        dino = (src == 'dino').mean()
        rs = Counter(T['fine_reject'][sel & (T['box_source'] == 'dino')])
        top = ', '.join(f"{r or 'reliable'} {c / max(sel.sum(), 1):.0%}" for r, c in rs.most_common(4))
        print(f"  {_bin_label(SIZE_BINS, k):>8s} {int(sel.sum()):7d} {fine:6.1%} {dino:6.1%} {red:6.1%}   {top}")
    src = T['box_source']
    if not np.isin(src, ['fine', 'redetect_fine']).any() and (T['fine_reject'] == '').any():
        print("  (no fine boxes used - a box_source=dino probe? 'reliable' = the fit passed the check)")
    print(f"  {'all':>8s} {len(src):7d} {np.isin(src, ['fine', 'redetect_fine']).mean():6.1%} "
          f"{(src == 'dino').mean():6.1%} {np.isin(src, ['redetect', 'redetect_fine']).mean():6.1%}")
    print(f"  any LightGlue fit (>= min_inliers): {T['fine_success'].mean():.1%} of frames; "
          f"median keypoints in crop {np.median(T['fine_num_keypoints']):.0f}")


def _cmp_line(label, iou_d, iou_f):
    if len(iou_d) == 0:
        return
    print(f"  {label:22s} n={len(iou_d):6d}  IoU dino {iou_d.mean():.3f}  fine {iou_f.mean():.3f}  "
          f"(diff {np.mean(iou_f - iou_d):+.3f}, median {np.median(iou_f - iou_d):+.3f})  "
          f"fine better {np.mean(iou_f > iou_d):.0%}  fine IoU<0.3 {np.mean(iou_f < 0.3):.0%}")


def report_quality(T, near):
    gt, dino, fine = T['gt'], T['dino'], T['fine']
    dc = np.linalg.norm(_center(dino) - _center(gt), axis=1) / np.maximum(T['size'], 1e-6)
    close = dc <= near
    has_fit = T['fine_success'].astype(bool) & (fine[:, 2] > 0)
    S = close & has_fit
    print(f"\n2. Box quality on frames with the DINO center within {near} target sizes of GT")
    print(f"  such frames: {close.sum()} ({close.mean():.0%} of all); with a LightGlue fit: {S.sum()} "
          f"({S.sum() / max(close.sum(), 1):.0%} of them)")
    if not S.any():
        return
    iou_d, iou_f = _iou(dino[S], gt[S]), _iou(fine[S], gt[S])
    rel = T['fine_reject'][S] == ''
    _cmp_line('all fits', iou_d, iou_f)
    _cmp_line('reliable fits', iou_d[rel], iou_f[rel])
    _cmp_line('rejected fits', iou_d[~rel], iou_f[~rel])
    for r, _ in Counter(T['fine_reject'][S][~rel]).most_common():
        q = T['fine_reject'][S] == r
        _cmp_line(f'  rejected: {r}', iou_d[q], iou_f[q])
    print("  by GT size (reliable fits):")
    sb = T['size_bin'][S]
    for k in range(len(SIZE_BINS) - 1):
        q = rel & (sb == k)
        _cmp_line(f'  {_bin_label(SIZE_BINS, k)} px', iou_d[q], iou_f[q])
    print("  by RANSAC inliers (all fits):")
    ib = np.clip(np.digitize(T['fine_num_inliers'][S], INL_BINS) - 1, 0, len(INL_BINS) - 2)
    for k in range(len(INL_BINS) - 1):
        q = ib == k
        _cmp_line(f'  {_bin_label(INL_BINS, k)} inl', iou_d[q], iou_f[q])
    ce_d = np.linalg.norm(_center(dino[S]) - _center(gt[S]), axis=1)
    ce_f = np.linalg.norm(_center(fine[S]) - _center(gt[S]), axis=1)
    print(f"  center error px (median): dino {np.median(ce_d):.1f}  fine {np.median(ce_f):.1f}  "
          f"(reliable fits: dino {np.median(ce_d[rel]) if rel.any() else np.nan:.1f}  "
          f"fine {np.median(ce_f[rel]) if rel.any() else np.nan:.1f})")


def report_size(T):
    out, prev, gt, dino = T['out'], T['prev_out'], T['gt'], T['dino']
    if np.isnan(out).all():
        return
    a_out, a_prev = out[:, 2] * out[:, 3], prev[:, 2] * prev[:, 3]
    changed = np.abs(np.log(np.maximum(a_out, 1e-6) / np.maximum(a_prev, 1e-6))) / 2 > 0.01
    ok = _iou(out, gt) > 0
    def serr(b):
        return np.abs(np.log(np.maximum(b[:, 2] * b[:, 3], 1e-6) / (gt[:, 2] * gt[:, 3]))) / 2
    print("\n3. Size adaptation")
    print(f"  output size changed >1% vs previous frame: {changed.mean():.1%} of frames")
    print(f"  median scale error |log sqrt(area/gt)| on frames with IoU>0: output {np.median(serr(out)[ok]):.3f}"
          f"  dino box {np.median(serr(dino)[ok]):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True, help='tracker results folder containing diag/ and <seq>.txt')
    ap.add_argument('--anno', required=True, help='UAV123 anno folder (anno/UAV123)')
    ap.add_argument('--sequences', default=None, help='comma-separated names or a .txt list')
    ap.add_argument('--near', type=float, default=0.3, help='DINO-center distance (target sizes) for the box test')
    args = ap.parse_args()
    T = load(args.results, args.anno, _load_names(args.sequences))
    print(f"{len(set(T['seq']))} sequences, {len(T['seq'])} frames with GT\n")
    report_source(T)
    report_quality(T, args.near)
    report_size(T)


if __name__ == '__main__':
    main()
