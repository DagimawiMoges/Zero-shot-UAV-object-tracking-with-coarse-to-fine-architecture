# Copyright 2026 Dagimawi Moges Wakumo
# SPDX-License-Identifier: Apache-2.0
"""
Failure analysis for got10k-format UAV123 results, optionally compared to a baseline run.

    python eval/analyze_uav123.py --results RESULTS/UAV123/Blihat-adaptive \
        --anno UAV123/anno/UAV123 [--baseline OLD/UAV123/Blihat-adaptive] [--sequences eval/uav123_failed.txt]

Metrics follow got10k's ExperimentUAV123: frames with NaN ground truth are skipped,
success = mean over 21 IoU thresholds, precision = center error <= 20 px. The init
frame is also skipped (it is the ground-truth box). With --baseline, only sequences
present in both runs are compared.
"""

import argparse
import glob
import os
from collections import defaultdict

import numpy as np

ATTRS = ['SV', 'ARC', 'LR', 'FM', 'FOC', 'POC', 'OV', 'BC', 'IV', 'VC', 'CM', 'SOB']
IOU_T = np.linspace(0, 1, 21)


def _iou(a, b):
    x1 = np.maximum(a[:, 0], b[:, 0]); y1 = np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2]); y2 = np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / np.maximum(a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter, 1e-12)


def _ce(a, b):
    return np.hypot(a[:, 0] + a[:, 2] / 2 - b[:, 0] - b[:, 2] / 2, a[:, 1] + a[:, 3] / 2 - b[:, 1] - b[:, 3] / 2)


def evaluate(result_dir, anno_dir, names=None):
    rows = {}
    for f in sorted(glob.glob(os.path.join(result_dir, '*.txt'))):
        s = os.path.basename(f)[:-4]
        if names is not None and s not in names:
            continue
        p = np.loadtxt(f, delimiter=',')
        g = np.genfromtxt(os.path.join(anno_dir, s + '.txt'), delimiter=',')
        n = min(len(p), len(g)); p, g = p[:n], g[:n]
        v = ~np.isnan(g).any(1); v[0] = False
        iou, ce = _iou(p[v], g[v]), _ce(p[v], g[v])
        zero = iou == 0
        onset = next((k for k in range(len(zero) - 30) if zero[k:k + 30].all()), None)
        tf = os.path.join(result_dir, 'times', s + '_time.txt')
        fps = 1 / np.mean(np.loadtxt(tf)[1:]) if os.path.exists(tf) else float('nan')
        af = os.path.join(anno_dir, 'att', s + '.txt')
        att = np.loadtxt(af, delimiter=',').astype(int) if os.path.exists(af) else np.zeros(12, int)
        rows[s] = dict(auc=float(np.mean([(iou > t).mean() for t in IOU_T])), prec=float((ce <= 20).mean()),
                       lost=float(zero.mean()), onset=None if onset is None else onset / len(zero),
                       fps=fps, att=att, cls=''.join(ch for ch in s.split('_')[0] if ch.isalpha()))
    return rows


def summary(rows, label):
    a = np.array([r['auc'] for r in rows.values()]); p = np.array([r['prec'] for r in rows.values()])
    perm = sum(r['onset'] is not None for r in rows.values())
    print(f"{label:10s} seqs {len(rows):3d}  AUC {a.mean():.3f}  P@20 {p.mean():.3f}  "
          f"AUC<0.1 {int((a < 0.1).sum()):3d}  never-recovered {perm:3d}  FPS {np.nanmedian([r['fps'] for r in rows.values()]):.1f}")


def by_group(rows, base=None):
    def line(key, sel):
        a = np.mean([rows[s]['auc'] for s in sel])
        d = f"  ({a - np.mean([base[s]['auc'] for s in sel]):+.3f})" if base else ''
        print(f"  {key:10s} n={len(sel):3d}  AUC {a:.3f}{d}")
    print('\nBy class:')
    groups = defaultdict(list)
    for s, r in rows.items():
        groups[r['cls']].append(s)
    for k, sel in sorted(groups.items(), key=lambda kv: -np.mean([rows[s]['auc'] for s in kv[1]])):
        line(k, sel)
    print('\nBy attribute:')
    for j, an in enumerate(ATTRS):
        sel = [s for s, r in rows.items() if r['att'][j]]
        if sel:
            line(an, sel)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True, help='folder with <seq>.txt (and times/)')
    ap.add_argument('--anno', required=True, help='UAV123 anno folder (with att/)')
    ap.add_argument('--baseline', help='second results folder to compare against')
    ap.add_argument('--sequences', help='txt file (one name per line, # comments) to restrict the evaluation')
    ap.add_argument('--top', type=int, default=15, help='how many per-sequence changes to list')
    args = ap.parse_args()

    names = None
    if args.sequences:
        with open(args.sequences) as f:
            names = {l.split('#')[0].strip() for l in f} - {''}

    rows = evaluate(args.results, args.anno, names)
    base = None
    if args.baseline:
        base = evaluate(args.baseline, args.anno, names)
        common = sorted(set(rows) & set(base))
        rows = {s: rows[s] for s in common}
        base = {s: base[s] for s in common}
        summary(base, 'baseline')
    summary(rows, 'results')
    by_group(rows, base)

    if base:
        d = sorted(rows, key=lambda s: rows[s]['auc'] - base[s]['auc'])
        print('\nLargest drops:')
        for s in d[:args.top]:
            print(f"  {s:12s} {base[s]['auc']:.3f} -> {rows[s]['auc']:.3f}")
        print('\nLargest gains:')
        for s in d[::-1][:args.top]:
            print(f"  {s:12s} {base[s]['auc']:.3f} -> {rows[s]['auc']:.3f}")
    else:
        print('\nWorst sequences:')
        for s in sorted(rows, key=lambda s: rows[s]['auc'])[:args.top]:
            r = rows[s]
            print(f"  {s:12s} AUC {r['auc']:.3f}  P {r['prec']:.3f}  lost {r['lost']:.2f}")


if __name__ == '__main__':
    main()
