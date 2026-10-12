# Copyright 2026 Dagimawi Moges Wakumo
# SPDX-License-Identifier: Apache-2.0
"""
Summarise the per-frame diagnostics that eval/run_benchmark.py saves to <results>/diag/.

    python eval/analyze_diagnostics.py --results RESULTS/UAV123/Blihat-recovery --anno UAV123/anno/UAV123

Answers three questions per run:
  * Scale: how often a non-1.0 search scale wins (is the box size adapting?)
  * Re-detection: how often it runs, how often it is accepted, why it is refused,
    and whether accepted jumps land on the target (needs --anno)
  * Loss: when the box leaves the target, does the tracker notice (low quality)
    or stay confident on a distractor ("silent" loss)? (needs --anno)
"""

import argparse
import glob
import json
import os
from collections import Counter

import numpy as np


def _iou(a, b):
    x1 = np.maximum(a[:, 0], b[:, 0]); y1 = np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2]); y2 = np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / np.maximum(a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter, 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True, help='tracker results folder containing diag/')
    ap.add_argument('--anno', help='UAV123 anno folder, enables IoU-based checks')
    ap.add_argument('--per-seq', action='store_true', help='print one line per sequence')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results, 'diag', '*.json')))
    if not files:
        raise SystemExit(f"No diagnostics in {os.path.join(args.results, 'diag')}")

    tot = Counter()
    scale_counts = Counter()
    rejects = Counter()
    rd_ratio = {'peak': [], 'identity': []}
    lines = []
    for f in files:
        d = json.load(open(f))
        s, fr = d['sequence'], d['frames']
        n = len(fr['branch'])
        low = np.array(['Low quality' in r for r in fr['reason']])
        sc = np.round(np.array(fr['scale'][1:], dtype=float), 4)
        scale_counts.update(sc.tolist())
        tried = np.array(fr.get('redetect_tried', [False] * n), bool)
        acc = np.array(fr.get('redetect_accepted', [False] * n), bool)
        rej = fr.get('redetect_reject', [''] * n)
        for i in np.where(tried & ~acc)[0]:
            rejects[rej[i]] += 1
            if rej[i] in rd_ratio:
                key = ('redetect_peak', 'redetect_peak_thr') if rej[i] == 'peak' else ('redetect_id', 'redetect_id_thr')
                thr = fr[key[1]][i]
                if thr > 0:
                    rd_ratio[rej[i]].append(fr[key[0]][i] / thr)
        tot.update(frames=n - 1, low=int(low[1:].sum()), tried=int(tried.sum()), accepted=int(acc.sum()),
                   scale_changed=int((sc != 1.0).sum()))
        line = dict(seq=s, n=n, low=low[1:].mean(), tried=int(tried.sum()), acc=int(acc.sum()),
                    scale_changed=float((sc != 1.0).mean()) if len(sc) else 0.0)

        if args.anno:
            res = os.path.join(args.results, s + '.txt')
            ann = os.path.join(args.anno, s + '.txt')
            if os.path.exists(res) and os.path.exists(ann):
                p = np.loadtxt(res, delimiter=','); g = np.genfromtxt(ann, delimiter=',')
                m = min(len(p), len(g), n)
                v = ~np.isnan(g[:m]).any(1)
                iou = np.full(m, np.nan); iou[v] = _iou(p[:m][v], g[:m][v])
                good = np.where(acc[:m])[0]
                hits = int(np.nansum(iou[good] > 0.3))
                tot.update(acc_with_gt=int(np.sum(~np.isnan(iou[good]))), acc_hit=hits)
                # loss onset: first frame where IoU stays 0 for 10 valid frames
                z = (iou == 0)
                onset = next((k for k in range(1, m - 10) if z[k:k + 10].all()), None)
                if onset is not None:
                    after = slice(onset, min(m, onset + 30))
                    silent = 1 - low[after].mean()        # share of first 30 lost frames judged healthy
                    lost_frames = z & v
                    tot.update(lost_seqs=1, lost_frames=int(lost_frames.sum()),
                               lost_but_confident=int((lost_frames & ~low[:m]).sum()))
                    line.update(onset=onset, silent=silent)
                    tot['silent_losses'] += int(silent > 0.5)
                tried_lost = tried[:m] & z
                tot.update(tried_while_lost=int(tried_lost.sum()))
                line.update(acc_hit=hits)
        lines.append(line)

    F = max(tot['frames'], 1)
    print(f"Sequences {len(files)}, frames {tot['frames']}")
    print(f"\nScale: non-1.0 scale chosen in {tot['scale_changed'] / F:.1%} of frames")
    common = ', '.join(f'{k}: {v / F:.1%}' for k, v in sorted(scale_counts.items()))
    print(f"  distribution  {common}")
    print(f"\nQuality: low-quality frames {tot['low'] / F:.1%}")
    print(f"\nRe-detection: {tot['tried']} attempts, {tot['accepted']} accepted")
    if rejects:
        print('  refused because: ' + ', '.join(f'{k or "?"} {v}' for k, v in rejects.most_common()))
    for k, vals in rd_ratio.items():
        if vals:
            print(f"  '{k}' refusals: best/threshold median {np.median(vals):.2f} "
                  f"(p90 {np.percentile(vals, 90):.2f})")
    if args.anno:
        if tot['accepted']:
            print(f"  accepted jumps landing on the target (IoU>0.3): {tot['acc_hit']}/{tot['acc_with_gt']}")
        print(f"  attempts made while the box was actually off the target: {tot['tried_while_lost']}")
        if tot['lost_seqs']:
            print(f"\nLoss: {tot['lost_seqs']} sequences lose the target (10+ frames at IoU 0)")
            print(f"  silent losses (tracker still 'healthy' for most of the first 30 lost frames): "
                  f"{tot['silent_losses']}/{tot['lost_seqs']}")
            print(f"  lost frames judged healthy: {tot['lost_but_confident'] / max(tot['lost_frames'], 1):.1%}")

    if args.per_seq:
        print(f"\n{'seq':12s} {'frames':>6s} {'low%':>5s} {'scale%':>6s} {'rd try/acc':>10s} {'hit':>4s} "
              f"{'onset':>6s} {'silent':>6s}")
        for l in lines:
            print(f"{l['seq']:12s} {l['n']:6d} {l['low']:5.0%} {l['scale_changed']:6.0%} "
                  f"{l['tried']:4d}/{l['acc']:<5d} {l.get('acc_hit', ''):>4} "
                  f"{l.get('onset', ''):>6} {('%.0f%%' % (100 * l['silent'])) if 'silent' in l else '':>6}")


if __name__ == '__main__':
    main()
