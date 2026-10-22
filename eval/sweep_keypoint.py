# Copyright 2026 Dagimawi Moges Wakumo
# SPDX-License-Identifier: Apache-2.0
"""
Tune the SuperPoint + LightGlue-only baseline (--tracker keypoint) with the same
protocol that selected Blihat's frozen config, so the fine-only row is not judged
on settings tuned for a tracker with DINO in the loop.

Protocol (same as the Blihat ablation in RESULTS.txt section 3):
  UAV123 scale subset (eval/uav123_scale.txt, 24 sequences), first 300 frames.
  1. "frozen": configs/blihat_final.yaml unchanged.
  2. One change at a time from the frozen config (SINGLE below).
  3. "combined": every single change that beats frozen on AUC, applied together.
     When two values of the same key both help, the better one is kept.
  The best of these is the tuned baseline; run it on the full benchmarks with the
  commands printed at the end.

    python eval/sweep_keypoint.py --dataset-root /path/to/UAV123 \
        [--result-dir runs/kp_sweep] [--blihat runs/scale/UAV123/blihat_final] [--dry-run]

Runs whose results already exist are skipped (got10k does the same), so an interrupted
sweep can simply be restarted. --blihat adds a Blihat run on the same subset as a
reference row (not used for selection).
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from analyze_uav123 import evaluate  # noqa: E402

CONFIG = os.path.join(ROOT, 'configs', 'blihat_final.yaml')
SUBSET = os.path.join(HERE, 'uav123_scale.txt')

# One change at a time from the frozen config. Each targets a setting that was chosen
# with DINO centering the match crop, and may suit a tracker without DINO differently.
SINGLE = [
    ('crop3',      ['adaptive.fine_crop_scale=3.0']),      # larger match window: no DINO to center it
    ('crop4',      ['adaptive.fine_crop_scale=4.0']),
    ('offset1',    ['adaptive.fine_max_offset=1.0']),      # fit may land further from the motion prediction
    ('ratio02',    ['adaptive.fine_min_inlier_ratio=0.2']),  # looser gate: fewer holds
    ('inliers15',  ['adaptive.fine_min_inliers=15']),      # stricter gate: fewer wrong fits / bad keyframes
    ('nolock',     ['adaptive.fine_scale_lock=false']),
    ('patience3',  ['adaptive.widen_patience=3']),         # widen later
    ('nocoast',    ['adaptive.motion_coast_frames=0']),    # hold in place instead of extrapolating
    ('kf20',       ['adaptive.keyframe_interval=20']),     # fewer keyframes: less drift without a DINO check
]


def _key(override):
    return override.split('=', 1)[0]


def run_variant(args, name, overrides):
    out = os.path.join(args.result_dir, 'UAV123', name)
    cmd = [sys.executable, os.path.join(HERE, 'run_benchmark.py'), '--tracker', 'keypoint',
           '--config', CONFIG, '--dataset', 'UAV123', '--dataset-root', args.dataset_root,
           '--result-dir', args.result_dir, '--name', name,
           '--sequences', SUBSET, '--max-frames', str(args.max_frames)]
    for o in overrides:
        cmd += ['--set', o]
    print(f"\n=== {name}: {' '.join(overrides) or '(frozen config)'}", flush=True)
    if args.dry_run:
        print(' '.join(cmd))
        return out
    subprocess.run(cmd, check=True, cwd=ROOT)
    return out


def score(path, anno, names):
    rows = evaluate(path, anno, names)
    if not rows:
        return None
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset-root', required=True, help='UAV123 root (as for run_benchmark.py)')
    ap.add_argument('--anno', default=None, help='UAV123 anno folder (default: <root>/anno/UAV123)')
    ap.add_argument('--result-dir', default='runs/kp_sweep',
                    help='separate folder: results here are truncated to --max-frames')
    ap.add_argument('--max-frames', type=int, default=300)
    ap.add_argument('--blihat', default=None,
                    help='optional Blihat results on the same subset/frames, shown as a reference row')
    ap.add_argument('--dry-run', action='store_true', help='print the commands only')
    args = ap.parse_args()
    args.result_dir = os.path.abspath(args.result_dir)   # runs execute from the repo root
    anno = args.anno or os.path.join(args.dataset_root, 'anno', 'UAV123')
    with open(SUBSET) as f:
        names = {l.split('#')[0].strip() for l in f} - {''}

    variants = [('kp_frozen', [])] + [(f'kp_{t}', o) for t, o in SINGLE]
    paths = {n: run_variant(args, n, o) for n, o in variants}
    if args.dry_run:
        return

    res = {n: score(p, anno, names) for n, p in paths.items()}
    base = res['kp_frozen']
    base_auc = np.mean([r['auc'] for r in base.values()])

    # Combine every helpful single change; per key keep the best value.
    best_per_key = {}
    for n, o in variants[1:]:
        auc = np.mean([r['auc'] for r in res[n].values()])
        if auc > base_auc:
            for ov in o:
                k = _key(ov)
                if k not in best_per_key or auc > best_per_key[k][0]:
                    best_per_key[k] = (auc, ov)
    combined = [ov for _, ov in best_per_key.values()]
    overrides = dict(variants)
    if len(combined) >= 2:
        variants.append(('kp_combined', combined))
        overrides['kp_combined'] = combined
        paths['kp_combined'] = run_variant(args, 'kp_combined', combined)
        res['kp_combined'] = score(paths['kp_combined'], anno, names)

    table = []
    for n, rows in res.items():
        common = sorted(set(rows) & set(base))
        a = np.array([rows[s]['auc'] for s in common]); b = np.array([base[s]['auc'] for s in common])
        table.append(dict(name=n, overrides=overrides[n], seqs=len(common), auc=float(a.mean()),
                          prec=float(np.mean([rows[s]['prec'] for s in common])),
                          beats=int((a > b + 0.005).sum()), loses=int((a < b - 0.005).sum()),
                          fps=float(np.nanmedian([rows[s]['fps'] for s in common]))))
    best = max(table, key=lambda r: r['auc'])

    print(f"\nKeypoint-only sweep: UAV123 scale subset, first {args.max_frames} frames")
    print(f"  {'variant':14s} {'AUC':>6s} {'Prec':>6s} {'vs frozen':>11s} {'FPS':>6s}  overrides")
    for r in table:
        mark = '  <- best' if r is best else ''
        vs = '' if r['name'] == 'kp_frozen' else f"{r['beats']}+ / {r['loses']}-"
        print(f"  {r['name']:14s} {r['auc']:6.3f} {r['prec']:6.3f} {vs:>11s} {r['fps']:6.1f}  "
              f"{' '.join(r['overrides']) or '-'}{mark}")
    if args.blihat:
        bl = score(args.blihat, anno, names)
        if bl:
            print(f"  {'Blihat (ref)':14s} {np.mean([r['auc'] for r in bl.values()]):6.3f} "
                  f"{np.mean([r['prec'] for r in bl.values()]):6.3f}")
    print("  vs frozen = sequences better / worse than kp_frozen by more than 0.005 AUC")

    with open(os.path.join(args.result_dir, 'keypoint_sweep.json'), 'w') as f:
        json.dump({'max_frames': args.max_frames, 'variants': table, 'best': best['name']}, f, indent=1)

    sets = ' '.join(f'--set {o}' for o in best['overrides'])
    print(f"\nTuned baseline = {best['name']}. Full benchmarks (repeat for DTB70 / UAV20L):")
    print(f"  python eval/run_benchmark.py --tracker keypoint --config configs/blihat_final.yaml {sets} \\\n"
          f"      --dataset UAV123 --dataset-root {args.dataset_root} --result-dir runs/full --name keypoint_tuned")


if __name__ == '__main__':
    main()
