# Copyright 2026 Dagimawi Moges Wakumo
# SPDX-License-Identifier: Apache-2.0
"""Run AdaptiveTracker from a given repo on a synthetic sequence; save boxes + diagnostics.
usage: python tests/run_seq.py REPO CONFIG OUT.npz [--set k=v ...] [--seq-seed S] [--n N] [--scale-end X] [--size W H]"""
import argparse, json, os, sys, warnings
warnings.filterwarnings('ignore')
ap = argparse.ArgumentParser()
ap.add_argument('repo'); ap.add_argument('config'); ap.add_argument('out')
ap.add_argument('--set', action='append', default=[])
ap.add_argument('--seq-seed', type=int, default=1); ap.add_argument('--n', type=int, default=80)
ap.add_argument('--scale-end', type=float, default=1.6)
ap.add_argument('--occl', type=int, nargs=2, default=None)
ap.add_argument('--size', type=float, nargs=2, default=(60, 40))
a = ap.parse_args()
sys.path.insert(0, a.repo); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, yaml, torch
from dataclasses import asdict
torch.manual_seed(0); torch.set_num_threads(int(os.environ.get("NT", "2")))
from blihat.config import Config
from blihat.adaptive_tracker import AdaptiveTracker
from synth import FakeBackbone, make_sequence
cfg = Config.from_yaml(os.path.join(a.repo, 'configs', a.config))
cfg.device = 'cpu'; cfg.adaptive.fg_mask = False
for kv in a.set:
    k, v = kv.split('=', 1); v = yaml.safe_load(v)
    setattr(cfg.adaptive, k.split('.')[-1], tuple(v) if isinstance(v, list) else v)
frames, gt = make_sequence(n=a.n, seed=a.seq_seed, scale_end=a.scale_end, size0=tuple(a.size), occl=a.occl)
tr = AdaptiveTracker(cfg, backbone=FakeBackbone())
tr.init(frames[0], tuple(gt[0]))
boxes = [gt[0]]
for f in frames[1:]:
    boxes.append(tr.track(f)['bbox'])
rows = [asdict(d) for d in tr.diagnostics]
np.savez(a.out, boxes=np.array(boxes, dtype=np.float64), gt=gt,
         diag=json.dumps({k: [r[k] for r in rows] for k in rows[0]}, default=float),
         summary=json.dumps(tr.get_diagnostics_summary(), default=str))
print('done', a.out, len(boxes))
