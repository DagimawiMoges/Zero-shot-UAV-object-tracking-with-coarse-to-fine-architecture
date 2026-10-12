# Copyright 2026 Dagimawi Moges Wakumo
# SPDX-License-Identifier: Apache-2.0
"""
Evaluation harness for the Blihat tracker.

Integrates with the got10k toolkit for standardized benchmark evaluation
on UAV123, UAV20L, DTB70, and VisDrone-SOT.
"""

import os
import json
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Optional

from blihat.config import Config
from blihat.tracker import CascadeTracker
from blihat.adaptive_tracker import AdaptiveTracker

TRACKERS = {'cascade': CascadeTracker, 'adaptive': AdaptiveTracker}


def build_tracker(config: Config, tracker_type: str = 'adaptive'):
    """Instantiate a tracker by name ('adaptive' or 'cascade')."""
    if tracker_type not in TRACKERS:
        raise ValueError(f"Unknown tracker '{tracker_type}'. Choose from {list(TRACKERS)}")
    return TRACKERS[tracker_type](config)

logger = logging.getLogger(__name__)


from got10k.trackers import Tracker as Got10kBaseTracker


class Got10kTrackerWrapper(Got10kBaseTracker):
    """Wraps CascadeTracker for the got10k evaluation toolkit interface.
    
    The got10k toolkit expects a tracker with:
    - init(image, box): initialize with first frame and bbox (x, y, w, h)
    - update(image): return predicted bbox for current frame
    - name: string identifier
    """
    
    def __init__(self, config: Optional[Config] = None, name: str = 'Blihat',
                 tracker_type: str = 'adaptive'):
        super().__init__(name=name, is_deterministic=True)
        self.config = config or Config.default()
        # Instantiate the tracker once to avoid loading models repeatedly
        self.tracker = build_tracker(self.config, tracker_type)
        # Optional per-frame diagnostics dump: set by run_got10k_benchmark
        self.diag_dir: Optional[str] = None
        self.img2seq: dict = {}

    def track(self, img_files, box, visualize=False):
        boxes, times = super().track(img_files, box, visualize=visualize)
        if self.diag_dir:
            try:
                self._dump_diagnostics(img_files)
            except Exception as e:  # never lose a benchmark run over logging
                logger.warning(f"Could not save diagnostics: {e}")
        return boxes, times

    def _dump_diagnostics(self, img_files):
        """Save the tracker's per-frame diagnostics as column-oriented JSON."""
        from dataclasses import asdict
        diags = getattr(self.tracker, 'diagnostics', None)
        if not diags:
            return
        first = img_files[0]
        seq = self.img2seq.get(first) or '%s_%s' % (
            os.path.basename(os.path.dirname(first)), os.path.splitext(os.path.basename(first))[0])
        rows = [asdict(d) for d in diags]
        cols = {k: [r[k] for r in rows] for k in rows[0] if k != 'bbox'}
        out = {'sequence': seq, 'summary': self.tracker.get_diagnostics_summary(), 'frames': cols}
        os.makedirs(self.diag_dir, exist_ok=True)
        with open(os.path.join(self.diag_dir, seq + '.json'), 'w') as f:
            json.dump(out, f, default=float)
    
    @staticmethod
    def _load_bgr_image(image):
        """Convert image input to BGR numpy array.
        
        got10k's Tracker.track() opens images with PIL (Image.open),
        so we may receive PIL Images, file paths, or numpy arrays.
        """
        import cv2
        if isinstance(image, str):
            return cv2.imread(image)
        elif not isinstance(image, np.ndarray):
            # PIL Image — convert to numpy RGB, then to BGR
            image = np.array(image)
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def init(self, image, box):
        """Initialize tracker on first frame.
        
        Args:
            image: PIL Image, file path (str), or RGB numpy array (H, W, 3)
            box: ndarray (x, y, w, h)
        """
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        bgr_image = self._load_bgr_image(image)
        
        bbox = tuple(box.tolist()) if hasattr(box, 'tolist') else tuple(box)
        self.tracker.init(bgr_image, bbox)
    
    def update(self, image):
        """Track target in new frame.
        
        Args:
            image: PIL Image, file path (str), or RGB numpy array (H, W, 3)
            
        Returns:
            ndarray (x, y, w, h) predicted bounding box
        """
        bgr_image = self._load_bgr_image(image)
        
        result = self.tracker.track(bgr_image)
        bbox = result['bbox']
        
        return np.array(bbox, dtype=np.float64)


def _load_sequence_list(spec: Optional[str]):
    """Comma-separated names, or a path to a text file with one name per line."""
    if not spec:
        return None
    if os.path.isfile(spec):
        with open(spec) as f:
            names = [l.split('#')[0].strip() for l in f]
    else:
        names = [n.strip() for n in spec.split(',')]
    return [n for n in names if n]


def _filter_dataset(dataset, names):
    """Restrict a got10k dataset to the given sequence names (in place)."""
    wanted = set(names)
    missing = wanted - set(dataset.seq_names)
    if missing:
        logger.warning(f"Unknown sequences ignored: {sorted(missing)}")
    n_all = len(dataset.seq_names)
    keep = [i for i, n in enumerate(dataset.seq_names) if n in wanted]
    # got10k datasets index parallel per-sequence lists by position; filter all of them
    for attr in ('seq_names', 'anno_files', 'seq_dirs'):
        vals = getattr(dataset, attr, None)
        if isinstance(vals, list) and len(vals) == n_all:
            setattr(dataset, attr, [vals[i] for i in keep])
    logger.info(f"Evaluating {len(keep)} sequences")


def _cap_frames(dataset, max_frames: int):
    """Make a got10k dataset return only the first max_frames frames of each sequence
    (images and annotations), for cheap test runs. Affects both run() and report()."""
    base = type(dataset)

    def getitem(self, index):
        img_files, anno = base.__getitem__(self, index)
        return img_files[:max_frames], anno[:max_frames]

    dataset.__class__ = type(f'{base.__name__}First{max_frames}', (base,), {'__getitem__': getitem})
    logger.info(f"Only the first {max_frames} frames of each sequence will be tracked")


def run_got10k_benchmark(config: Config, dataset_name: str, dataset_root: str,
                         result_dir: str, subset: str = 'test', tracker_type: str = 'adaptive',
                         report_dir: str = 'reports', sequences=None, name: Optional[str] = None,
                         save_diagnostics: bool = True, max_frames: Optional[int] = None):
    """Run evaluation using got10k toolkit.
    
    Args:
        config: tracker configuration
        dataset_name: one of 'UAV123', 'UAV20L', 'DTB70'
        dataset_root: path to dataset directory
        result_dir: where to save results
        subset: dataset subset
    """
    try:
        from got10k.experiments import ExperimentUAV123, ExperimentDTB70
    except ImportError:
        logger.error("got10k toolkit not installed. Run: pip install got10k")
        return
    
    if name is None:
        name = 'Blihat' if tracker_type == 'adaptive' else f'Blihat-{tracker_type}'
    tracker = Got10kTrackerWrapper(config, name=name, tracker_type=tracker_type)
    
    experiment_map = {
        'UAV123': lambda: ExperimentUAV123(dataset_root, result_dir=result_dir, report_dir=report_dir),
        'UAV20L': lambda: ExperimentUAV123(dataset_root, version='UAV20L', result_dir=result_dir, report_dir=report_dir),
        'DTB70': lambda: ExperimentDTB70(dataset_root, result_dir=result_dir, report_dir=report_dir),
    }
    
    if dataset_name not in experiment_map:
        logger.error(f"Unsupported dataset: {dataset_name}. Supported: {list(experiment_map.keys())}")
        logger.info("For UAV20L and VisDrone-SOT, use the custom evaluation functions.")
        return
    
    experiment = experiment_map[dataset_name]()
    if sequences:
        _filter_dataset(experiment.dataset, sequences)
    if max_frames:
        _cap_frames(experiment.dataset, max_frames)
    
    # Run tracker on all sequences
    if save_diagnostics:
        tracker.diag_dir = os.path.join(experiment.result_dir, name, 'diag')
        for seq_name in experiment.dataset.seq_names:
            img_files, _ = experiment.dataset[seq_name]
            tracker.img2seq[img_files[0]] = seq_name

    experiment.run(tracker)
    
    # Generate report
    experiment.report([tracker.name])
    
    logger.info(f"Results saved to {result_dir}")


def run_custom_benchmark(config: Config, sequences_dir: str, result_dir: str,
                         dataset_name: str = 'custom', tracker_type: str = 'adaptive'):
    """Run evaluation on a custom dataset with standard folder structure.
    
    Expected structure:
        sequences_dir/
            sequence_name/
                img/
                    0001.jpg
                    0002.jpg
                    ...
                groundtruth_rect.txt  (one bbox per line: x,y,w,h)
    
    Args:
        config: tracker configuration
        sequences_dir: path to sequences directory
        result_dir: where to save results
        dataset_name: name for reporting
    """
    import cv2
    
    sequences_path = Path(sequences_dir)
    results_path = Path(result_dir) / dataset_name
    results_path.mkdir(parents=True, exist_ok=True)
    
    all_ious = []
    all_center_errors = []
    sequence_results = {}
    
    sequence_dirs = sorted([d for d in sequences_path.iterdir() if d.is_dir()])
    
    # Instantiate tracker once
    tracker = build_tracker(config, tracker_type)
    import torch
    
    for seq_dir in sequence_dirs:
        seq_name = seq_dir.name
        logger.info(f"Processing sequence: {seq_name}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Load ground truth
        gt_file = seq_dir / 'groundtruth_rect.txt'
        if not gt_file.exists():
            gt_file = seq_dir / 'groundtruth.txt'
        if not gt_file.exists():
            logger.warning(f"No ground truth found for {seq_name}, skipping")
            continue
        
        gt_bboxes = np.loadtxt(str(gt_file), delimiter=',')
        if gt_bboxes.ndim == 1:
            gt_bboxes = gt_bboxes.reshape(1, -1)
        
        # Load images
        img_dir = seq_dir / 'img'
        if not img_dir.exists():
            img_dir = seq_dir  # Images might be directly in the sequence folder
        
        img_files = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))
        
        if len(img_files) == 0:
            logger.warning(f"No images found for {seq_name}, skipping")
            continue
        
        predictions = []
        
        for i, img_file in enumerate(img_files):
            frame = cv2.imread(str(img_file))
            if frame is None:
                logger.warning(f"Could not read {img_file}")
                continue
            
            if i == 0:
                bbox = tuple(gt_bboxes[0].tolist())
                tracker.init(frame, bbox)
                predictions.append(gt_bboxes[0])
            else:
                result = tracker.track(frame)
                predictions.append(np.array(result['bbox']))
        
        predictions = np.array(predictions)
        
        # Compute metrics
        n_frames = min(len(predictions), len(gt_bboxes))
        pred = predictions[:n_frames]
        gt = gt_bboxes[:n_frames]
        
        ious = compute_iou_array(pred, gt)
        center_errors = compute_center_error(pred, gt)
        
        all_ious.extend(ious.tolist())
        all_center_errors.extend(center_errors.tolist())
        
        sequence_results[seq_name] = {
            'mean_iou': float(np.mean(ious)),
            'success_auc': float(compute_success_auc(ious)),
            'precision_20': float(np.mean(center_errors < 20)),
            'mean_center_error': float(np.mean(center_errors)),
            'num_frames': n_frames,
            'fps': 1000.0 / np.mean([d.elapsed_ms for d in tracker.diagnostics[1:]]) if len(tracker.diagnostics) > 1 else 0
        }
        
        # Save per-sequence predictions
        np.savetxt(str(results_path / f'{seq_name}.txt'), predictions, delimiter=',', fmt='%.4f')
        
        # Save diagnostics
        diag_summary = tracker.get_diagnostics_summary()
        with open(str(results_path / f'{seq_name}_diagnostics.json'), 'w') as f:
            json.dump(diag_summary, f, indent=2, default=str)
    
    # Overall metrics
    all_ious = np.array(all_ious)
    all_center_errors = np.array(all_center_errors)
    
    overall_results = {
        'dataset': dataset_name,
        'num_sequences': len(sequence_results),
        'total_frames': len(all_ious),
        'success_auc': float(compute_success_auc(all_ious)),
        'precision_20': float(np.mean(all_center_errors < 20)),
        'mean_iou': float(np.mean(all_ious)),
        'mean_center_error': float(np.mean(all_center_errors)),
        'per_sequence': sequence_results
    }
    
    with open(str(results_path / 'results.json'), 'w') as f:
        json.dump(overall_results, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Results for {dataset_name}:")
    logger.info(f"  Success AUC:   {overall_results['success_auc']:.4f}")
    logger.info(f"  Precision@20:  {overall_results['precision_20']:.4f}")
    logger.info(f"  Mean IoU:      {overall_results['mean_iou']:.4f}")
    logger.info(f"  Mean CE:       {overall_results['mean_center_error']:.2f}")
    logger.info(f"{'='*60}")
    
    return overall_results


def compute_iou_array(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Compute IoU between predicted and ground truth bounding boxes.
    
    Args:
        pred: (N, 4) predicted bboxes in xywh format
        gt: (N, 4) ground truth bboxes in xywh format
        
    Returns:
        (N,) IoU values
    """
    # Convert xywh to xyxy
    pred_x1 = pred[:, 0]
    pred_y1 = pred[:, 1]
    pred_x2 = pred[:, 0] + pred[:, 2]
    pred_y2 = pred[:, 1] + pred[:, 3]
    
    gt_x1 = gt[:, 0]
    gt_y1 = gt[:, 1]
    gt_x2 = gt[:, 0] + gt[:, 2]
    gt_y2 = gt[:, 1] + gt[:, 3]
    
    # Intersection
    inter_x1 = np.maximum(pred_x1, gt_x1)
    inter_y1 = np.maximum(pred_y1, gt_y1)
    inter_x2 = np.minimum(pred_x2, gt_x2)
    inter_y2 = np.minimum(pred_y2, gt_y2)
    
    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    
    # Union
    pred_area = pred[:, 2] * pred[:, 3]
    gt_area = gt[:, 2] * gt[:, 3]
    union_area = pred_area + gt_area - inter_area
    
    iou = inter_area / np.maximum(union_area, 1e-8)
    return iou


def compute_center_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Compute center error between predicted and ground truth bboxes.
    
    Args:
        pred: (N, 4) xywh
        gt: (N, 4) xywh
        
    Returns:
        (N,) center errors in pixels
    """
    pred_cx = pred[:, 0] + pred[:, 2] / 2
    pred_cy = pred[:, 1] + pred[:, 3] / 2
    gt_cx = gt[:, 0] + gt[:, 2] / 2
    gt_cy = gt[:, 1] + gt[:, 3] / 2
    
    return np.sqrt((pred_cx - gt_cx)**2 + (pred_cy - gt_cy)**2)


def compute_success_auc(ious: np.ndarray, thresholds: Optional[np.ndarray] = None) -> float:
    """Compute success AUC (area under the success plot curve).
    
    Args:
        ious: (N,) IoU values
        thresholds: IoU thresholds for the success plot. Default: 0 to 1 in 101 steps
        
    Returns:
        AUC value in [0, 1]
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)
    
    success_rates = np.array([np.mean(ious >= t) for t in thresholds])
    auc = np.trapz(success_rates, thresholds)
    
    return auc


def apply_overrides(config: Config, overrides):
    """Apply 'section.key=value' overrides (value parsed as YAML) to a Config."""
    import yaml
    for item in overrides:
        key, sep, raw = item.partition('=')
        if not sep or '.' not in key:
            raise ValueError(f"Bad --set '{item}', expected section.key=value")
        section, field_name = key.strip().split('.', 1)
        target = getattr(config, section)
        if not hasattr(target, field_name):
            raise ValueError(f"Unknown config field '{key}'")
        value = yaml.safe_load(raw)
        if isinstance(value, list):
            value = tuple(value)
        setattr(target, field_name, value)
        logger.info(f"Config override: {key} = {value!r}")


def main():
    parser = argparse.ArgumentParser(description='Blihat Tracker Evaluation')
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Dataset name or path to sequences directory')
    parser.add_argument('--dataset-root', type=str, default=None,
                        help='Root directory for got10k-supported datasets')
    parser.add_argument('--result-dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--tracker', type=str, default='adaptive', choices=['adaptive', 'cascade'],
                        help='Tracker variant to evaluate')
    parser.add_argument('--report-dir', type=str, default=None,
                        help='Where got10k writes reports (default: <result-dir>/reports)')
    parser.add_argument('--sequences', type=str, default=None,
                        help='Only run these sequences: comma-separated names or a .txt file (one per line)')
    parser.add_argument('--name', type=str, default=None,
                        help='Tracker name used for the results folder (default: Blihat, or Blihat-cascade)')
    parser.add_argument('--set', action='append', default=[], metavar='SECTION.KEY=VALUE',
                        help='Override a config value, e.g. --set adaptive.scale_penalty=1.0 '
                             '--set "adaptive.scales=[0.9,1.0,1.1]" (repeatable)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Track only the first N frames of each sequence (quick tests). '
                             'Use a separate --result-dir: got10k skips sequences that already have results.')
    parser.add_argument('--no-diagnostics', action='store_true',
                        help='Do not save per-frame tracker diagnostics (<result-dir>/<dataset>/<name>/diag/)')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s: %(message)s'
    )
    
    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config.default()
    apply_overrides(config, args.set)
    
    # Determine evaluation mode
    if args.dataset in ('UAV123', 'UAV20L', 'DTB70') and args.dataset_root:
        run_got10k_benchmark(config, args.dataset, args.dataset_root, args.result_dir,
                             tracker_type=args.tracker,
                             report_dir=args.report_dir or os.path.join(args.result_dir, 'reports'),
                             sequences=_load_sequence_list(args.sequences), name=args.name,
                             save_diagnostics=not args.no_diagnostics,
                             max_frames=args.max_frames)
    elif os.path.isdir(args.dataset):
        dataset_name = Path(args.dataset).name
        run_custom_benchmark(config, args.dataset, args.result_dir, dataset_name,
                             tracker_type=args.tracker)
    else:
        logger.error(f"Cannot find dataset: {args.dataset}")
        logger.info("Provide --dataset-root for got10k datasets, or a path to a sequences directory")


if __name__ == '__main__':
    main()
