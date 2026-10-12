#!/usr/bin/env python3
# Copyright 2026 Dagimawi Moges Wakumo
# SPDX-License-Identifier: Apache-2.0
"""
Quick demo: run the Blihat tracker on a video file or webcam.

Usage:
    python demo.py --video path/to/video.mp4
    python demo.py --video path/to/video.mp4 --bbox 100,200,50,80
    python demo.py  # webcam mode (select ROI on first frame)
"""

import argparse
import cv2
import numpy as np
import logging
import time

from blihat.config import Config
from blihat.tracker import CascadeTracker
from blihat.adaptive_tracker import AdaptiveTracker
from blihat.utils.debug_viz import DebugRecorder

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Blihat Tracker Demo')
    parser.add_argument('--video', type=str, default=None, help='Path to video file (default: webcam)')
    parser.add_argument('--bbox', type=str, default=None, help='Initial bbox as x,y,w,h (default: ROI selector)')
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML')
    parser.add_argument('--save', type=str, default=None, help='Save output video to file (prefix)')
    parser.add_argument('--tracker', type=str, default='adaptive', choices=['adaptive', 'cascade'],
                        help='Tracker variant')
    parser.add_argument('--debug-dir', type=str, default=None,
                        help='Save the queried template and DINO heatmap of every frame here '
                             '(template/, heatmap/ PNGs; plus <save>_debug.mp4 if --save is set)')
    parser.add_argument('--debug-raw', action='store_true',
                        help='Also save raw response maps and kernels as .npz per frame')
    parser.add_argument('--no-display', action='store_true', help='Disable display window')
    parser.add_argument('--save-boxes', type=str, default=None,
                        help='Write per-frame boxes (x,y,w,h, frame 1 = init box) to this .txt in '
                             'UAV123/got10k format; per-frame times go to <name>_time.txt')
    args = parser.parse_args()

    # Load config
    config = Config.from_yaml(args.config) if args.config else Config.default()

    # Open video
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error(f"Cannot open video: {args.video or 'webcam'}")
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        logger.error("Cannot read first frame")
        return

    frame_h, frame_w = frame.shape[:2]

    # Video writers
    writer_final = None
    writer_heatmap = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        final_path = f"{args.save}_final.mp4"
        heatmap_path = f"{args.save}_heatmap.mp4"
        writer_final = cv2.VideoWriter(final_path, fourcc, fps_video, (frame_w, frame_h))
        writer_heatmap = cv2.VideoWriter(heatmap_path, fourcc, fps_video, (frame_w, frame_h))

    # Get initial bbox
    if args.bbox:
        bbox = tuple(map(float, args.bbox.split(',')))
    else:
        logger.info("Select target region with mouse, then press ENTER or SPACE")
        bbox = cv2.selectROI("Blihat Tracker - Select Target", frame, fromCenter=False)
        cv2.destroyWindow("Blihat Tracker - Select Target")
        if bbox[2] == 0 or bbox[3] == 0:
            logger.error("Invalid selection")
            return

    # Initialize tracker
    logger.info(f"Initializing tracker with bbox: {bbox}")
    tracker = AdaptiveTracker(config) if args.tracker == 'adaptive' else CascadeTracker(config)
    tracker.init(frame, bbox)

    recorder = None
    if args.debug_dir:
        tracker.debug = True
        recorder = DebugRecorder(args.debug_dir, save_raw=args.debug_raw,
                                 video_path=f"{args.save}_debug.mp4" if args.save else None,
                                 fps=fps_video)
        logger.info(f"Debug dumps -> {args.debug_dir}")

    frame_idx = 0
    fps_times = []
    saved_boxes = [tuple(float(v) for v in bbox)]   # frame 1 = initial box (benchmark convention)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        t_start = time.perf_counter()

        result = tracker.track(frame)
        if recorder is not None:
            recorder.write(result.get('debug'))

        elapsed = time.perf_counter() - t_start
        fps_times.append(elapsed)
        if frame_idx % 100 == 0:
            logger.info(f"frame {frame_idx}: {1.0 / np.mean(fps_times[-100:]):.1f} FPS")

        bbox = result['bbox']
        saved_boxes.append(tuple(float(v) for v in bbox))
        confidence = result['confidence']
        branch = result['branch']

        # Draw result
        x, y, w, h = [int(v) for v in bbox]

        # Color by branch
        color_map = {
            'fine': (0, 255, 0),      # Green
            'coarse': (0, 165, 255),   # Orange
            'blend': (255, 255, 0),    # Cyan
            'lost': (0, 0, 255),       # Red
            'redetect': (255, 0, 255), # Magenta
        }
        color = color_map.get(branch, (255, 255, 255))

        # Info text
        current_fps = 1.0 / elapsed if elapsed > 0 else 0
        avg_fps = 1.0 / np.mean(fps_times[-30:]) if fps_times else 0
        
        info_lines = [
            f"Frame: {frame_idx}",
            f"Branch: {branch}",
            f"Conf: {confidence:.3f}",
            f"FPS: {current_fps:.1f} (avg: {avg_fps:.1f})",
        ]

        if not args.no_display or args.save:
            display_frame = frame.copy()
            heatmap_frame = np.zeros_like(frame)  # Pure DINOv3 output on black
            
            # Extract heatmap and search_bbox
            heatmap_tensor = result.get('heatmap', None)
            search_bbox = result.get('search_bbox', None)
            
            if heatmap_tensor is not None and search_bbox is not None:
                # Convert (1, 1, H, W) tensor to numpy
                hm_np = heatmap_tensor.squeeze().cpu().numpy()
                # Normalize to 0-255
                hm_min, hm_max = hm_np.min(), hm_np.max()
                if hm_max > hm_min:
                    hm_norm = ((hm_np - hm_min) / (hm_max - hm_min) * 255).astype(np.uint8)
                else:
                    hm_norm = np.zeros_like(hm_np, dtype=np.uint8)
                
                # Apply colormap
                hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
                
                # Resize to search_bbox dimensions
                sx, sy, sw, sh = [int(v) for v in search_bbox]
                # Clamp search_bbox to image bounds for overlay
                ix1, iy1 = max(0, sx), max(0, sy)
                ix2, iy2 = min(frame_w, sx + sw), min(frame_h, sy + sh)
                
                if ix2 > ix1 and iy2 > iy1:
                    hm_resized = cv2.resize(hm_color, (sw, sh))
                    # Crop the resized heatmap to fit exactly what is on-screen
                    hm_cropped = hm_resized[(iy1-sy):(iy2-sy), (ix1-sx):(ix2-sx)]
                    
                    # Overlay heatmap on video for the main display
                    roi = display_frame[iy1:iy2, ix1:ix2]
                    display_frame[iy1:iy2, ix1:ix2] = cv2.addWeighted(roi, 0.5, hm_cropped, 0.5, 0)
                    
                    # Pure DINOv3 correlation output (no blending with video)
                    heatmap_frame[iy1:iy2, ix1:ix2] = hm_cropped
                    
                    # Draw search box boundary
                    cv2.rectangle(display_frame, (ix1, iy1), (ix2, iy2), (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.rectangle(heatmap_frame, (ix1, iy1), (ix2, iy2), (255, 255, 255), 1, cv2.LINE_AA)
            
            # Draw result bbox
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

            for i, line in enumerate(info_lines):
                cv2.putText(display_frame, line, (10, 25 + i * 22), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if writer_final:
            writer_final.write(display_frame)
        if writer_heatmap:
            writer_heatmap.write(heatmap_frame)

        if not args.no_display:
            cv2.imshow('Blihat Tracker', display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break

    # Summary
    summary = tracker.get_diagnostics_summary()
    logger.info("\nTracking Summary:")
    logger.info(f"  Total frames: {summary.get('total_frames', 0)}")
    logger.info(f"  Average FPS: {summary.get('avg_fps', 0):.1f}")
    logger.info(f"  Branch usage: {summary.get('branch_counts', {})}")
    logger.info(f"  Fine ratio: {summary.get('fine_ratio', 0):.2%}")
    logger.info(f"  Lost frames: {summary.get('lost_frames', 0)}")

    if args.save_boxes:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.save_boxes)), exist_ok=True)
        np.savetxt(args.save_boxes, np.array(saved_boxes), delimiter=',', fmt='%.3f')
        time_path = os.path.splitext(args.save_boxes)[0] + '_time.txt'
        np.savetxt(time_path, np.array(fps_times), fmt='%.8f')
        logger.info(f"Saved {len(saved_boxes)} boxes -> {args.save_boxes} (times -> {time_path})")

    if recorder is not None:
        recorder.close()
    cap.release()
    if writer_final:
        writer_final.release()
    if writer_heatmap:
        writer_heatmap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
