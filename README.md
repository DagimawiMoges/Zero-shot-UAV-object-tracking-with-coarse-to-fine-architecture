# Blihat: Training-Free Coarse-to-Fine UAV Object Tracking

Blihat is a zero-shot single-object tracker for drone (UAV) footage. It needs no training and no fine-tuning:

- **Coarse stage — where is the target?** Frozen **DINOv3** features (ViT-L/16) are correlated with an adaptive template memory over a multi-scale search region.
- **Fine stage — what is the exact box?** **SuperPoint + LightGlue** match the current crop against the first frame and a few keyframes, and one similarity transform maps the exact frame-1 box into the current frame.
- **Confidence-gated fallback.** The fine box is used only when the geometric fit passes a reliability check; otherwise the coarse DINO box is used.

## Results

Success AUC / precision (20 px) with the frozen configuration `configs/blihat_final.yaml`. All settings were selected on UAV123 subsets; DTB70 and UAV20L are held-out evaluations.

| Benchmark | Coarse only (DINOv3) | Blihat (coarse + fine) | AUC gain | FPS (coarse → full) |
|-----------|----------------------|------------------------|----------|---------------------|
| UAV123 (123 seq.) | 0.400 / 0.614 | **0.490 / 0.690** | +0.090 | 6.9 → 4.6 |
| DTB70 (70 seq.)   | 0.383 / 0.628 | **0.420 / 0.664** | +0.037 | 6.1 → 4.4 |
| UAV20L (20 seq.)  | 0.287 / 0.444 | **0.338 / 0.472** | +0.051 | 6.7 → 4.5 |

FPS measured on an NVIDIA A100 (Google Colab). The UAV20L gain is not statistically significant on 20 sequences. Ablations, per-class/per-attribute breakdowns, failure analysis and exact reproduction commands are in [`RESULTS.txt`](RESULTS.txt).

## Method

```
Frame → search crop → DINOv3 features → correlation with template memory → coarse center
                                                                               │
        crop around center → SuperPoint + LightGlue vs. anchor/keyframes → RANSAC similarity → fine box
                                                                               │
                                                  reliability check ── pass → fine box
                                                                     └─ fail → DINO box
```

Two trackers are included:

- **`AdaptiveTracker`** (`blihat/adaptive_tracker.py`): the Blihat tracker used for all reported results (`--tracker adaptive`, the default)
  - *DINO feature memory* (`blihat/coarse/memory.py`): anchor (frame 1) + long-term EMA + short-term ring buffer, combined into one weighted kernel. Updates are gated by *relative* quality (peak vs. its running median, similarity to the anchor vs. its running median), so no absolute thresholds on DINO scores are needed.
  - *Keyframe geometry* (`blihat/fine/keyframes.py`): every keyframe stores its transform back to frame 1, so the box is always fitted to the exact frame-1 box and box errors do not accumulate. Crops are scale-normalised so small targets still get keypoints.
  - *Never lost*: after sustained low-quality frames the search area widens and tiled re-detection is used instead of declaring the target lost.
  - *Box source* (`adaptive.box_source`):
    - `fine` (final config): DINO tracks, LightGlue gives the box when the fit is reliable (inlier count, inlier ratio, per-frame size step, offset from the DINO center).
    - `agree`: the fine box is used, and memories are updated, only when the fine and DINO estimates agree.
    - `dino`: the fine stage runs and is logged, but the output equals a coarse-only run (a box-quality probe for `eval/analyze_fine_box.py`).
- **`CascadeTracker`** (`blihat/tracker.py`): the original, simpler cascade with a single updating template (`--tracker cascade`, `configs/default.yaml`).

## Installation

Requires Python ≥ 3.10 and a CUDA GPU for reasonable speed.

```bash
git clone https://github.com/DagimawiMoges/Zero-shot-UAV-object-tracking-with-coarse-to-fine-architecture.git Blihat
cd Blihat
pip install -r requirements.txt
pip install git+https://github.com/cvg/LightGlue.git   # fine stage
pip install -e .
```

The DINOv3 weights are downloaded from Hugging Face on first use. They are gated: accept the license on the [model page](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) and log in with `huggingface-cli login` first.

## Usage

### Demo

```bash
# Select the target with the mouse
python demo.py --tracker adaptive --config configs/blihat_final.yaml --video drone.mp4

# Give the initial box (x,y,w,h) and save the output video
python demo.py --tracker adaptive --config configs/blihat_final.yaml --video drone.mp4 \
    --bbox 100,200,50,80 --save out
```

### Benchmarks

Datasets are not included; download UAV123 / UAV20L / DTB70 from their official sources. Evaluation uses the [got10k toolkit](https://github.com/got-10k/toolkit).

```bash
# Full UAV123 with the final config
python eval/run_benchmark.py --tracker adaptive --config configs/blihat_final.yaml \
    --dataset UAV123 --dataset-root /path/to/UAV123 --result-dir runs/full --name blihat_final

# Coarse-only baseline with identical tracking
python eval/run_benchmark.py --tracker adaptive --config configs/blihat_final.yaml \
    --set adaptive.fine_enabled=false \
    --dataset UAV123 --dataset-root /path/to/UAV123 --result-dir runs/full --name coarse_final

# Score a result folder (optionally against a baseline)
python eval/analyze_uav123.py --results runs/full/UAV123/blihat_final \
    --anno /path/to/UAV123/anno/UAV123 --baseline runs/full/UAV123/coarse_final
```

Useful options: `--sequences eval/uav123_scale.txt` (subset), `--max-frames 300`, `--set SECTION.KEY=VALUE` (override any config value).

Analysis scripts:

| Script | Purpose |
|--------|---------|
| `eval/analyze_uav123.py` | Success/precision per sequence, class and attribute; comparison to a baseline |
| `eval/analyze_fine_box.py` | Box quality of the DINO vs. LightGlue box and how often each is used |
| `eval/analyze_diagnostics.py` | Re-detection events and silent losses from per-frame diagnostics |
| `eval/check_features.py` | Quick check of how discriminative the DINO response is |

### Debug output

`--debug-dir` saves, for every frame, the queried template and the DINO response:

```bash
python demo.py --tracker adaptive --video drone.mp4 --bbox x,y,w,h --no-display \
    --debug-dir debug/ --debug-raw --save out    # also writes out_debug.mp4
```

- `debug/template/NNNNNN.png`: template crops, PCA of the memory kernels and patch weights
- `debug/heatmap/NNNNNN.png`: heatmap over the search crop with peak, box and gate values
- `debug/raw/NNNNNN.npz` (with `--debug-raw`): raw responses and kernels

In your own loop, set `tracker.debug = True` and pass `result['debug']` to `blihat.utils.debug_viz.DebugRecorder.write()`.

### Tests

CPU-only smoke test: runs `AdaptiveTracker` with a fake DINO backbone on a synthetic sequence (no GPU or model download; the fine stage runs if LightGlue is installed):

```bash
python tests/run_seq.py . blihat_final.yaml /tmp/out.npz --n 30
```

## Project structure

```
Blihat/
├── blihat/
│   ├── adaptive_tracker.py   # AdaptiveTracker (reported results)
│   ├── tracker.py            # CascadeTracker (baseline cascade)
│   ├── config.py             # Dataclass configuration + YAML loading
│   ├── backbone/             # DINOv3 / DINOv2 feature extraction
│   ├── coarse/               # Correlation, heatmap, template memory
│   ├── fine/                 # SuperPoint + LightGlue, geometry, keyframes, confidence gate
│   └── utils/                # Boxes, background removal, debug visualisation
├── configs/                  # YAML configurations (blihat_final.yaml = reported results)
├── eval/                     # Benchmark runner and analysis scripts
├── tests/                    # CPU smoke test (synthetic data)
├── demo.py
└── RESULTS.txt               # Full results and ablations
```

## Limitations

- A single similarity transform cannot fit deforming targets (people, animals); these often fall back to the DINO box.
- Aspect ratio is fixed to the first frame.
- With a thin target and a lot of background in the first box, LightGlue can lock onto the background (see `RESULTS.txt`, DTB70 Girl2).
- Re-detection after full occlusion or long losses remains the main weakness on long sequences (UAV20L).

## License

This project is licensed under the [Apache License 2.0](LICENSE). Copyright 2026 Dagimawi Moges Wakumo.

It builds on third-party models that have **their own licenses**, which are not covered by this repository's license. In particular, the DINOv3 weights are under Meta's DINOv3 License, and the SuperPoint weights used by LightGlue are restricted to non-commercial use. See [NOTICE](NOTICE) before any commercial use.

## Acknowledgements

[DINOv3](https://github.com/facebookresearch/dinov3) and [DINOv2](https://github.com/facebookresearch/dinov2) (Meta AI), [LightGlue](https://github.com/cvg/LightGlue) (ETH Zurich CVG), [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork) (Magic Leap), [rembg](https://github.com/danielgatis/rembg), and the [got10k toolkit](https://github.com/got-10k/toolkit). Benchmarks: UAV123 / UAV20L (Mueller et al., ECCV 2016) and DTB70 (Li & Yeung, AAAI 2017).

## Citation

```bibtex
@misc{wakumo2026blihat,
  title  = {Blihat: Training-Free Zero-Shot Aerial Tracking via Multi-Scale Foundation Correlation and Geometry-Locked Keypoint Correspondence},
  author = {Wakumo, Dagimawi Moges},
  year   = {2026},
  howpublished = {\url{https://github.com/DagimawiMoges/Zero-shot-UAV-object-tracking-with-coarse-to-fine-architecture}}
}
```
