# Copyright 2026 Dagimawi Moges Wakumo
# SPDX-License-Identifier: Apache-2.0
import yaml
from dataclasses import dataclass, asdict, field

@dataclass
class BackboneConfig:
    name: str = "dinov3_vitl16"  # DINOv3 ViT-L via HuggingFace (DINOv2 variants available as fallback)
    patch_size: int = 16
    feature_layers: tuple = (8, 9, 10, 11)  # Last 4 layers
    coarse_layer_idx: int = -1  # Use deepest layer for coarse stage
    normalize_features: bool = True
    final_norm: bool = True  # DINOv3: apply the final LayerNorm to hidden states (False = pre-fix behaviour)
    dtype: str = "float32"   # DINOv3 weights/compute: float32 | bfloat16 | float16 (use bfloat16 for dinov3_vit7b16)

@dataclass
class CoarseConfig:
    search_scale: float = 4.0
    template_scale: float = 2.0
    cosine_window_weight: float = 0.3
    multi_scale_factors: tuple = (1.0,)
    peak_threshold: float = 0.15
    psr_threshold: float = 2.0          # Minimum Peak-to-Sidelobe Ratio for target presence
    sharpness_threshold: float = 1.15    # Minimum peak-to-second-peak ratio
    bg_removal: bool = True          # Remove background from template crops (rembg)
    bg_removal_fill: str = "zero"    # "zero" or "mean" (ImageNet mean)

@dataclass  
class FineConfig:
    matcher: str = "lightglue"  # only "lightglue" is implemented
    max_keypoints: int = 1024
    ransac_threshold: float = 3.0
    ransac_method: str = "USAC_MAGSAC"  # cv2 RANSAC method
    inlier_threshold_high: float = 0.5
    inlier_threshold_low: float = 0.15
    min_matches: int = 4
    crop_scale: float = 4.0
    area_ratio_bounds: tuple = (0.25, 4.0)  # Reject if projected area outside this range

@dataclass
class TrackingConfig:
    ema_position: float = 0.0
    ema_size: float = 0.8
    template_update_interval: int = 10
    template_update_threshold: float = 0.6
    template_update_alpha: float = 0.90  # EMA alpha: 90% history, 10% new observation
    lost_threshold: float = 0.3
    lost_patience: int = 5
    velocity_smoothing: int = 3
    matcher_every_n: int = 1

@dataclass
class AdaptiveConfig:
    """Settings for AdaptiveTracker (dynamic DINO memory + keyframe geometry)."""
    # ── Coarse geometry (DINO) ──
    template_patches: int = 12          # template grid side; target spans template_patches / template_scale patches
    template_scale: float = 2.0         # context around the target in the template crop
    search_scale: float = 4.0           # base search area (x target size)
    max_search_scale: float = 6.0       # upper bound when widening after low-quality frames
    widen_factor: float = 1.25
    widen_patience: int = 3             # consecutive low-quality frames before widening
    kernel_margin: int = 1              # context patches kept around the target in the kernel
    context_weight: float = 0.3         # weight of context patches in the correlation
    fg_mask: bool = True                # weight target patches with a U2-Net mask (frame 1 only)
    fg_mask_floor: float = 0.3          # min weight of target patches under the mask
    window_weight: float = 0.3          # Hann window penalty strength
    scales: tuple = (0.9639, 1.0, 1.0375)
    scale_penalty: float = 0.9745
    scale_lr: float = 0.59
    scale_q_min: float = -1.0           # q_rel needed to apply a scale change; <0 = same gate as memory updates (q_min & id_min)
    # ── Memory ──
    memory_weights: tuple = (0.3, 0.3, 0.4)   # anchor, long-term, short-term
    long_lr: float = 0.05
    short_len: int = 3
    stats_window: int = 30
    q_min: float = 0.8                  # min peak / median(peak) to update
    id_min: float = 0.85                # min anchor-similarity / its median to update
    dino_only_strength: float = 0.5     # update strength when the fine stage gave no verdict
    # ── Fine stage (SuperPoint + LightGlue keyframes) ──
    fine_enabled: bool = True
    max_keypoints: int = 1024
    target_norm_px: float = 160.0       # target is resampled to ~this size before SuperPoint
    fine_crop_scale: float = 2.0
    num_keyframes: int = 2
    keyframe_interval: int = 10
    min_inliers: int = 8
    ransac_threshold: float = 3.0       # pixels in the normalised crop
    agree_radius: float = 1.5           # max fine/DINO center distance, in DINO patches
    agree_scale_tol: float = 1.25       # max fine/DINO size ratio
    fine_size_lr: float = 0.5           # "agree" mode: fraction (in log space) of the fine size change applied
    # ── Box source (who produces the output box) ──
    # "agree": fine box only if it agrees with DINO (original behaviour, the default)
    # "fine":  DINO tracks, LightGlue gives the box whenever its fit is reliable, DINO box otherwise
    # "dino":  fine stage runs for diagnostics only; output and memory exactly as coarse-only
    box_source: str = "agree"
    fine_min_target_px: float = 0.0     # "fine"/"dino": no fine box below this target size sqrt(w*h) (0 = off)
    fine_min_inliers: int = 10          # "fine"/"dino": reliability check - min RANSAC inliers
    fine_min_inlier_ratio: float = 0.3  # ... min inliers / LightGlue matches
    fine_max_scale_step: float = 1.2    # ... max size ratio vs the previous output box, per frame
    fine_scale_step_frames: int = 5     # ... the step compounds per frame without a fine box; after this many, no size check
    fine_scale_lock: bool = False       # ... True: the size check never switches off (step capped at step**frames)
    fine_max_offset: float = 0.75       # ... max fit-center offset from the DINO center, in target sizes
                                        #     (the match crop spans +-fine_crop_scale/2 target sizes)
    fine_box_size_lr: float = 1.0       # "fine": fraction (log space) of the fine size change applied (1 = pure LightGlue)
    fine_memory_obs: str = "search"     # "fine": DINO memory observation on fine frames:
                                        #   "search" = search-grid window nearest the fine center (free)
                                        #   "template" = re-extract a template crop at the fine box (+1 small forward)
                                        #   "dino" = DINO peak window (as in "agree")
    redetect_fine: str = "off"          # re-detection + fine stage: "off" | "refine" (use the fine box if reliable)
                                        #   | "require" (reject jumps without a reliable fit); ignored for "dino"
    # ── Motion ──
    velocity_smoothing: int = 3
    motion_min_history: int = 2         # healthy velocity samples needed before the motion prior is used
    motion_coast_frames: int = 0        # keep extrapolating (decayed) velocity through this many low-quality frames
    velocity_decay: float = 0.8         # per low-quality frame decay of the coasting velocity
    # ── Recovery (defaults = original behaviour; see configs/adaptive_recovery.yaml) ──
    widen_single_scale: bool = False    # while the search is widened, search only scale 1.0 (bounds compute)
    stats_mode: str = "all"             # frames feeding the q_rel/id_rel medians: "all" | "healthy"
    stats_leak_every: int = 10          # "healthy" mode: still record every k-th low-quality frame (absorbs real appearance change)
    redetect: bool = False              # tiled re-detection around the last confident position while lost
    redetect_patience: int = 5          # consecutive low-quality frames before the first attempt
    redetect_interval: int = 5          # frames between attempts while still lost
    redetect_grid: int = 3              # grid x grid search tiles (coverage ~ grid x reachable span per axis)
    redetect_batch: int = 3             # tiles per backbone forward pass
    redetect_q_min: float = 0.85        # accept if peak >= q_min x median healthy peak
    redetect_id_min: float = 0.85       # ... and anchor similarity >= id_min x median healthy similarity


@dataclass
class Config:
    backbone: BackboneConfig
    coarse: CoarseConfig
    fine: FineConfig
    tracking: TrackingConfig
    device: str = "cuda"
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from a YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            
        return cls(
            backbone=BackboneConfig(**data.get('backbone', {})),
            coarse=CoarseConfig(**data.get('coarse', {})),
            fine=FineConfig(**data.get('fine', {})),
            tracking=TrackingConfig(**data.get('tracking', {})),
            device=data.get('device', 'cuda'),
            adaptive=AdaptiveConfig(**data.get('adaptive', {}))
        )
    
    @classmethod
    def default(cls) -> 'Config':
        """Return default configuration."""
        return cls(
            backbone=BackboneConfig(),
            coarse=CoarseConfig(),
            fine=FineConfig(),
            tracking=TrackingConfig()
        )
    
    def to_yaml(self, path: str):
        """Save configuration to a YAML file."""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
