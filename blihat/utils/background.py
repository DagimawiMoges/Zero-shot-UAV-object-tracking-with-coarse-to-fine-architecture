# Copyright 2026 Dagimawi Moges Wakumo
# SPDX-License-Identifier: Apache-2.0
"""
Background removal for template crops using rembg (U²-Net).

Isolates the foreground object in a template crop so that DINOv2 features
extracted from the template represent only the target, not surrounding clutter.
"""

import numpy as np
import cv2
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# ImageNet mean in BGR order (used as neutral fill when bg_fill="mean")
_IMAGENET_MEAN_BGR = np.array([103.53, 116.28, 123.675], dtype=np.float32)


class BackgroundRemover:
    """Removes background from image crops using rembg.

    The rembg model is loaded lazily on the first call to ``remove_background``
    so that no GPU memory is consumed if the feature is disabled in config.
    """

    def __init__(self, fill: str = "zero"):
        """
        Args:
            fill: What to put in place of background pixels.
                  ``"zero"``  – fill with black (0, 0, 0).
                  ``"mean"``  – fill with ImageNet mean colour.
        """
        if fill not in ("zero", "mean"):
            raise ValueError(f"Unsupported bg fill mode: {fill!r}. Use 'zero' or 'mean'.")
        self.fill = fill
        self._session = None  # lazy

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    def _ensure_session(self):
        """Import rembg and create a reusable inference session."""
        if self._session is not None:
            return

        try:
            from rembg import new_session
        except ImportError:
            raise ImportError(
                "rembg is not installed. Install it with:\n"
                "  pip install rembg[gpu]   # for CUDA acceleration\n"
                "  pip install rembg        # CPU-only fallback"
            )

        logger.info("Loading rembg background-removal model (first use)…")
        self._session = new_session(model_name="u2net")
        logger.info("rembg model loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def remove_background(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove background from a BGR image crop.

        Args:
            image: (H, W, 3) uint8 BGR image.

        Returns:
            masked_image: (H, W, 3) uint8 BGR image with background replaced.
            mask: (H, W) float32 in [0, 1], foreground probability.
        """
        self._ensure_session()

        from rembg import remove

        # rembg expects RGB input and returns RGBA
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgba = remove(rgb, session=self._session)  # (H, W, 4) uint8

        # Extract the alpha channel as the foreground mask
        alpha = rgba[:, :, 3].astype(np.float32) / 255.0  # [0, 1]

        # Build the fill colour
        if self.fill == "zero":
            fill_color = np.zeros(3, dtype=np.float32)
        else:  # "mean"
            fill_color = _IMAGENET_MEAN_BGR

        # Blend: foreground * alpha + fill * (1 - alpha)
        alpha_3 = alpha[:, :, np.newaxis]  # (H, W, 1)
        masked = (
            image.astype(np.float32) * alpha_3
            + fill_color[np.newaxis, np.newaxis, :] * (1.0 - alpha_3)
        )
        masked = np.clip(masked, 0, 255).astype(np.uint8)

        return masked, alpha
