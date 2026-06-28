"""SAMSegmenter — field boundary segmentation via Segment Anything Model.

Uses the HuggingFace Inference API (free tier) to run facebook/sam-vit-base.
Given a satellite imagery patch and the approximate centre of a plot, SAM
returns a binary mask of the field it finds there.

The mask is used as a second confidence signal in the pipeline:
  - If SAM produces a mask that overlaps the cross-correlation-corrected plot,
    the confidence is boosted.
  - If SAM finds no plausible field (low mask quality score), the raw
    cross-correlation confidence is left unchanged.

Graceful degradation
--------------------
This module is fully optional.  If ``HF_TOKEN`` is absent, the API returns an
error, or the request times out, :meth:`SAMSegmenter.segment` returns ``None``
and the pipeline continues without the ML boost.  The core pipeline never
depends on this module for correctness.

HuggingFace free tier limits
-----------------------------
- Model: ``facebook/sam-vit-base``
- Rate: ~300 requests/hour on the free Inference API
- Max image size: 1024×1024 px (we resize to 512 px on the longest side)
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_HF_API_URL = (
    "https://api-inference.huggingface.co/models/facebook/sam-vit-base"
)
_MAX_SIDE_PX = 512      # resize before sending to stay within free-tier limits
_REQUEST_TIMEOUT = 15   # seconds


class SAMSegmenter:
    """Wraps the HuggingFace Inference API for SAM field segmentation.

    Parameters
    ----------
    hf_token:
        HuggingFace API token (``HF_TOKEN`` env var is used when omitted).
        Pass an empty string or ``None`` to run in no-op mode (always returns
        ``None`` without making any network calls).
    max_retries:
        Number of times to retry on transient errors (503 model loading).
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        max_retries: int = 2,
    ) -> None:
        token = hf_token or os.environ.get("HF_TOKEN", "")
        self._token      = token.strip()
        self._max_retries = max_retries
        self._enabled    = bool(self._token)
        if not self._enabled:
            log.info(
                "SAMSegmenter: HF_TOKEN not set — running in no-op mode. "
                "Set HF_TOKEN in .env to enable ML field segmentation."
            )

    # ── public API ────────────────────────────────────────────────────────────

    def segment(
        self,
        image_rgb: np.ndarray,
        point_xy: tuple[int, int],
    ) -> Optional[np.ndarray]:
        """Segment the field containing *point_xy* in *image_rgb*.

        Parameters
        ----------
        image_rgb:
            ``(H, W, 3)`` uint8 RGB array — the satellite patch from
            :func:`bhume.geo.patch_for_plot`.
        point_xy:
            ``(col, row)`` pixel coordinate of the plot centre in *image_rgb*.

        Returns
        -------
        Optional[np.ndarray]
            Binary uint8 mask of shape ``(H, W)`` where 1 = field, 0 = other.
            Returns ``None`` on any failure (API error, timeout, no token).
        """
        if not self._enabled:
            return None

        try:
            return self._call_api(image_rgb, point_xy)
        except Exception as exc:  # noqa: BLE001
            log.debug("SAM segment() failed: %s", exc)
            return None

    def confidence_boost(
        self,
        mask: Optional[np.ndarray],
        base_confidence: float,
        plot_mask: np.ndarray,
    ) -> float:
        """Compute a boosted confidence using the SAM mask overlap.

        Parameters
        ----------
        mask:
            SAM binary mask (or ``None`` when unavailable).
        base_confidence:
            Raw cross-correlation confidence from :class:`BoundaryAligner`.
        plot_mask:
            Binary mask of the corrected plot rasterised at the same resolution
            as *mask*.

        Returns
        -------
        float
            Boosted confidence in [0, 1].  If *mask* is ``None`` or has no
            overlap with *plot_mask*, *base_confidence* is returned unchanged.
        """
        if mask is None or plot_mask is None:
            return base_confidence

        try:
            if mask.shape != plot_mask.shape:
                return base_confidence
            intersection = float(np.logical_and(mask, plot_mask).sum())
            union        = float(np.logical_or(mask, plot_mask).sum())
            if union < 1:
                return base_confidence
            iou = intersection / union
            # Blend: 70% base + 30% SAM IoU
            blended = 0.70 * base_confidence + 0.30 * iou
            return float(np.clip(blended, 0.0, 1.0))
        except Exception:
            return base_confidence

    # ── private helpers ───────────────────────────────────────────────────────

    def _call_api(
        self,
        image_rgb: np.ndarray,
        point_xy: tuple[int, int],
    ) -> Optional[np.ndarray]:
        """Make the HuggingFace Inference API call with retry logic."""
        import requests
        from PIL import Image

        h, w = image_rgb.shape[:2]
        scale = min(1.0, _MAX_SIDE_PX / max(h, w))
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        pil_img = Image.fromarray(image_rgb).resize((new_w, new_h))

        scaled_x = int(point_xy[0] * scale)
        scaled_y = int(point_xy[1] * scale)

        # Encode image as base64 PNG
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        payload = {
            "inputs": {
                "image":  img_b64,
                "points": [[scaled_x, scaled_y]],
                "labels": [1],          # 1 = foreground point
            }
        }
        headers = {"Authorization": f"Bearer {self._token}"}

        for attempt in range(self._max_retries + 1):
            try:
                resp = requests.post(
                    _HF_API_URL,
                    json=payload,
                    headers=headers,
                    timeout=_REQUEST_TIMEOUT,
                )
                if resp.status_code == 503:
                    # Model is loading — wait and retry
                    wait = float(resp.json().get("estimated_time", 10))
                    log.debug("SAM model loading, waiting %.1fs …", wait)
                    time.sleep(min(wait, 20))
                    continue
                if resp.status_code != 200:
                    log.debug("SAM API %d: %s", resp.status_code, resp.text[:120])
                    return None

                return self._parse_response(resp.json(), h, w, scale)

            except requests.exceptions.Timeout:
                log.debug("SAM request timed out (attempt %d)", attempt + 1)
            except Exception as exc:
                log.debug("SAM request error: %s", exc)
                return None

        return None

    @staticmethod
    def _parse_response(
        response: list | dict,
        orig_h: int,
        orig_w: int,
        scale: float,
    ) -> Optional[np.ndarray]:
        """Extract and upscale the best mask from the API response."""
        from PIL import Image

        # SAM returns a list of dicts: [{score, label, mask}, ...]
        # or a single dict — normalise to a list
        if isinstance(response, dict):
            response = [response]
        if not response:
            return None

        # Pick the mask with the highest score
        best = max(response, key=lambda r: r.get("score", 0.0))
        mask_data = best.get("mask")
        if mask_data is None:
            return None

        # Decode base64 PNG mask
        try:
            raw = base64.b64decode(mask_data)
            mask_img = Image.open(io.BytesIO(raw)).convert("L")
        except Exception:
            return None

        # Resize back to the original image dimensions
        if scale < 1.0:
            mask_img = mask_img.resize((orig_w, orig_h), Image.NEAREST)

        mask_arr = (np.array(mask_img) > 127).astype(np.uint8)
        return mask_arr
