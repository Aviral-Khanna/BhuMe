"""GeminiVisionAnalyzer — boundary visibility scoring via Gemini 2.5 Flash vision.

For each uncertain plot, this module sends the satellite imagery patch to Gemini
and asks it to assess how visible the agricultural field boundary is.  The score
is used to adjust the classical alignment confidence before rank-normalisation,
providing a genuinely independent second signal.

Why this is legitimate
----------------------
The classical confidence (SNR × density × consistency) measures how well the
cross-correlation found a feature in boundaries.tif.  Gemini directly *sees* the
raw satellite imagery and reports whether a field edge is visually present —
completely independent of the raster processing pipeline.  Agreement between the
two signals means the correction is likely correct.  Disagreement (e.g., high
classical SNR but Gemini says no visible boundary) flags an uncertain correction.

Which plots are analysed
------------------------
Running Gemini on all 2,457 / 2,508 plots would take hours even with generous
rate limits.  Instead we analyse only the *uncertain zone* — the bottom
``uncertain_pct`` fraction of plots sorted by raw_confidence.  Plots already
clearly correct (high confidence) or clearly wrong (already flagged at confidence=0)
skip analysis; the uncertain middle is where vision adds the most value.

Graceful degradation
--------------------
If the API key is absent, the API quota is exhausted, the imagery file is
missing, or any single plot call fails, the analyzer falls back silently — the
classical confidence is returned unchanged.  The pipeline always runs.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

#: Gemini model to use for vision analysis.
GEMINI_MODEL: str = "models/gemini-2.5-flash"

#: Maximum imagery patch size (pixels) sent to Gemini.
#: Resized to this on the longest side before encoding.
MAX_PATCH_PX: int = 256

#: Seconds to wait between Gemini calls.
#: Free tier allows 5 requests/min → 12 s minimum between calls.
CALL_DELAY_S: float = 13.0

#: Maximum seconds to wait when retrying after a 429 / 503 response.
MAX_RETRY_WAIT_S: float = 30.0

#: Fraction of plots (by raw_confidence, lowest first) to analyse with Gemini.
#: 0.30 = bottom 30 % of corrected plots.
UNCERTAIN_FRACTION: float = 0.30

#: Weight of the Gemini vision score in the blended confidence.
#: final_raw = (1 - BLEND_W) * classical + BLEND_W * gemini
GEMINI_BLEND_WEIGHT: float = 0.30

#: The prompt sent to Gemini for each imagery patch.
_BOUNDARY_PROMPT: str = (
    "This is a satellite image of an agricultural land parcel in Maharashtra, India.\n\n"
    "Assess the visibility of the field boundary and respond with valid JSON only — "
    "no markdown, no extra text:\n"
    '{"boundary_confidence": <0.0-1.0>, "edge_quality": "<clear|moderate|weak|none>"}\n\n'
    "boundary_confidence: probability that a clear agricultural field edge is visible "
    "near the centre of the image (1.0 = very clear, 0.0 = no visible boundary).\n"
    "edge_quality: subjective clarity of the dominant boundary in the image."
)


# ── main class ────────────────────────────────────────────────────────────────

class GeminiVisionAnalyzer:
    """Scores boundary visibility for uncertain plots using Gemini vision.

    Parameters
    ----------
    api_key:
        Google Gemini API key.  Reads ``GEMINI_API_KEY`` from the environment
        when not provided.  Pass an empty string to run in no-op mode.
    imagery_path:
        Path to the village ``imagery.tif`` file.
    max_plots:
        Hard cap on the number of Gemini calls per village run.
    """

    def __init__(
        self,
        api_key:      Optional[str] = None,
        imagery_path: Optional[Path] = None,
        max_plots:    int = 50,
    ) -> None:
        key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
        self._enabled      = bool(key) and (imagery_path is not None)
        self._imagery_path = imagery_path
        self._max_plots    = max_plots
        self._client       = None

        if self._enabled:
            try:
                from google import genai as _genai  # type: ignore[import]
                self._client = _genai.Client(api_key=key)
                self._genai  = _genai
                log.info("GeminiVisionAnalyzer ready — model %s", GEMINI_MODEL)
            except ImportError:
                log.warning(
                    "google-genai not installed; run `uv add google-genai` "
                    "to enable vision analysis."
                )
                self._enabled = False
        else:
            log.info(
                "GeminiVisionAnalyzer: disabled "
                "(GEMINI_API_KEY missing or imagery_path not set)."
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── public API ─────────────────────────────────────────────────────────────

    def analyse_uncertain_plots(
        self,
        results:       list,        # list[AlignmentResult] — imported lazily to avoid circular
        plots_gdf,                  # GeoDataFrame
        uncertain_pct: float = UNCERTAIN_FRACTION,
    ) -> list:
        """Blend Gemini vision scores into raw_confidence for uncertain plots.

        Only the bottom ``uncertain_pct`` fraction of corrected plots (by
        raw_confidence) are sent to Gemini — clearing high-confidence plots and
        already-zero-confidence plots from the API queue.

        Returns a new list of :class:`~solution.types.AlignmentResult` with
        ``raw_confidence`` updated where Gemini was called.  Results for plots
        not analysed are returned unchanged.
        """
        if not self._enabled:
            return results

        from solution.types import AlignmentResult  # lazy import

        corrected_idx = [
            i for i, r in enumerate(results)
            if r.status == "corrected" and r.raw_confidence > 0
        ]
        if not corrected_idx:
            return results

        # Sort corrected plots by confidence, take bottom uncertain_pct
        corrected_idx.sort(key=lambda i: results[i].raw_confidence)
        n_uncertain = min(
            int(len(corrected_idx) * uncertain_pct),
            self._max_plots,
        )
        uncertain_idx = corrected_idx[:n_uncertain]

        log.info(
            "GeminiVisionAnalyzer: analysing %d uncertain plots "
            "out of %d corrected",
            len(uncertain_idx), len(corrected_idx),
        )

        updated = list(results)
        done, skipped = 0, 0

        from bhume.geo import open_imagery, patch_for_plot
        from pyproj import Transformer
        from shapely.affinity import translate
        from shapely.ops import transform as shp_transform
        from solution.align import _utm_for, _reproject

        with open_imagery(self._imagery_path) as imagery_src:
            for list_idx in uncertain_idx:
                r  = results[list_idx]
                pn = r.plot_number

                try:
                    official_geom = plots_gdf.loc[pn, "geometry"]
                except KeyError:
                    skipped += 1
                    continue

                # Use the CORRECTED position for imagery extraction.
                # Asking Gemini "is there a field edge here?" at the corrected
                # position is far more informative than asking at the official
                # (wrong) position.
                try:
                    utm_crs = _utm_for(official_geom)
                    to_utm  = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
                    frm_utm = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
                    off_u   = _reproject(official_geom, to_utm)
                    corr_u  = translate(off_u, r.dx_m, r.dy_m)
                    corrected_geom = shp_transform(
                        lambda xs, ys, zs=None: frm_utm.transform(xs, ys), corr_u
                    )
                except Exception:
                    corrected_geom = official_geom  # fallback

                gemini_conf = self._score_one(imagery_src, corrected_geom, patch_for_plot)
                if gemini_conf is None:
                    skipped += 1
                    continue

                # Blend: classical carries more weight, Gemini adds an independent signal
                blended_conf = (
                    (1.0 - GEMINI_BLEND_WEIGHT) * r.raw_confidence
                    + GEMINI_BLEND_WEIGHT * gemini_conf
                )
                blended_conf = float(np.clip(blended_conf, 0.0, 1.0))

                updated[list_idx] = AlignmentResult(
                    plot_number=r.plot_number,
                    dx_m=r.dx_m,
                    dy_m=r.dy_m,
                    raw_confidence=blended_conf,
                    calibrated_confidence=blended_conf,
                    status=r.status,
                    method_note=r.method_note + f" gemini={gemini_conf:.2f}",
                    boundary_density=r.boundary_density,
                    consistency_factor=r.consistency_factor,
                )
                done += 1
                time.sleep(CALL_DELAY_S)

        log.info(
            "GeminiVisionAnalyzer: done — %d analysed, %d skipped",
            done, skipped,
        )
        return updated

    # ── private ────────────────────────────────────────────────────────────────

    def _score_one(
        self,
        imagery_src,
        geom,
        patch_for_plot_fn,
    ) -> Optional[float]:
        """Return Gemini boundary_confidence [0,1] for one plot, or None on error."""
        try:
            patch = patch_for_plot_fn(imagery_src, geom, pad_m=20)
            img_bytes = _encode_patch(patch.image)
        except Exception as exc:
            log.debug("patch extraction failed: %s", exc)
            return None

        try:
            from google.genai import types as _types  # type: ignore[import]

            for attempt in range(3):
                try:
                    resp = self._client.models.generate_content(  # type: ignore[union-attr]
                        model=GEMINI_MODEL,
                        contents=[
                            _types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                            _BOUNDARY_PROMPT,
                        ],
                    )
                    return _parse_response(resp.text)
                except Exception as exc:
                    msg = str(exc)
                    # Parse retry_delay from 429/503 responses and wait
                    import re as _re
                    delay_match = _re.search(r"retry.*?(\d+)\s*s", msg, _re.I)
                    wait = min(
                        float(delay_match.group(1)) + 2 if delay_match else MAX_RETRY_WAIT_S,
                        MAX_RETRY_WAIT_S,
                    )
                    if attempt < 2 and ("429" in msg or "503" in msg):
                        log.debug(
                            "Gemini rate-limited, waiting %.0fs (attempt %d)",
                            wait, attempt + 1,
                        )
                        time.sleep(wait)
                    else:
                        log.debug("Gemini call failed: %s", msg[:80])
                        return None
            return None
        except Exception as exc:
            log.debug("Gemini setup failed: %s", exc)
            return None


# ── helpers ───────────────────────────────────────────────────────────────────

def _encode_patch(image: np.ndarray) -> bytes:
    """Resize to MAX_PATCH_PX on the longest side and encode as PNG bytes."""
    from PIL import Image

    h, w = image.shape[:2]
    scale = min(1.0, MAX_PATCH_PX / max(h, w, 1))
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))

    pil = Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _parse_response(text: str) -> Optional[float]:
    """Extract boundary_confidence from Gemini's JSON response."""
    try:
        # Strip markdown fences if present
        clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        data  = json.loads(clean)
        conf  = float(data.get("boundary_confidence", 0.5))
        return float(np.clip(conf, 0.0, 1.0))
    except Exception as exc:
        log.debug("response parse failed (%s): %r", exc, text[:120])
        # Try regex fallback
        m = re.search(r'"boundary_confidence"\s*:\s*([\d.]+)', text)
        if m:
            return float(np.clip(float(m.group(1)), 0.0, 1.0))
        return None
