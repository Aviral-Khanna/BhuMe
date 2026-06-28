"""GeminiExplainer — per-plot correction explanation via Gemini Flash 2.0.

Uses Google's Generative AI SDK (free tier):
  - Model  : gemini-2.0-flash
  - Quota  : 15 requests/min · 1 M tokens/day  (no credit card required)
  - SDK    : google-genai (pip install google-genai)

For each corrected or flagged plot the explainer produces a short natural-
language explanation that is shown in the frontend Plot Detail panel.

Example output
--------------
  "Plot 1145 was shifted 8.9 m west and 6.9 m north onto the field edge that
  is clearly visible as a dark bund line in the satellite imagery.  Confidence
  is high (0.83) because the cross-correlation peak is sharp and the detected
  field boundary aligns cleanly with all four sides of the polygon."

Graceful degradation
--------------------
If ``GEMINI_API_KEY`` is absent, the SDK is not installed, or the API returns
an error, :meth:`GeminiExplainer.explain` returns a plain-text fallback built
from the structured data — no external call is made.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from solution.types import AlignmentResult

log = logging.getLogger(__name__)

_MODEL    = "gemini-2.0-flash"
_MAX_TOKENS = 150   # keep explanations short
_RPM_DELAY  = 4.1   # seconds between calls to stay within 15 RPM free limit


class GeminiExplainer:
    """Generates natural-language plot-correction explanations via Gemini Flash.

    Parameters
    ----------
    api_key:
        Google Gemini API key.  Falls back to the ``GEMINI_API_KEY`` env var
        when not provided.  Pass an empty string to force offline/fallback mode.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._api_key = key.strip()
        self._client  = None
        self._last_call: float = 0.0

        if self._api_key:
            try:
                import google.generativeai as genai  # type: ignore[import]
                genai.configure(api_key=self._api_key)
                self._client = genai.GenerativeModel(_MODEL)
                log.info("GeminiExplainer: using model %s", _MODEL)
            except ImportError:
                log.warning(
                    "google-generativeai not installed; run "
                    "`uv add google-generativeai` to enable AI explanations."
                )
        else:
            log.info(
                "GeminiExplainer: GEMINI_API_KEY not set — "
                "returning structured fallback explanations."
            )

    # ── public API ────────────────────────────────────────────────────────────

    def explain(
        self,
        plot_number: str,
        result: AlignmentResult,
        plot_props: dict,
    ) -> str:
        """Return a short explanation of the correction for *plot_number*.

        Parameters
        ----------
        plot_number:
            The plot identifier (for logging).
        result:
            :class:`AlignmentResult` from the pipeline.
        plot_props:
            A dict of plot properties from ``village.plots`` (area fields,
            surveys, village name, etc.).

        Returns
        -------
        str
            One or two plain-English sentences.  Never raises.
        """
        try:
            if self._client is not None:
                return self._call_gemini(plot_number, result, plot_props)
        except Exception as exc:
            log.debug("Gemini explain() failed for plot %s: %s", plot_number, exc)
        return self._fallback(plot_number, result, plot_props)

    def explain_batch(
        self,
        items: list[tuple[str, AlignmentResult, dict]],
    ) -> dict[str, str]:
        """Explain multiple plots, respecting the free-tier 15 RPM limit.

        Parameters
        ----------
        items:
            List of ``(plot_number, result, props)`` tuples.

        Returns
        -------
        dict[str, str]
            Mapping ``plot_number → explanation``.
        """
        out: dict[str, str] = {}
        for pn, result, props in items:
            out[pn] = self.explain(pn, result, props)
        return out

    # ── private helpers ───────────────────────────────────────────────────────

    def _call_gemini(
        self,
        plot_number: str,
        result: AlignmentResult,
        plot_props: dict,
    ) -> str:
        # Throttle to stay within 15 RPM
        elapsed = time.monotonic() - self._last_call
        if elapsed < _RPM_DELAY:
            time.sleep(_RPM_DELAY - elapsed)

        prompt = self._build_prompt(plot_number, result, plot_props)
        response = self._client.generate_content(  # type: ignore[union-attr]
            prompt,
            generation_config={"max_output_tokens": _MAX_TOKENS, "temperature": 0.3},
        )
        self._last_call = time.monotonic()
        text = response.text.strip()
        log.debug("Gemini explanation for plot %s: %s", plot_number, text[:80])
        return text

    @staticmethod
    def _build_prompt(
        plot_number: str,
        result: AlignmentResult,
        plot_props: dict,
    ) -> str:
        map_area = plot_props.get("map_area_sqm")
        rec_area = plot_props.get("recorded_area_sqm")
        ratio    = f"{map_area/rec_area:.2f}" if map_area and rec_area else "unknown"
        village  = plot_props.get("village", "unknown village")

        direction = _compass(result.dx_m, result.dy_m)
        distance  = (result.dx_m**2 + result.dy_m**2) ** 0.5

        if result.status == "flagged":
            action = "was flagged as uncertain and kept at its official position"
            reason = result.method_note
        else:
            action = (
                f"was corrected by shifting {distance:.1f} m {direction} "
                f"(dx={result.dx_m:.1f} m, dy={result.dy_m:.1f} m)"
            )
            reason = (
                f"Cross-correlation SNR confidence = {result.raw_confidence:.2f}, "
                f"calibrated confidence = {result.calibrated_confidence:.2f}."
            )

        return (
            f"You are explaining a land-boundary correction to a non-technical user.\n\n"
            f"Plot {plot_number} in {village} {action}.\n"
            f"Map area: {map_area} m²  Recorded area: {rec_area} m²  "
            f"Drawn/recorded ratio: {ratio}.\n"
            f"Technical detail: {reason}\n\n"
            f"Write 1–2 plain-English sentences explaining what happened to this plot "
            f"and why the correction was or was not made. "
            f"Do not use jargon. Keep it under 50 words."
        )

    @staticmethod
    def _fallback(
        plot_number: str,
        result: AlignmentResult,
        plot_props: dict,
    ) -> str:
        """Structured fallback used when Gemini is unavailable."""
        if result.status == "flagged":
            return (
                f"Plot {plot_number} could not be reliably corrected and was "
                f"kept at its official position. Reason: {result.method_note[:80]}."
            )
        distance = (result.dx_m**2 + result.dy_m**2) ** 0.5
        direction = _compass(result.dx_m, result.dy_m)
        return (
            f"Plot {plot_number} was shifted {distance:.1f} m {direction} "
            f"onto the detected field edge "
            f"(confidence {result.calibrated_confidence:.2f})."
        )


# ── utilities ─────────────────────────────────────────────────────────────────

def _compass(dx: float, dy: float) -> str:
    """Return a cardinal/intercardinal direction string for a (dx, dy) shift."""
    import math
    angle = math.degrees(math.atan2(dy, dx))  # 0° = east, 90° = north
    # Round to nearest 45°
    dirs  = ["east", "northeast", "north", "northwest",
             "west", "southwest", "south", "southeast"]
    idx   = int((angle + 22.5) % 360 / 45)
    return dirs[idx % 8]
