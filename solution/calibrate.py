"""ConfidenceCalibrator — spreads confidence scores and flags uncertain plots.

Three-stage process
-------------------
1. **Area-ratio pre-filter**
   Plots whose drawn / recorded-area ratio is outside [area_ratio_min,
   area_ratio_max] have a *shape* error, not a placement error.
   Repositioning won't help: they are flagged before alignment matters.
   Plots with missing recorded area skip this filter.

2. **Rank-normalisation**
   ``raw_confidence`` at this point already encodes three signals multiplied
   together: SNR of the cross-correlation peak × boundary density factor ×
   neighbourhood consistency factor.  Rank-normalising spreads the distribution
   uniformly across [CONF_LO, CONF_HI], preserving the ordering while preventing
   all values from clustering near one point (which would hurt the Spearman /
   AUC calibration score).

3. **Flag threshold**
   Plots in the bottom ``flag_threshold`` fraction (by raw confidence, after the
   pre-filter) are converted to ``status="flagged"``.  Their geometry is kept as
   the official (original) position.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np

from solution.types import AlignmentResult

log = logging.getLogger(__name__)

_CONF_LO: float = 0.10
_CONF_HI: float = 0.90


class ConfidenceCalibrator:
    """Post-processes a village's :class:`~solution.types.AlignmentResult` list.

    Parameters
    ----------
    flag_threshold:
        Fraction of results (by raw confidence) to flag.  Default 0.15.
    area_ratio_min:
        Minimum acceptable drawn/recorded area ratio; plots below are pre-flagged.
    area_ratio_max:
        Maximum acceptable drawn/recorded area ratio; plots above are pre-flagged.
    """

    def __init__(
        self,
        flag_threshold:  float = 0.15,
        area_ratio_min:  float = 0.50,
        area_ratio_max:  float = 2.00,
    ) -> None:
        if not 0.0 <= flag_threshold <= 1.0:
            raise ValueError(f"flag_threshold must be in [0, 1], got {flag_threshold}")
        if area_ratio_min <= 0.0 or area_ratio_max <= area_ratio_min:
            raise ValueError("area_ratio_min must be > 0 and < area_ratio_max")

        self._flag_threshold = flag_threshold
        self._area_ratio_min = area_ratio_min
        self._area_ratio_max = area_ratio_max

    # ── public ──────────────────────────────────────────────────────────────────

    def calibrate(
        self,
        results:   list[AlignmentResult],
        plots_gdf: gpd.GeoDataFrame,
    ) -> list[AlignmentResult]:
        """Return a new list with calibrated confidence and status.

        Parameters
        ----------
        results:
            Raw :class:`AlignmentResult` objects from :class:`Predictor`
            (all ``status="corrected"``; ``raw_confidence`` encodes SNR ×
            density × consistency).
        plots_gdf:
            Village plots GeoDataFrame indexed by ``plot_number``.

        Returns
        -------
        list[AlignmentResult]
            Same length and order as *results* with:

            * ``calibrated_confidence`` ∈ [0.1, 0.9] for corrected plots
            * ``status="flagged"`` for pre-filtered or low-confidence plots
            * ``calibrated_confidence=0.0`` for flagged plots
        """
        if not results:
            return []

        area_flagged = self._area_filter(results, plots_gdf)
        calibrated   = self._rank_normalise(results, area_flagged)
        final        = self._apply_flag_threshold(calibrated, area_flagged)

        n_flagged = sum(1 for r in final if r.status == "flagged")
        log.info(
            "calibrated %d plots: %d corrected  %d flagged  "
            "(%d area-error pre-flags)",
            len(final),
            len(final) - n_flagged,
            n_flagged,
            sum(area_flagged),
        )
        return final

    # ── private ─────────────────────────────────────────────────────────────────

    def _area_filter(
        self,
        results:   list[AlignmentResult],
        plots_gdf: gpd.GeoDataFrame,
    ) -> list[bool]:
        """Return True for each plot that should be pre-flagged (area error)."""
        flagged: list[bool] = []
        for r in results:
            pn = r.plot_number
            try:
                row      = plots_gdf.loc[pn]
                map_area = float(row.get("map_area_sqm") or 0)
                rec_area = float(row.get("recorded_area_sqm") or 0)
                if map_area > 0 and rec_area > 0:
                    ratio = map_area / rec_area
                    if ratio < self._area_ratio_min or ratio > self._area_ratio_max:
                        flagged.append(True)
                        log.debug("plot %s area-ratio %.2f → pre-flagged", pn, ratio)
                        continue
            except (KeyError, TypeError, ValueError):
                pass
            flagged.append(False)
        return flagged

    def _rank_normalise(
        self,
        results:     list[AlignmentResult],
        area_flagged: list[bool],
    ) -> list[AlignmentResult]:
        """Rank-normalise raw confidences for non-pre-flagged plots to [CONF_LO, CONF_HI].

        The raw_confidence already encodes SNR × density × consistency.
        Rank-normalisation preserves this ordering while spreading the
        distribution so Spearman and AUC metrics are computed on a well-spread
        signal rather than values clustered near 0 or 1.
        """
        active_idx = [i for i, f in enumerate(area_flagged) if not f]

        if not active_idx:
            return results

        raw_confs = np.array([results[i].raw_confidence for i in active_idx])

        if len(raw_confs) == 1:
            norm_confs = np.array([(_CONF_LO + _CONF_HI) / 2.0])
        else:
            ranks      = np.argsort(np.argsort(raw_confs)).astype(float)
            norm_confs = _CONF_LO + (_CONF_HI - _CONF_LO) * ranks / (len(ranks) - 1)

        updated = list(results)
        for list_idx, norm_c in zip(active_idx, norm_confs):
            r = updated[list_idx]
            updated[list_idx] = AlignmentResult(
                plot_number=r.plot_number,
                dx_m=r.dx_m,
                dy_m=r.dy_m,
                raw_confidence=r.raw_confidence,
                calibrated_confidence=float(norm_c),
                status=r.status,
                method_note=r.method_note,
                boundary_density=r.boundary_density,
                consistency_factor=r.consistency_factor,
            )
        return updated

    def _apply_flag_threshold(
        self,
        results:     list[AlignmentResult],
        area_flagged: list[bool],
    ) -> list[AlignmentResult]:
        """Flag the bottom ``flag_threshold`` fraction by raw confidence.

        Area-error plots are always flagged regardless of confidence.
        """
        active_confs = [
            results[i].raw_confidence
            for i, f in enumerate(area_flagged)
            if not f
        ]
        cutoff = 0.0
        if active_confs and self._flag_threshold > 0.0:
            cutoff = float(np.quantile(active_confs, self._flag_threshold))

        final: list[AlignmentResult] = []
        for r, pre_flag in zip(results, area_flagged):
            if pre_flag or r.raw_confidence <= cutoff:
                reason = "area_error_flag" if pre_flag else "low_confidence_flag"
                final.append(AlignmentResult(
                    plot_number=r.plot_number,
                    dx_m=r.dx_m,
                    dy_m=r.dy_m,
                    raw_confidence=r.raw_confidence,
                    calibrated_confidence=0.0,
                    status="flagged",
                    method_note=f"{reason} {r.method_note}",
                    boundary_density=r.boundary_density,
                    consistency_factor=r.consistency_factor,
                ))
            else:
                final.append(r)
        return final
