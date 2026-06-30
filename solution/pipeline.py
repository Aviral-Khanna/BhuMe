"""Predictor — orchestrates the full village prediction pipeline.

Pipeline stages
---------------
1. **Global shift** — ``global_median_shift()`` estimates the village-level
   translation from example truths.  Falls back to a zero shift when no
   truths are present (per-plot alignment still runs from the official
   position with the wider search window in that case).

2. **Per-plot alignment** — :class:`BoundaryAligner` refines each plot with:
   - an *adaptive search radius* proportional to its physical size
   - a *boundary density factor* that down-weights low-signal patches

3. **Neighbourhood consistency** — after all plots are aligned, each plot's
   confidence is further scaled by how well its shift agrees with the shifts
   of its 20 nearest neighbours.  Outlier shifts (e.g. a plot moved 25 m
   when all its neighbours moved ~10 m) receive a strong penalty, preventing
   false-confident wrong corrections in dense urban areas.

3b. **Gemini vision pass** *(optional, requires GEMINI_API_KEY)* — for the
    bottom ``uncertain_fraction`` of corrected plots, the satellite imagery
    patch is sent to Gemini 2.5 Flash, which independently scores how visible
    the agricultural field boundary is.  This score is blended (30 % weight)
    into the classical confidence, providing a genuinely independent second
    signal without touching the example truths.

4. **Confidence calibration** — :class:`ConfidenceCalibrator` applies the
   area-ratio pre-filter, rank-normalises confidence scores, and flags the
   bottom ``flag_threshold`` fraction.

5. **Geometry reconstruction** — shifts are applied in UTM (metres), then
   converted back to EPSG:4326 for the output GeoDataFrame.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable
from typing import Optional

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from shapely.affinity import translate
from shapely.ops import transform as shp_transform

from bhume.baseline import global_median_shift
from bhume.io import Village
from solution.align import BoundaryAligner, _utm_for, _reproject, DENSITY_FULL_CREDIT
from solution.calibrate import ConfidenceCalibrator
from solution.gemini_vision import GeminiVisionAnalyzer
from solution.types import AlignmentResult

log = logging.getLogger(__name__)

# ── module-level constants ─────────────────────────────────────────────────────

#: Number of nearest neighbours used to compute the local median shift for the
#: neighbourhood consistency check.
_CONSISTENCY_N_NEIGHBOURS: int = 20

#: Shifts within this distance (metres) of the local median receive no penalty.
_CONSISTENCY_TOLERANCE_M: float = 3.0

#: Characteristic distance (metres) for the consistency exponential decay.
#: A shift CONSISTENCY_TOLERANCE_M + CONSISTENCY_DECAY_M from the local median
#: receives a penalty of exp(−1) ≈ 37 %.
_CONSISTENCY_DECAY_M: float = 8.0

#: Minimum number of results required before neighbourhood consistency is
#: applied.  Below this the village is too small for meaningful local statistics.
_MIN_RESULTS_FOR_CONSISTENCY: int = 5

#: Log a progress message every this many plots during alignment.
_PROGRESS_LOG_INTERVAL: int = 250

#: Approximate metres per degree of latitude (WGS-84 mean Earth radius × π / 180).
_M_PER_DEG_LAT: float = 111_320.0

#: Decimal places used when rounding calibrated_confidence in the output GeoDataFrame.
_CONFIDENCE_DECIMAL_PLACES: int = 4


class Predictor:
    """Runs the full correction pipeline for a village.

    Parameters
    ----------
    search_radius_m:
        Maximum translation search window passed to :class:`BoundaryAligner`.
        The actual per-plot radius is capped at 30% of sqrt(area_sqm), so
        tiny plots cannot be over-shifted.
    flag_threshold:
        Fraction of low-confidence plots to flag (passed to
        :class:`ConfidenceCalibrator`).
    """

    def __init__(
        self,
        search_radius_m:   float = 17.0,
        flag_threshold:    float = 0.20,
        gemini_api_key:    Optional[str] = None,
        gemini_max_plots:  int = 300,
    ) -> None:
        self._search_radius_m  = search_radius_m
        self._flag_threshold   = flag_threshold
        self._gemini_api_key   = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        self._gemini_max_plots = gemini_max_plots

    # ── public API ──────────────────────────────────────────────────────────────

    def predict(
        self,
        village:     Village,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> gpd.GeoDataFrame:
        """Correct all plots in a village and return a predictions GeoDataFrame.

        Parameters
        ----------
        village:
            Loaded :class:`~bhume.io.Village` bundle.
        progress_cb:
            Optional ``(done, total)`` callable for streaming progress
            (WebSocket, progress bar, etc.).

        Returns
        -------
        :class:`~geopandas.GeoDataFrame`
            Columns: ``plot_number``, ``status``, ``confidence``,
            ``method_note``, ``geometry``.  CRS EPSG:4326.
        """
        plots = village.plots

        # ── stage 1: global shift ──────────────────────────────────────────
        if village.example_truths is not None and len(village.example_truths) > 0:
            log.info(
                "computing global median shift from %d example truths",
                len(village.example_truths),
            )
            global_preds = global_median_shift(village)
            global_note  = global_preds["method_note"].iloc[0]
        else:
            log.info("no example truths — using zero global shift")
            global_preds = plots.copy()
            global_preds["status"]      = "corrected"
            global_preds["confidence"]  = 0.5
            global_preds["method_note"] = "zero_global_shift (no example_truths)"
            global_note = "zero_global_shift"

        # ── stage 2: per-plot alignment ────────────────────────────────────
        if village.boundaries_path is None:
            log.warning(
                "boundaries.tif not found — falling back to global shift for all plots"
            )
            results = self._zero_confidence_results(plots, global_preds, global_note)
        else:
            results = self._run_alignment(village, plots, global_preds, progress_cb)

        # ── stage 3: neighbourhood consistency ────────────────────────────
        results = self._neighborhood_consistency(results, plots)

        # ── stage 3b: Gemini vision pass (optional) ────────────────────────
        if self._gemini_api_key and village.imagery_path:
            vision = GeminiVisionAnalyzer(
                api_key=self._gemini_api_key,
                imagery_path=village.imagery_path,
                max_plots=self._gemini_max_plots,
            )
            if vision.enabled:
                results = vision.analyse_uncertain_plots(results, plots)

        # ── stage 4: confidence calibration ───────────────────────────────
        calibrator = ConfidenceCalibrator(flag_threshold=self._flag_threshold)
        calibrated = calibrator.calibrate(results, plots)

        # ── stage 5: geometry reconstruction + GeoDataFrame ───────────────
        return self._build_geodataframe(calibrated, plots)

    # ── private helpers ────────────────────────────────────────────────────────

    def _run_alignment(
        self,
        village:     Village,
        plots:       gpd.GeoDataFrame,
        global_preds: gpd.GeoDataFrame,
        progress_cb: Optional[Callable[[int, int], None]],
    ) -> list[AlignmentResult]:
        results: list[AlignmentResult] = []
        total = len(plots)

        with rasterio.open(village.boundaries_path) as bsrc:
            aligner = BoundaryAligner(bsrc, search_radius_m=self._search_radius_m)

            for done, pn in enumerate(plots.index, start=1):
                official_geom = plots.loc[pn, "geometry"]
                shifted_geom  = (
                    global_preds.loc[pn, "geometry"]
                    if pn in global_preds.index
                    else official_geom
                )
                # Pass the drawn map area so the aligner can adapt its search radius
                area_sqm = float(plots.loc[pn].get("map_area_sqm") or 0) or None

                result = aligner.align(
                    plot_number=str(pn),
                    official_geom=official_geom,
                    globally_shifted_geom=shifted_geom,
                    plot_area_sqm=area_sqm,
                )
                results.append(result)

                if progress_cb is not None:
                    progress_cb(done, total)

                if done % _PROGRESS_LOG_INTERVAL == 0 or done == total:
                    log.info("  aligned %d / %d plots", done, total)

        return results

    @staticmethod
    def _neighborhood_consistency(
        results: list[AlignmentResult],
        plots:   gpd.GeoDataFrame,
    ) -> list[AlignmentResult]:
        """Scale confidence by each plot's agreement with its nearest neighbours.

        Plots whose estimated shift is an outlier relative to their local
        neighbourhood receive a penalty:

            consistency_factor = exp(−max(0, outlier_dist − tolerance) / decay)

        where ``outlier_dist`` is the distance between the plot's shift and the
        *median shift of its 20 nearest neighbours* in UTM metres.

        This catches the scenario where a plot is moved 25 m while its neighbours
        are all shifted ~10 m — a strong signal that the cross-correlation landed
        on a wrong edge (building wall, road, etc.).
        """
        if len(results) < _MIN_RESULTS_FOR_CONSISTENCY:
            return results

        # Build arrays of centroids and shifts
        pn_to_idx = {r.plot_number: i for i, r in enumerate(results)}
        lons, lats, dxs, dys = [], [], [], []

        # Use a representative CRS for distance — approximate with degrees is OK
        # for neighbour-search (we only need relative ordering)
        for r in results:
            pn = r.plot_number
            try:
                geom = plots.loc[pn, "geometry"]
                lons.append(float(geom.centroid.x))
                lats.append(float(geom.centroid.y))
            except (KeyError, AttributeError):
                # fallback: use mean of all if geometry missing
                lons.append(np.nan)
                lats.append(np.nan)
            dxs.append(r.dx_m)
            dys.append(r.dy_m)

        lons_arr = np.array(lons)
        lats_arr = np.array(lats)
        dxs_arr  = np.array(dxs)
        dys_arr  = np.array(dys)

        # Approximate metres-per-degree at the village centroid latitude.
        # Uses _M_PER_DEG_LAT (standard WGS-84 approximation) — sufficient for
        # neighbour-distance ranking; not used for final shift calculations.
        lat_mean = np.nanmean(lats_arr)
        m_per_deg_lat = _M_PER_DEG_LAT
        m_per_deg_lon = _M_PER_DEG_LAT * math.cos(math.radians(lat_mean))

        updated: list[AlignmentResult] = []
        k = _CONSISTENCY_N_NEIGHBOURS

        for i, r in enumerate(results):
            if math.isnan(lons_arr[i]):
                updated.append(r)
                continue

            # Compute distances to all other plots (vectorised)
            dlat_m = (lats_arr - lats_arr[i]) * m_per_deg_lat
            dlon_m = (lons_arr - lons_arr[i]) * m_per_deg_lon
            dists  = np.sqrt(dlat_m ** 2 + dlon_m ** 2)

            # K nearest (excluding self)
            dists[i] = np.inf
            nn_idx = np.argpartition(dists, min(k, len(dists) - 1))[:k]

            nb_dx  = np.median(dxs_arr[nn_idx])
            nb_dy  = np.median(dys_arr[nn_idx])

            outlier_dist = math.sqrt(
                (r.dx_m - nb_dx) ** 2 + (r.dy_m - nb_dy) ** 2
            )

            # Only penalise outlier shifts in LOW-DENSITY areas.
            # High boundary density means the cross-correlation found a real
            # field edge — trust the result even if the shift looks unusual.
            # Low density means we may have locked onto a false edge (building
            # wall, road) — outlier shifts are suspicious there.
            if r.boundary_density >= DENSITY_FULL_CREDIT:
                consistency = 1.0
            else:
                excess = max(0.0, outlier_dist - _CONSISTENCY_TOLERANCE_M)
                consistency = math.exp(-excess / _CONSISTENCY_DECAY_M)

            # Composite raw_confidence = original × consistency_factor
            new_raw = float(np.clip(r.raw_confidence * consistency, 0.0, 1.0))

            updated.append(AlignmentResult(
                plot_number=r.plot_number,
                dx_m=r.dx_m,
                dy_m=r.dy_m,
                raw_confidence=new_raw,
                calibrated_confidence=new_raw,
                status=r.status,
                method_note=r.method_note,
                boundary_density=r.boundary_density,
                consistency_factor=consistency,
            ))

        return updated

    @staticmethod
    def _zero_confidence_results(
        plots:       gpd.GeoDataFrame,
        global_preds: gpd.GeoDataFrame,
        note:        str,
    ) -> list[AlignmentResult]:
        """Global-shift results with confidence=0 (no boundary raster available)."""
        results = []
        for pn in plots.index:
            official_geom = plots.loc[pn, "geometry"]
            shifted_geom  = (
                global_preds.loc[pn, "geometry"]
                if pn in global_preds.index
                else official_geom
            )
            try:
                utm_crs = _utm_for(official_geom)
                to_utm  = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
                off_u   = _reproject(official_geom, to_utm)
                sft_u   = _reproject(shifted_geom,  to_utm)
                dx = sft_u.centroid.x - off_u.centroid.x
                dy = sft_u.centroid.y - off_u.centroid.y
            except Exception:
                dx, dy = 0.0, 0.0
            results.append(AlignmentResult(
                plot_number=str(pn),
                dx_m=dx, dy_m=dy,
                raw_confidence=0.0, calibrated_confidence=0.0,
                status="corrected",
                method_note=f"no_boundaries_raster {note}",
            ))
        return results

    @staticmethod
    def _build_geodataframe(
        calibrated: list[AlignmentResult],
        plots:      gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """Apply UTM shifts to official geometries and assemble the output GDF."""
        rows = []
        for r in calibrated:
            pn = r.plot_number
            if pn not in plots.index:
                log.warning("plot %s missing from plots index — skipping", pn)
                continue

            official_geom = plots.loc[pn, "geometry"]

            if r.status == "flagged":
                rows.append({
                    "plot_number": pn,
                    "status":      "flagged",
                    "confidence":  None,
                    "method_note": r.method_note,
                    "geometry":    official_geom,
                })
            else:
                try:
                    utm_crs = _utm_for(official_geom)
                    to_utm  = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
                    frm_utm = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

                    off_u  = shp_transform(
                        lambda xs, ys, zs=None: to_utm.transform(xs, ys),
                        official_geom,
                    )
                    corr_u = translate(off_u, r.dx_m, r.dy_m)
                    corrected_4326 = shp_transform(
                        lambda xs, ys, zs=None: frm_utm.transform(xs, ys),
                        corr_u,
                    )
                except Exception as exc:
                    log.warning(
                        "geometry shift failed for plot %s: %s — flagging", pn, exc
                    )
                    rows.append({
                        "plot_number": pn,
                        "status":      "flagged",
                        "confidence":  None,
                        "method_note": f"geometry_error {r.method_note}",
                        "geometry":    official_geom,
                    })
                    continue

                rows.append({
                    "plot_number": pn,
                    "status":      "corrected",
                    "confidence":  round(r.calibrated_confidence, _CONFIDENCE_DECIMAL_PLACES),
                    "method_note": r.method_note,
                    "geometry":    corrected_4326,
                })

        gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
        gdf["plot_number"] = gdf["plot_number"].astype(str)
        return gdf.set_index("plot_number", drop=False)
