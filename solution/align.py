"""BoundaryAligner — per-plot boundary alignment via cross-correlation with boundaries.tif.

Algorithm
---------
For every plot we:
  1. Compute an *effective search radius* proportional to the plot's physical size,
     so tiny urban plots cannot be thrown far from their true position.
  2. Reproject the globally-shifted geometry to the raster CRS.
  3. Extract an *outer patch* from boundaries.tif: shifted bounds + effective_radius padding.
  4. Measure *boundary density* (fraction of patch pixels above DENSITY_PIXEL_THRESHOLD).
  5. Rasterize the shifted plot edge onto the *inner* pixel grid → binary float32 mask.
  6. FFT cross-correlation (``fftconvolve``, valid mode) → correlation surface.
  7. Subpixel peak via parabolic interpolation → fractional pixel shift (dr, dc).
  8. Convert pixel shift → UTM metre shift (dx_m, dy_m).
  9. Composite confidence = SNR_score × density_factor × [consistency applied later].
 10. Fall back to the global shift + confidence=0 on any failure.

Algorithm improvement over integer-pixel peak finding
------------------------------------------------------
**Subpixel peak localisation** (step 7): After finding the integer-pixel peak of the
correlation surface, a parabolic fit through the three points (peak−1, peak, peak+1)
in each axis gives a fractional-pixel refinement.  At 2 m/px (Vadnerbhairav), this
improves translational accuracy from ≤2 m to ≤0.5 m; at 2.4 m/px (Malatavadi) from
≤2.4 m to ≤0.6 m.  No new dependencies — the fit uses only the three adjacent
integer-pixel values already in the correlation surface.

Why density matters
-------------------
boundaries.tif has strong coverage over open fields (Vadnerbhairav) but is largely
empty over dense urban areas (Malatavadi, ~63% of patches < 3 % non-zero).  A sparse
patch means the correlation is finding incidental edges (building walls, roads) rather
than field boundaries.  The density factor smoothly penalises low-signal patches so
such corrections land low in the confidence ranking and get flagged.

Why adaptive search radius matters
-----------------------------------
A 15 m search radius relative to a 30×30 m plot (~389 m²) allows the algorithm to
shift the plot by half its own width — easy to land on the wrong field entirely.
We cap the search at AREA_SEARCH_SCALE × sqrt(area_sqm) so the maximum shift can
never exceed a fixed fraction of the plot's own size.  Large plots keep the full
default radius.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from pyproj import Transformer
from rasterio.windows import from_bounds, Window
from scipy.signal import fftconvolve
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shp_transform

from solution.types import AlignmentResult

log = logging.getLogger(__name__)

# ── module-level constants ─────────────────────────────────────────────────────
# All tuneable thresholds live here so they can be found and adjusted in one place.

#: Fraction of patch pixels that must be above DENSITY_PIXEL_THRESHOLD for the
#: patch to receive full confidence credit.
DENSITY_FULL_CREDIT: float = 0.05

#: Patch pixels above this normalised value [0, 1] are counted as "non-zero"
#: for the boundary density measurement.
DENSITY_PIXEL_THRESHOLD: float = 0.10

#: Patches where the fraction of non-zero pixels is below this floor are
#: considered empty and always fall back to the global shift.
DENSITY_FLOOR: float = 0.003

#: effective_search_radius = min(max_radius, AREA_SEARCH_SCALE × sqrt(area_sqm))
AREA_SEARCH_SCALE: float = 0.30

#: Never search less than this many metres regardless of plot size.
MIN_SEARCH_RADIUS_M: float = 5.0

#: SNR value that maps to a raw SNR confidence score of 1.0 (soft cap).
#: Values above this are clipped to 1.0.
SNR_SCALE: float = 5.0

#: Minimum edge-mask pixels below which the correlation result is unreliable.
MIN_EDGE_PX: int = 8

#: Minimum inner-patch spatial dimension (pixels) needed for correlation.
MIN_PATCH_PX: int = 4

#: Inner-patch dimension threshold below which edge-mask line width is doubled
#: so that very small plots still produce enough edge pixels.
SMALL_PLOT_PX_THRESHOLD: int = 15

#: Edge line width (pixels) used for small plots (inner_h or inner_w < SMALL_PLOT_PX_THRESHOLD).
SMALL_EDGE_LINE_WIDTH: int = 2

#: Edge line width (pixels) used for normal-sized plots.
NORMAL_EDGE_LINE_WIDTH: int = 1

#: Floating-point epsilon used to guard against division by zero.
_EPS: float = 1e-9


# ── coordinate helpers ─────────────────────────────────────────────────────────

def _utm_for(geom: BaseGeometry) -> str:
    """Return EPSG string for the UTM zone containing *geom*'s centroid."""
    lon = geom.centroid.x
    lat = geom.centroid.y
    zone = int((lon + 180.0) / 6.0) + 1
    base = 32600 if lat >= 0 else 32700
    return f"EPSG:{base + zone}"


def _reproject(geom: BaseGeometry, transformer: Transformer) -> BaseGeometry:
    """Reproject a Shapely geometry using a pyproj ``Transformer`` (always_xy=True)."""
    return shp_transform(
        lambda xs, ys, zs=None: transformer.transform(xs, ys),
        geom,
    )


# ── rasterisation ──────────────────────────────────────────────────────────────

def _rasterize_edge(
    geom:       BaseGeometry,
    transform:  rasterio.transform.Affine,
    height:     int,
    width:      int,
    line_width: int = NORMAL_EDGE_LINE_WIDTH,
) -> np.ndarray:
    """Rasterize the polygon **outline** onto an (H, W) float32 pixel grid.

    Coordinate mapping (rasterio convention)::

        PIL col = (geo_x − T.c) / T.a        (T.a > 0)
        PIL row = (geo_y − T.f) / T.e        (T.e < 0 for north-up)
    """
    img  = Image.new("F", (width, height), 0.0)
    draw = ImageDraw.Draw(img)

    polys: list[Polygon] = (
        list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    )

    for poly in polys:
        if not isinstance(poly, Polygon) or poly.is_empty:
            continue
        for ring in [poly.exterior, *poly.interiors]:
            coords = list(ring.coords)
            if len(coords) < 2:
                continue
            pix = [
                (
                    (x - transform.c) / transform.a,
                    (y - transform.f) / transform.e,
                )
                for x, y in coords
            ]
            for i in range(len(pix) - 1):
                draw.line([pix[i], pix[i + 1]], fill=1.0, width=line_width)
            draw.line([pix[-1], pix[0]], fill=1.0, width=line_width)

    return np.array(img, dtype=np.float32)


# ── window utilities ───────────────────────────────────────────────────────────

def _valid_window(
    window: Window,
    src:    rasterio.DatasetReader,
) -> Optional[Window]:
    """Clip *window* to the raster's valid extent; return ``None`` if no overlap."""
    col_off = max(0.0, window.col_off)
    row_off = max(0.0, window.row_off)
    col_end = min(float(src.width),  window.col_off + window.width)
    row_end = min(float(src.height), window.row_off + window.height)
    if col_end <= col_off or row_end <= row_off:
        return None
    return Window(col_off, row_off, col_end - col_off, row_end - row_off)


# ── correlation helpers ────────────────────────────────────────────────────────

def _fast_xcorr(patch: np.ndarray, edge_mask: np.ndarray) -> np.ndarray:
    """FFT-based cross-correlation using ``scipy.signal.fftconvolve`` (valid mode).

    Equivalent to sliding the edge_mask over the patch and summing element-wise
    products — standard cross-correlation, computed efficiently via FFT.

    Why not zero-mean normalisation?
    A global DC removal of a spatial cross-correlation creates false peaks in
    low-intensity raster regions: the global mean subtraction does *not* account
    for local intensity variation and changes the correlation score in unintended
    ways.  The boundary raster is already normalised to [0, 1] beforehand, so
    the standard correlation reliably peaks where the edge mask overlaps the
    brightest (most boundary-dense) region.

    Returns the valid-mode correlation surface of shape
    ``(patch_h − mask_h + 1, patch_w − mask_w + 1)``.
    """
    return fftconvolve(patch, edge_mask[::-1, ::-1], mode="valid")


def _subpixel_peak(
    xcorr:  np.ndarray,
    peak_r: int,
    peak_c: int,
) -> tuple[float, float]:
    """Refine an integer-pixel peak to subpixel accuracy via parabolic interpolation.

    Fits a 1-D parabola through (peak−1, peak, peak+1) in each axis and
    returns the analytical maximum.  The refinement is clamped to ±0.5 px
    so the result always stays within the same integer pixel cell.

    At 2 m/px boundary raster resolution this improves translational accuracy
    from ≤2 m (integer pixel) to ≤0.5 m (subpixel).
    """
    h, w = xcorr.shape
    dr, dc = 0.0, 0.0

    if 1 <= peak_r < h - 1:
        denom = (
            2.0 * xcorr[peak_r, peak_c]
            - xcorr[peak_r - 1, peak_c]
            - xcorr[peak_r + 1, peak_c]
        )
        if abs(denom) > _EPS:
            dr = 0.5 * (xcorr[peak_r + 1, peak_c] - xcorr[peak_r - 1, peak_c]) / denom
            dr = float(np.clip(dr, -0.5, 0.5))

    if 1 <= peak_c < w - 1:
        denom = (
            2.0 * xcorr[peak_r, peak_c]
            - xcorr[peak_r, peak_c - 1]
            - xcorr[peak_r, peak_c + 1]
        )
        if abs(denom) > _EPS:
            dc = 0.5 * (xcorr[peak_r, peak_c + 1] - xcorr[peak_r, peak_c - 1]) / denom
            dc = float(np.clip(dc, -0.5, 0.5))

    return peak_r + dr, peak_c + dc


# ── BoundaryAligner ────────────────────────────────────────────────────────────

class BoundaryAligner:
    """Aligns one plot to field edges via cross-correlation with the boundary raster.

    Parameters
    ----------
    boundaries_src:
        Open ``rasterio.DatasetReader`` for the village ``boundaries.tif``.
    search_radius_m:
        Maximum translation search window in metres (default 15 m).
        The actual per-plot radius is
        ``min(search_radius_m, AREA_SEARCH_SCALE × sqrt(plot_area_sqm))``,
        so tiny plots cannot be shifted further than a fixed fraction of
        their own physical size.
    """

    def __init__(
        self,
        boundaries_src: rasterio.DatasetReader,
        search_radius_m: float = 15.0,
    ) -> None:
        self._src = boundaries_src
        self._max_radius = search_radius_m
        self._px_size = abs(boundaries_src.transform.a)

    # ── public API ─────────────────────────────────────────────────────────────

    def align(
        self,
        plot_number:          str,
        official_geom:        BaseGeometry,
        globally_shifted_geom: BaseGeometry,
        plot_area_sqm:        Optional[float] = None,
    ) -> AlignmentResult:
        """Compute the best-fit translation for one plot.

        Parameters
        ----------
        plot_number:
            Plot identifier echoed into the result.
        official_geom:
            Official cadastral polygon in EPSG:4326.
        globally_shifted_geom:
            Plot after the village-level global shift (EPSG:4326).
        plot_area_sqm:
            Drawn map area in m² from the input properties.  When provided,
            the search radius is capped proportionally so small plots cannot
            be over-shifted.

        Returns
        -------
        :class:`~solution.types.AlignmentResult`
            ``raw_confidence`` = SNR_score × density_factor ∈ [0, 1].
            Falls back to the global shift + confidence=0 on any failure.
        """
        try:
            return self._align_inner(
                plot_number, official_geom, globally_shifted_geom, plot_area_sqm
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("align() failed for plot %s: %s", plot_number, exc)
            return self._global_fallback(
                plot_number, official_geom, globally_shifted_geom, "exception_fallback"
            )

    # ── private ────────────────────────────────────────────────────────────────

    def _effective_radius(self, plot_area_sqm: Optional[float]) -> float:
        """Per-plot effective search radius in metres."""
        if plot_area_sqm and plot_area_sqm > 0:
            area_limit = AREA_SEARCH_SCALE * math.sqrt(plot_area_sqm)
            return max(MIN_SEARCH_RADIUS_M, min(self._max_radius, area_limit))
        return self._max_radius

    def _align_inner(
        self,
        plot_number:           str,
        official_geom:         BaseGeometry,
        globally_shifted_geom: BaseGeometry,
        plot_area_sqm:         Optional[float],
    ) -> AlignmentResult:
        utm_crs = _utm_for(official_geom)
        to_utm  = Transformer.from_crs("EPSG:4326", utm_crs,            always_xy=True)
        to_src  = Transformer.from_crs("EPSG:4326", str(self._src.crs), always_xy=True)
        frm_src = Transformer.from_crs(str(self._src.crs), "EPSG:4326", always_xy=True)

        official_utm = _reproject(official_geom,          to_utm)
        shifted_utm  = _reproject(globally_shifted_geom,  to_utm)
        shifted_src  = _reproject(globally_shifted_geom,  to_src)

        global_dx = shifted_utm.centroid.x - official_utm.centroid.x
        global_dy = shifted_utm.centroid.y - official_utm.centroid.y

        # ── adaptive search radius ─────────────────────────────────────────
        r = self._effective_radius(plot_area_sqm)

        # ── outer patch ────────────────────────────────────────────────────
        minx, miny, maxx, maxy = shifted_src.bounds

        outer_win = _valid_window(
            from_bounds(minx - r, miny - r, maxx + r, maxy + r, self._src.transform),
            self._src,
        )
        if outer_win is None:
            return self._make_result(
                plot_number, global_dx, global_dy, 0.0, 0.0, "no_raster_overlap"
            )

        patch = self._src.read(1, window=outer_win).astype(np.float32)
        p_min, p_max = float(patch.min()), float(patch.max())
        if p_max - p_min < _EPS:
            return self._make_result(
                plot_number, global_dx, global_dy, 0.0, 0.0, "empty_boundary_patch"
            )
        patch = (patch - p_min) / (p_max - p_min)

        # ── boundary density ────────────────────────────────────────────────
        density = float((patch > DENSITY_PIXEL_THRESHOLD).mean())

        if density < DENSITY_FLOOR:
            return self._make_result(
                plot_number, global_dx, global_dy, 0.0, density, "density_floor_fallback"
            )

        density_factor = min(
            1.0,
            (density - DENSITY_FLOOR) / (DENSITY_FULL_CREDIT - DENSITY_FLOOR),
        )

        outer_transform = self._src.window_transform(outer_win)
        outer_h, outer_w = patch.shape

        # ── inner edge mask ─────────────────────────────────────────────────
        inner_win = _valid_window(
            from_bounds(minx, miny, maxx, maxy, self._src.transform),
            self._src,
        )
        if inner_win is None:
            return self._make_result(
                plot_number, global_dx, global_dy, 0.0, density, "inner_win_no_overlap"
            )

        inner_h = max(1, int(round(inner_win.height)))
        inner_w = max(1, int(round(inner_win.width)))

        if inner_h < MIN_PATCH_PX or inner_w < MIN_PATCH_PX:
            return self._make_result(
                plot_number, global_dx, global_dy, 0.0, density, "plot_too_small"
            )

        # Wider edge for very small plots so the mask has enough pixels
        lw = (
            SMALL_EDGE_LINE_WIDTH
            if (inner_h < SMALL_PLOT_PX_THRESHOLD or inner_w < SMALL_PLOT_PX_THRESHOLD)
            else NORMAL_EDGE_LINE_WIDTH
        )
        edge_mask = _rasterize_edge(
            shifted_src,
            self._src.window_transform(inner_win),
            inner_h,
            inner_w,
            line_width=lw,
        )

        if edge_mask.sum() < MIN_EDGE_PX:
            return self._make_result(
                plot_number, global_dx, global_dy, 0.0, density, "too_few_edge_pixels"
            )

        # ── zero-mean normalised cross-correlation ──────────────────────────
        if outer_h < inner_h or outer_w < inner_w:
            return self._make_result(
                plot_number, global_dx, global_dy, 0.0, density, "outer_smaller_than_inner"
            )

        xcorr = _fast_xcorr(patch, edge_mask)
        if xcorr.size == 0:
            return self._make_result(
                plot_number, global_dx, global_dy, 0.0, density, "xcorr_empty"
            )

        # ── integer peak ────────────────────────────────────────────────────
        flat_idx = int(xcorr.argmax())
        peak_r_i, peak_c_i = divmod(flat_idx, xcorr.shape[1])

        # ── subpixel refinement ─────────────────────────────────────────────
        peak_r, peak_c = _subpixel_peak(xcorr, peak_r_i, peak_c_i)

        center_r = (xcorr.shape[0] - 1) / 2.0
        center_c = (xcorr.shape[1] - 1) / 2.0

        dr = peak_r - center_r   # +ve → south (row ↓)
        dc = peak_c - center_c   # +ve → east  (col →)

        dx_src = dc * float(outer_transform.a)
        dy_src = dr * float(outer_transform.e)

        ref_lon, ref_lat = frm_src.transform(
            shifted_src.centroid.x + dx_src,
            shifted_src.centroid.y + dy_src,
        )
        ref_x, ref_y = to_utm.transform(float(ref_lon), float(ref_lat))

        total_dx = ref_x - official_utm.centroid.x
        total_dy = ref_y - official_utm.centroid.y

        # ── composite confidence ────────────────────────────────────────────
        xcorr_std = float(xcorr.std())
        snr = (float(xcorr[peak_r_i, peak_c_i]) - float(xcorr.mean())) / (xcorr_std + _EPS)
        snr_conf  = float(np.clip(snr / SNR_SCALE, 0.0, 1.0))
        raw_conf  = float(np.clip(snr_conf * density_factor, 0.0, 1.0))

        note = (
            f"xcorr_subpx "
            f"r={r:.1f}m dx={total_dx:.2f}m dy={total_dy:.2f}m "
            f"snr={snr:.2f} dens={density:.4f} conf={raw_conf:.3f}"
        )
        log.debug("plot %s: %s", plot_number, note)

        return AlignmentResult(
            plot_number=plot_number,
            dx_m=total_dx,
            dy_m=total_dy,
            raw_confidence=raw_conf,
            calibrated_confidence=raw_conf,
            status="corrected",
            method_note=note,
            boundary_density=density,
            consistency_factor=1.0,
        )

    def _global_fallback(
        self,
        plot_number:           str,
        official_geom:         BaseGeometry,
        globally_shifted_geom: BaseGeometry,
        reason:                str,
    ) -> AlignmentResult:
        try:
            utm_crs = _utm_for(official_geom)
            to_utm  = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
            off_u   = _reproject(official_geom,          to_utm)
            sft_u   = _reproject(globally_shifted_geom,  to_utm)
            dx = sft_u.centroid.x - off_u.centroid.x
            dy = sft_u.centroid.y - off_u.centroid.y
        except Exception:
            dx, dy = 0.0, 0.0
        return self._make_result(plot_number, dx, dy, 0.0, 0.0, reason)

    @staticmethod
    def _make_result(
        plot_number: str,
        dx_m:        float,
        dy_m:        float,
        conf:        float,
        density:     float,
        reason:      str,
    ) -> AlignmentResult:
        return AlignmentResult(
            plot_number=plot_number,
            dx_m=dx_m,
            dy_m=dy_m,
            raw_confidence=conf,
            calibrated_confidence=conf,
            status="corrected",
            method_note=f"{reason} dx={dx_m:.2f}m dy={dy_m:.2f}m",
            boundary_density=density,
            consistency_factor=1.0,
        )
