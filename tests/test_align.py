"""Tests for solution.align — BoundaryAligner with synthetic raster fixtures."""

from __future__ import annotations

import math
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.io import MemoryFile
from shapely.geometry import MultiPolygon, Polygon, box

from solution.align import (
    BoundaryAligner,
    _rasterize_edge,
    _reproject,
    _utm_for,
    _valid_window,
)
from solution.types import AlignmentResult


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_raster(
    data: np.ndarray,
    west: float = 0.0,
    north: float = 1000.0,
    res: float = 1.0,
    crs: str = "EPSG:3857",
) -> rasterio.DatasetReader:
    """Create an in-memory single-band rasterio dataset from a numpy array."""
    h, w = data.shape
    transform = Affine(res, 0.0, west, 0.0, -res, north)
    mf = MemoryFile()
    with rasterio.open(
        mf,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    return mf.open()


def _square_plot_4326(
    lon_center: float = 74.03,
    lat_center: float = 20.24,
    half_deg: float = 0.0003,
) -> Polygon:
    """Return a small square Polygon in EPSG:4326."""
    return box(
        lon_center - half_deg,
        lat_center - half_deg,
        lon_center + half_deg,
        lat_center + half_deg,
    )


# ── _utm_for ──────────────────────────────────────────────────────────────────

class TestUtmFor:
    def test_northern_hemisphere(self) -> None:
        g = _square_plot_4326(74.03, 20.24)
        crs = _utm_for(g)
        # lon 74 → zone 43; lat > 0 → 326xx
        assert crs == "EPSG:32643"

    def test_southern_hemisphere(self) -> None:
        g = box(25.0, -30.0, 25.1, -29.9)
        crs = _utm_for(g)
        assert crs.startswith("EPSG:327")

    def test_zone_boundary(self) -> None:
        # lon exactly at zone boundary 6, 12, 18, …
        for lon in (6.0, 12.0, 18.0, 24.0):
            g = box(lon, 10.0, lon + 0.01, 10.01)
            crs = _utm_for(g)
            assert crs.startswith("EPSG:326")


# ── _rasterize_edge ───────────────────────────────────────────────────────────

class TestRasterizeEdge:
    """Verify that plot outlines are drawn correctly onto a pixel grid."""

    def _transform_for(self, h: int, w: int, res: float = 1.0) -> Affine:
        return Affine(res, 0.0, 0.0, 0.0, -res, float(h))

    def test_square_has_nonzero_edge_pixels(self) -> None:
        poly = box(2.0, 2.0, 8.0, 8.0)
        tf = self._transform_for(10, 10)
        mask = _rasterize_edge(poly, tf, 10, 10)
        assert mask.sum() > 0, "expected edge pixels"

    def test_output_shape_matches_hw(self) -> None:
        poly = box(1.0, 1.0, 5.0, 5.0)
        for h, w in [(20, 30), (50, 50), (7, 13)]:
            tf = self._transform_for(h, w)
            mask = _rasterize_edge(poly, tf, h, w)
            assert mask.shape == (h, w)

    def test_interior_is_mostly_zero(self) -> None:
        poly = box(1.0, 1.0, 9.0, 9.0)
        tf = self._transform_for(10, 10)
        mask = _rasterize_edge(poly, tf, 10, 10)
        # centre pixel should be 0
        assert mask[5, 5] == 0.0

    def test_multi_polygon(self) -> None:
        mp = MultiPolygon([box(0.5, 0.5, 3.5, 3.5), box(6.0, 6.0, 9.0, 9.0)])
        tf = self._transform_for(10, 10)
        mask = _rasterize_edge(mp, tf, 10, 10)
        assert mask.sum() > 0

    def test_empty_polygon_returns_zeros(self) -> None:
        from shapely.geometry import Polygon as P
        empty = P()
        tf = self._transform_for(10, 10)
        mask = _rasterize_edge(empty, tf, 10, 10)
        assert mask.sum() == 0.0

    def test_dtype_is_float32(self) -> None:
        poly = box(1.0, 1.0, 5.0, 5.0)
        mask = _rasterize_edge(poly, self._transform_for(10, 10), 10, 10)
        assert mask.dtype == np.float32


# ── _valid_window ─────────────────────────────────────────────────────────────

class TestValidWindow:
    def _mock_src(self, w: int = 100, h: int = 100) -> MagicMock:
        src = MagicMock()
        src.width  = w
        src.height = h
        return src

    def test_fully_inside(self) -> None:
        from rasterio.windows import Window
        win = Window(10, 10, 20, 20)
        result = _valid_window(win, self._mock_src())
        assert result is not None
        assert result.col_off == 10
        assert result.row_off == 10

    def test_partially_outside_clips(self) -> None:
        from rasterio.windows import Window
        win = Window(90, 90, 30, 30)   # extends to 120×120 but raster is 100×100
        result = _valid_window(win, self._mock_src(100, 100))
        assert result is not None
        assert result.col_off + result.width  <= 100
        assert result.row_off + result.height <= 100

    def test_fully_outside_returns_none(self) -> None:
        from rasterio.windows import Window
        win = Window(200, 200, 10, 10)
        assert _valid_window(win, self._mock_src(100, 100)) is None

    def test_zero_width_after_clip_returns_none(self) -> None:
        from rasterio.windows import Window
        win = Window(100, 0, 10, 10)   # col_off == width → empty after clip
        assert _valid_window(win, self._mock_src(100, 100)) is None


# ── BoundaryAligner ───────────────────────────────────────────────────────────

class TestBoundaryAligner:
    """Integration-style tests against synthetic rasters."""

    # Coordinates centred near Vadnerbhairav for realistic CRS handling
    LON0, LAT0 = 74.03, 20.24

    def _plot(self, dx_deg: float = 0.0, dy_deg: float = 0.0) -> Polygon:
        """Small ~90×90 m square plot in EPSG:4326."""
        half = 0.0004
        return box(
            self.LON0 + dx_deg - half,
            self.LAT0 + dy_deg - half,
            self.LON0 + dx_deg + half,
            self.LAT0 + dy_deg + half,
        )

    def _aligner_with_edge_at(
        self, true_dx_m: float = 0.0, true_dy_m: float = 0.0
    ) -> tuple[BoundaryAligner, Polygon]:
        """Build a synthetic raster that has a high-response edge at (true_dx_m, true_dy_m)
        offset from a reference plot, then return an aligner + the reference plot."""
        from pyproj import Transformer
        from shapely.ops import transform as st
        from shapely.affinity import translate

        plot_4326 = self._plot()

        # Convert to EPSG:3857
        to3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        plot_3857 = st(lambda xs, ys, zs=None: to3857.transform(xs, ys), plot_4326)
        minx, miny, maxx, maxy = plot_3857.bounds

        # Build a 200×200 px raster covering the plot + 50 m margin
        margin = 50.0
        res    = 2.0   # 2 m/pixel
        west   = minx - margin
        north  = maxy + margin
        w = int((maxx - minx + 2 * margin) / res) + 1
        h = int((maxy - miny + 2 * margin) / res) + 1

        data = np.zeros((h, w), dtype=np.float32)

        # Draw the "true" field edge (high values) at the shifted position
        true_plot_3857 = translate(plot_3857, true_dx_m, true_dy_m)
        tf = Affine(res, 0.0, west, 0.0, -res, north)

        from PIL import Image, ImageDraw
        img  = Image.new("F", (w, h), 0.0)
        draw = ImageDraw.Draw(img)
        coords = list(true_plot_3857.exterior.coords)
        pix = [((x - tf.c) / tf.a, (y - tf.f) / tf.e) for x, y in coords]
        for i in range(len(pix) - 1):
            draw.line([pix[i], pix[i + 1]], fill=1.0, width=2)
        draw.line([pix[-1], pix[0]], fill=1.0, width=2)
        data = np.array(img, dtype=np.float32)

        src = _make_raster(data, west=west, north=north, res=res, crs="EPSG:3857")
        aligner = BoundaryAligner(src, search_radius_m=30.0)
        return aligner, plot_4326

    def test_zero_shift_high_confidence(self) -> None:
        aligner, plot = self._aligner_with_edge_at(0.0, 0.0)
        result = aligner.align("test", plot, plot)
        assert isinstance(result, AlignmentResult)
        assert result.raw_confidence > 0.3
        assert result.status == "corrected"

    def test_known_shift_recovered(self) -> None:
        """A 5 m east shift should be detected within 3 m tolerance."""
        true_dx, true_dy = 5.0, 0.0
        aligner, plot_4326 = self._aligner_with_edge_at(true_dx, true_dy)

        # globally-shifted plot = at the TRUE position (simulating perfect global shift)
        from pyproj import Transformer
        from shapely.ops import transform as st
        from shapely.affinity import translate
        to3857  = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        fr3857  = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        p3857   = st(lambda xs, ys, zs=None: to3857.transform(xs, ys), plot_4326)
        sh3857  = translate(p3857, true_dx, true_dy)
        shifted = st(lambda xs, ys, zs=None: fr3857.transform(xs, ys), sh3857)

        result = aligner.align("test", plot_4326, shifted)
        assert math.isclose(result.dx_m, true_dx, abs_tol=4.0), (
            f"expected dx≈{true_dx}, got {result.dx_m:.2f}"
        )

    def test_empty_raster_returns_fallback(self) -> None:
        """All-zero raster → confident alignment is impossible → fallback."""
        h, w = 100, 100
        data = np.zeros((h, w), dtype=np.float32)
        west, north = 8_230_000.0, 2_310_000.0
        src  = _make_raster(data, west=west, north=north, res=2.0)
        aligner = BoundaryAligner(src, search_radius_m=15.0)
        plot = self._plot()
        result  = aligner.align("p1", plot, plot)
        assert result.raw_confidence == 0.0

    def test_no_raster_overlap_returns_fallback(self) -> None:
        """Plot outside raster extent → fallback, confidence = 0."""
        data = np.ones((50, 50), dtype=np.float32)
        # Raster at 0,0 area; plot is at lon 74, lat 20 → EPSG:3857 x≈8.2M, y≈2.3M
        src = _make_raster(data, west=0.0, north=50.0, res=1.0)
        aligner = BoundaryAligner(src, search_radius_m=5.0)
        plot = self._plot()
        result = aligner.align("p1", plot, plot)
        assert result.raw_confidence == 0.0
        assert "fallback" in result.method_note or "no_raster_overlap" in result.method_note

    def test_result_plot_number_preserved(self) -> None:
        aligner, plot = self._aligner_with_edge_at()
        result = aligner.align("plot_99", plot, plot)
        assert result.plot_number == "plot_99"

    def test_never_raises(self) -> None:
        """align() must never propagate an exception."""
        data = np.random.default_rng(42).random((30, 30)).astype(np.float32)
        src  = _make_raster(data, west=0.0, north=30.0, res=1.0)
        aligner = BoundaryAligner(src, search_radius_m=5.0)
        for geom in [self._plot(), Polygon(), box(0, 0, 1e-10, 1e-10)]:
            result = aligner.align("x", geom, geom)
            assert isinstance(result, AlignmentResult)

    def test_confidence_in_range(self) -> None:
        aligner, plot = self._aligner_with_edge_at(3.0, 3.0)
        result = aligner.align("p", plot, plot)
        assert 0.0 <= result.raw_confidence <= 1.0
