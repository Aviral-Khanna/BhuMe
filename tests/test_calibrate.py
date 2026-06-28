"""Tests for solution.calibrate — ConfidenceCalibrator."""

from __future__ import annotations

import pytest
import geopandas as gpd
from shapely.geometry import box

from solution.calibrate import ConfidenceCalibrator, _CONF_LO, _CONF_HI
from solution.types import AlignmentResult


# ── fixtures ──────────────────────────────────────────────────────────────────

def _result(pn: str, conf: float) -> AlignmentResult:
    return AlignmentResult(
        plot_number=pn, dx_m=0.0, dy_m=0.0,
        raw_confidence=conf, calibrated_confidence=conf,
        status="corrected", method_note="test",
    )


def _plots_gdf(
    plot_numbers: list[str],
    map_areas: list[float | None] | None = None,
    rec_areas: list[float | None] | None = None,
) -> gpd.GeoDataFrame:
    """Minimal plots GDF with area columns for area-ratio tests."""
    if not plot_numbers:
        return gpd.GeoDataFrame(
            columns=["plot_number", "map_area_sqm", "recorded_area_sqm", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        ).set_index("plot_number", drop=False)
    rows = []
    for i, pn in enumerate(plot_numbers):
        rows.append({
            "plot_number":       pn,
            "map_area_sqm":      (map_areas[i]  if map_areas  else 1000.0),
            "recorded_area_sqm": (rec_areas[i]  if rec_areas  else 1000.0),
            "geometry":          box(74.0 + i * 0.01, 20.0, 74.01 + i * 0.01, 20.01),
        })
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    gdf["plot_number"] = gdf["plot_number"].astype(str)
    return gdf.set_index("plot_number", drop=False)


# ── constructor validation ────────────────────────────────────────────────────

class TestConstructor:
    def test_invalid_flag_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="flag_threshold"):
            ConfidenceCalibrator(flag_threshold=1.5)

    def test_negative_flag_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            ConfidenceCalibrator(flag_threshold=-0.1)

    def test_invalid_area_ratio_raises(self) -> None:
        with pytest.raises(ValueError):
            ConfidenceCalibrator(area_ratio_min=2.0, area_ratio_max=1.0)

    def test_zero_area_ratio_min_raises(self) -> None:
        with pytest.raises(ValueError):
            ConfidenceCalibrator(area_ratio_min=0.0)

    def test_valid_defaults(self) -> None:
        cal = ConfidenceCalibrator()
        assert cal._flag_threshold  == 0.15
        assert cal._area_ratio_min  == 0.50
        assert cal._area_ratio_max  == 2.00


# ── empty input ───────────────────────────────────────────────────────────────

class TestEmpty:
    def test_empty_list_returns_empty(self) -> None:
        cal = ConfidenceCalibrator()
        plots = _plots_gdf([])
        assert cal.calibrate([], plots) == []


# ── rank normalisation ────────────────────────────────────────────────────────

class TestRankNormalisation:
    def test_calibrated_range(self) -> None:
        pns = [str(i) for i in range(10)]
        results = [_result(pn, float(i) / 9) for i, pn in enumerate(pns)]
        plots   = _plots_gdf(pns)
        cal     = ConfidenceCalibrator(flag_threshold=0.0)
        out     = cal.calibrate(results, plots)
        corrected = [r for r in out if r.status == "corrected"]
        cals = [r.calibrated_confidence for r in corrected]
        assert min(cals) >= _CONF_LO - 1e-9
        assert max(cals) <= _CONF_HI + 1e-9

    def test_rank_order_preserved(self) -> None:
        """Higher raw confidence → higher calibrated confidence."""
        pns     = ["a", "b", "c", "d"]
        confs   = [0.1, 0.4, 0.7, 0.9]
        results = [_result(pn, c) for pn, c in zip(pns, confs)]
        plots   = _plots_gdf(pns)
        cal     = ConfidenceCalibrator(flag_threshold=0.0)
        out_map = {r.plot_number: r.calibrated_confidence for r in cal.calibrate(results, plots)
                   if r.status == "corrected"}
        cal_vals = [out_map[pn] for pn in pns]
        assert cal_vals == sorted(cal_vals), "rank order not preserved"

    def test_single_active_plot_gets_midpoint(self) -> None:
        results = [_result("only", 0.5)]
        plots   = _plots_gdf(["only"])
        cal     = ConfidenceCalibrator(flag_threshold=0.0)
        out     = cal.calibrate(results, plots)
        assert len(out) == 1
        mid = (_CONF_LO + _CONF_HI) / 2.0
        assert abs(out[0].calibrated_confidence - mid) < 1e-9


# ── flag threshold ────────────────────────────────────────────────────────────

class TestFlagThreshold:
    def test_bottom_fraction_flagged(self) -> None:
        pns     = [str(i) for i in range(20)]
        results = [_result(pn, float(i) / 19) for i, pn in enumerate(pns)]
        plots   = _plots_gdf(pns)
        cal     = ConfidenceCalibrator(flag_threshold=0.20)
        out     = cal.calibrate(results, plots)
        n_flagged = sum(1 for r in out if r.status == "flagged")
        # 20% of 20 = 4 (approximately; quantile behaviour may vary by 1)
        assert 3 <= n_flagged <= 5

    def test_zero_threshold_flags_nothing(self) -> None:
        pns     = ["a", "b", "c"]
        results = [_result(pn, 0.5) for pn in pns]
        plots   = _plots_gdf(pns)
        cal     = ConfidenceCalibrator(flag_threshold=0.0)
        out     = cal.calibrate(results, plots)
        assert all(r.status == "corrected" for r in out)

    def test_flagged_confidence_is_zero(self) -> None:
        pns     = [str(i) for i in range(5)]
        results = [_result(pn, float(i) / 4) for i, pn in enumerate(pns)]
        plots   = _plots_gdf(pns)
        cal     = ConfidenceCalibrator(flag_threshold=0.20)
        out     = cal.calibrate(results, plots)
        for r in out:
            if r.status == "flagged":
                assert r.calibrated_confidence == 0.0


# ── area-ratio pre-filter ─────────────────────────────────────────────────────

class TestAreaRatioFilter:
    def test_ratio_too_low_is_flagged(self) -> None:
        """map_area 100, recorded_area 1000 → ratio 0.1 < 0.5 → pre-flagged."""
        results = [_result("p1", 0.9)]
        plots   = _plots_gdf(["p1"], map_areas=[100.0], rec_areas=[1000.0])
        cal     = ConfidenceCalibrator(flag_threshold=0.0)
        out     = cal.calibrate(results, plots)
        assert out[0].status == "flagged"

    def test_ratio_too_high_is_flagged(self) -> None:
        """map_area 10000, recorded_area 1000 → ratio 10 > 2.0 → pre-flagged."""
        results = [_result("p1", 0.99)]
        plots   = _plots_gdf(["p1"], map_areas=[10000.0], rec_areas=[1000.0])
        cal     = ConfidenceCalibrator(flag_threshold=0.0)
        out     = cal.calibrate(results, plots)
        assert out[0].status == "flagged"

    def test_ratio_near_1_is_kept(self) -> None:
        results = [_result("p1", 0.8)]
        plots   = _plots_gdf(["p1"], map_areas=[1050.0], rec_areas=[1000.0])
        cal     = ConfidenceCalibrator(flag_threshold=0.0)
        out     = cal.calibrate(results, plots)
        assert out[0].status == "corrected"

    def test_null_recorded_area_skips_filter(self) -> None:
        """No recorded area → area filter is skipped, confidence threshold applies."""
        results = [_result("p1", 0.9)]
        plots   = _plots_gdf(["p1"], map_areas=[500.0], rec_areas=[None])
        cal     = ConfidenceCalibrator(flag_threshold=0.0)
        out     = cal.calibrate(results, plots)
        assert out[0].status == "corrected"

    def test_missing_plot_number_skips_gracefully(self) -> None:
        """Plot not in GDF → area filter skipped, no crash."""
        results = [_result("unknown_plot", 0.8)]
        plots   = _plots_gdf(["other_plot"])
        cal     = ConfidenceCalibrator(flag_threshold=0.0)
        out     = cal.calibrate(results, plots)
        assert len(out) == 1


# ── output invariants ─────────────────────────────────────────────────────────

class TestOutputInvariants:
    def test_output_length_matches_input(self) -> None:
        pns     = [str(i) for i in range(15)]
        results = [_result(pn, float(i) / 14) for i, pn in enumerate(pns)]
        plots   = _plots_gdf(pns)
        out     = ConfidenceCalibrator().calibrate(results, plots)
        assert len(out) == len(results)

    def test_plot_numbers_unchanged(self) -> None:
        pns     = ["a", "b", "c"]
        results = [_result(pn, 0.5) for pn in pns]
        plots   = _plots_gdf(pns)
        out     = ConfidenceCalibrator(flag_threshold=0.0).calibrate(results, plots)
        assert [r.plot_number for r in out] == pns

    def test_dx_dy_unchanged(self) -> None:
        r = AlignmentResult("p", 3.5, -2.1, 0.7, 0.7, "corrected", "test")
        plots = _plots_gdf(["p"])
        out   = ConfidenceCalibrator(flag_threshold=0.0).calibrate([r], plots)
        assert out[0].dx_m == pytest.approx(3.5)
        assert out[0].dy_m == pytest.approx(-2.1)
