"""End-to-end integration test for Predictor with a synthetic 3-plot Village."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.io import MemoryFile
from shapely.geometry import box, mapping

from bhume.io import Village
from solution.pipeline import Predictor
from solution.types import AlignmentResult


# ── synthetic village fixture ─────────────────────────────────────────────────

def _make_raster_file(tmp_path: Path) -> Path:
    """Write a 200×200 float32 raster with random non-zero data to tmp_path."""
    rng  = np.random.default_rng(42)
    data = rng.random((200, 200)).astype(np.float32)
    path = tmp_path / "boundaries.tif"
    # Raster covers a small area near Vadnerbhairav in EPSG:3857
    west, north, res = 8_230_000.0, 2_310_000.0, 2.0
    transform = Affine(res, 0.0, west, 0.0, -res, north)
    with rasterio.open(
        path, "w", driver="GTiff", height=200, width=200,
        count=1, dtype="float32", crs="EPSG:3857", transform=transform,
    ) as dst:
        dst.write(data, 1)
    return path


def _make_village(tmp_path: Path) -> Village:
    """Build a minimal synthetic Village with 3 plots."""
    plots_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(box(74.030, 20.240, 74.031, 20.241)),
                "properties": {
                    "plot_number": "1",
                    "village":     "test",
                    "map_area_sqm":      1000.0,
                    "recorded_area_sqm": 1000.0,
                    "recorded_area_ha":  0.1,
                    "pot_kharaba_ha":    None,
                    "surveys": [],
                },
            },
            {
                "type": "Feature",
                "geometry": mapping(box(74.032, 20.240, 74.033, 20.241)),
                "properties": {
                    "plot_number": "2",
                    "village":     "test",
                    "map_area_sqm":      1200.0,
                    "recorded_area_sqm": 1000.0,
                    "recorded_area_ha":  0.1,
                    "pot_kharaba_ha":    None,
                    "surveys": [],
                },
            },
            {
                "type": "Feature",
                "geometry": mapping(box(74.034, 20.240, 74.035, 20.241)),
                "properties": {
                    "plot_number": "3",
                    "village":     "test",
                    "map_area_sqm":      5000.0,   # ratio = 5 → area-error, will be flagged
                    "recorded_area_sqm": 1000.0,
                    "recorded_area_ha":  0.1,
                    "pot_kharaba_ha":    None,
                    "surveys": [],
                },
            },
        ],
    }

    truths_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(box(74.030, 20.2405, 74.031, 20.2415)),
                "properties": {"plot_number": "1", "status": "aligned", "note": ""},
            },
        ],
    }

    village_dir = tmp_path / "test_village"
    village_dir.mkdir()

    (village_dir / "input.geojson").write_text(json.dumps(plots_data))
    (village_dir / "example_truths.geojson").write_text(json.dumps(truths_data))

    _make_raster_file(village_dir)  # writes village_dir/boundaries.tif directly
    # Write a minimal imagery placeholder so Village can be constructed
    img_path = village_dir / "imagery.tif"
    transform = Affine(2.0, 0.0, 8_230_000.0, 0.0, -2.0, 2_310_000.0)
    with rasterio.open(
        img_path, "w", driver="GTiff", height=50, width=50,
        count=3, dtype="uint8", crs="EPSG:3857", transform=transform,
    ) as dst:
        dst.write(np.zeros((3, 50, 50), dtype=np.uint8))

    from bhume.io import load
    return load(village_dir)


# ── tests ─────────────────────────────────────────────────────────────────────

class TestPredictorIntegration:
    @pytest.fixture()
    def village(self, tmp_path: Path) -> Village:
        return _make_village(tmp_path)

    def test_output_covers_all_plots(self, village: Village) -> None:
        preds = Predictor().predict(village)
        assert set(preds["plot_number"].astype(str)) == {"1", "2", "3"}

    def test_output_has_required_columns(self, village: Village) -> None:
        preds = Predictor().predict(village)
        for col in ("plot_number", "status", "geometry"):
            assert col in preds.columns

    def test_status_values_are_valid(self, village: Village) -> None:
        preds = Predictor().predict(village)
        assert set(preds["status"]).issubset({"corrected", "flagged"})

    def test_confidence_in_range(self, village: Village) -> None:
        preds = Predictor().predict(village)
        corrected = preds[preds["status"] == "corrected"]
        if len(corrected):
            assert corrected["confidence"].dropna().between(0.0, 1.0).all()

    def test_area_error_plot_is_flagged(self, village: Village) -> None:
        """Plot 3 has map/recorded ratio = 5 → must be flagged."""
        preds = Predictor(flag_threshold=0.0).predict(village)
        row = preds.loc["3"]
        assert row["status"] == "flagged", (
            f"expected plot 3 flagged (area ratio=5), got {row['status']}"
        )

    def test_crs_is_4326(self, village: Village) -> None:
        preds = Predictor().predict(village)
        assert preds.crs is not None
        assert preds.crs.to_epsg() == 4326

    def test_progress_callback_called(self, village: Village) -> None:
        calls: list[tuple[int, int]] = []
        Predictor().predict(village, progress_cb=lambda d, t: calls.append((d, t)))
        assert len(calls) == len(village.plots)
        assert calls[-1] == (len(village.plots), len(village.plots))

    def test_no_example_truths_still_runs(self, tmp_path: Path) -> None:
        """Pipeline must succeed even without example_truths.geojson."""
        v = _make_village(tmp_path)
        # Strip example truths
        from dataclasses import replace
        v_no_truth = Village(
            slug=v.slug, dir=v.dir,
            plots=v.plots, imagery_path=v.imagery_path,
            boundaries_path=v.boundaries_path, example_truths=None,
        )
        preds = Predictor().predict(v_no_truth)
        assert len(preds) == 3

    def test_missing_boundaries_path(self, tmp_path: Path) -> None:
        """Without boundaries.tif the pipeline falls back gracefully."""
        v = _make_village(tmp_path)
        from dataclasses import replace
        v_no_bnd = Village(
            slug=v.slug, dir=v.dir,
            plots=v.plots, imagery_path=v.imagery_path,
            boundaries_path=None, example_truths=v.example_truths,
        )
        preds = Predictor().predict(v_no_bnd)
        assert len(preds) == 3
        # All status values still valid
        assert set(preds["status"]).issubset({"corrected", "flagged"})
