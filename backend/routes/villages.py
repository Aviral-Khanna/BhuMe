"""Village data endpoints — plots, imagery metadata, predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from backend.core.config import Settings, get_settings

router = APIRouter(prefix="/api/villages", tags=["villages"])


def _village_dir(slug: str, settings: Settings) -> Path:
    d = settings.data_dir / slug
    if not d.exists():
        raise HTTPException(404, f"Village '{slug}' not found in {settings.data_dir}")
    return d


# ── list villages ─────────────────────────────────────────────────────────────

@router.get("/")
def list_villages(settings: Annotated[Settings, Depends(get_settings)]):
    """Return metadata for every village bundle present in data_dir."""
    villages = []
    for d in sorted(settings.data_dir.iterdir()):
        if not d.is_dir():
            continue
        input_path = d / "input.geojson"
        if not input_path.exists():
            continue
        try:
            with open(input_path) as f:
                data = json.load(f)
            n_plots = len(data.get("features", []))
        except Exception:
            n_plots = 0

        villages.append({
            "slug":               d.name,
            "n_plots":            n_plots,
            "has_imagery":        (d / "imagery.tif").exists(),
            "has_boundaries":     (d / "boundaries.tif").exists(),
            "has_example_truths": (d / "example_truths.geojson").exists(),
            "has_predictions":    (d / "predictions.geojson").exists(),
        })
    return villages


# ── village plots ─────────────────────────────────────────────────────────────

@router.get("/{slug}/plots")
def get_plots(
    slug: str,
    settings: Annotated[Settings, Depends(get_settings)],
):
    """Return the official input plots as a GeoJSON FeatureCollection."""
    d = _village_dir(slug, settings)
    path = d / "input.geojson"
    if not path.exists():
        raise HTTPException(404, "input.geojson not found")
    return FileResponse(path, media_type="application/geo+json")


# ── predictions ───────────────────────────────────────────────────────────────

@router.get("/{slug}/predictions")
def get_predictions(
    slug: str,
    settings: Annotated[Settings, Depends(get_settings)],
):
    """Return saved predictions.geojson (404 if pipeline hasn't run yet)."""
    d = _village_dir(slug, settings)
    path = d / "predictions.geojson"
    if not path.exists():
        raise HTTPException(404, "No predictions yet — POST /api/villages/{slug}/run first")
    return FileResponse(path, media_type="application/geo+json")


# ── example truths ────────────────────────────────────────────────────────────

@router.get("/{slug}/truths")
def get_truths(
    slug: str,
    settings: Annotated[Settings, Depends(get_settings)],
):
    """Return example_truths.geojson when available."""
    d = _village_dir(slug, settings)
    path = d / "example_truths.geojson"
    if not path.exists():
        raise HTTPException(404, "example_truths.geojson not found")
    return FileResponse(path, media_type="application/geo+json")
