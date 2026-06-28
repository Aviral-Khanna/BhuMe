"""AI explanation endpoint — calls GeminiExplainer for a single plot."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.core.config import Settings, get_settings

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/explain", tags=["ai"])


class ExplainRequest(BaseModel):
    slug:        str
    plot_number: str


@router.post("/")
def explain_plot(
    req: ExplainRequest,
    settings: Annotated[Settings, Depends(get_settings)],
):
    """Return a Gemini Flash explanation for one plot's correction result.

    Falls back to a structured plain-text explanation when the API key is
    absent or the API call fails.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from solution.ai_explain import GeminiExplainer
    from solution.types import AlignmentResult

    village_dir  = settings.data_dir / req.slug
    preds_path   = village_dir / "predictions.geojson"
    input_path   = village_dir / "input.geojson"

    if not preds_path.exists():
        raise HTTPException(404, "Run the pipeline first")
    if not input_path.exists():
        raise HTTPException(404, "input.geojson not found")

    # Load prediction for this plot
    with open(preds_path) as f:
        preds = json.load(f)
    pred_feat = next(
        (ft for ft in preds["features"]
         if str(ft["properties"]["plot_number"]) == str(req.plot_number)),
        None,
    )
    if pred_feat is None:
        raise HTTPException(404, f"Plot {req.plot_number} not in predictions")

    # Load plot properties
    with open(input_path) as f:
        inputs = json.load(f)
    input_feat = next(
        (ft for ft in inputs["features"]
         if str(ft["properties"]["plot_number"]) == str(req.plot_number)),
        None,
    )
    plot_props = input_feat["properties"] if input_feat else {}

    # Build a lightweight AlignmentResult from the stored prediction
    props = pred_feat["properties"]
    result = AlignmentResult(
        plot_number=req.plot_number,
        dx_m=0.0, dy_m=0.0,
        raw_confidence=float(props.get("confidence") or 0),
        calibrated_confidence=float(props.get("confidence") or 0),
        status=props.get("status", "corrected"),
        method_note=props.get("method_note", ""),
    )

    explainer = GeminiExplainer(api_key=settings.gemini_api_key or None)
    explanation = explainer.explain(req.plot_number, result, plot_props)

    return {
        "plot_number": req.plot_number,
        "explanation": explanation,
        "status":      result.status,
        "confidence":  result.calibrated_confidence,
    }
