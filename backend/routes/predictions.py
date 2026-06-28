"""Pipeline execution endpoint with WebSocket progress streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from backend.core.config import Settings, get_settings

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/villages", tags=["pipeline"])

# In-memory job state (sufficient for single-server dev; replace with Redis for prod)
_jobs: dict[str, dict] = {}   # slug → {status, progress, total, error, scorecard}


# ── trigger pipeline run ──────────────────────────────────────────────────────

@router.post("/{slug}/run")
def run_pipeline(
    slug: str,
    settings: Annotated[Settings, Depends(get_settings)],
):
    """Trigger the correction pipeline for *slug* in a background thread.

    Returns immediately with ``{"job_id": slug, "status": "started"}``.
    Poll ``GET /api/villages/{slug}/status`` or connect to
    ``WS /ws/villages/{slug}/progress`` for live updates.
    """
    d = settings.data_dir / slug
    if not d.exists():
        raise HTTPException(404, f"Village '{slug}' not found")

    if _jobs.get(slug, {}).get("status") == "running":
        return {"job_id": slug, "status": "already_running"}

    _jobs[slug] = {"status": "running", "progress": 0, "total": 0, "error": None}

    thread = threading.Thread(
        target=_run_pipeline_sync,
        args=(slug, d, settings),
        daemon=True,
    )
    thread.start()
    return {"job_id": slug, "status": "started"}


@router.get("/{slug}/status")
def job_status(slug: str):
    """Return current pipeline job state for *slug*."""
    state = _jobs.get(slug)
    if state is None:
        return {"status": "not_started"}
    return state


# ── WebSocket progress stream ─────────────────────────────────────────────────

@router.websocket("/{slug}/ws")
async def ws_progress(websocket: WebSocket, slug: str):
    """Stream ``{progress, total, status}`` messages until the job completes."""
    await websocket.accept()
    try:
        while True:
            state = _jobs.get(slug, {"status": "not_started", "progress": 0, "total": 0})
            await websocket.send_json(state)
            if state["status"] in ("done", "error", "not_started"):
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()


# ── background worker ─────────────────────────────────────────────────────────

def _run_pipeline_sync(slug: str, village_dir: Path, settings: Settings) -> None:
    """Run the full Predictor pipeline in a background thread."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    try:
        from bhume import load, score, write_predictions
        from solution.pipeline import Predictor

        village = load(village_dir)
        _jobs[slug]["total"] = len(village.plots)

        def _cb(done: int, total: int) -> None:
            _jobs[slug]["progress"] = done
            _jobs[slug]["total"]    = total

        predictor = Predictor(
            search_radius_m=settings.search_radius_m,
            flag_threshold=settings.flag_threshold,
        )
        preds = predictor.predict(village, progress_cb=_cb)
        write_predictions(village_dir / "predictions.geojson", preds)

        # Self-score if example truths are present
        scorecard = None
        if village.example_truths is not None:
            sc = score(preds, village)
            scorecard = {
                "median_iou_pred":     sc.median_iou_pred,
                "median_iou_official": sc.median_iou_official,
                "improvement":         sc.median_improvement,
                "spearman":            sc.spearman_conf_vs_iou,
                "auc":                 sc.auc_accurate_vs_conf,
                "n_corrected":         sc.n_corrected,
                "n_flagged":           sc.n_flagged,
            }

        _jobs[slug].update({
            "status":    "done",
            "progress":  _jobs[slug]["total"],
            "scorecard": scorecard,
        })
        log.info("Pipeline done for %s", slug)

    except Exception as exc:
        log.exception("Pipeline failed for %s", slug)
        _jobs[slug].update({"status": "error", "error": str(exc)})
