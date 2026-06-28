"""FastAPI application — BhuMe boundary correction API.

Start with:
    uv run uvicorn backend.main:app --reload --port 8000

Endpoints
---------
GET  /api/villages/                     list villages
GET  /api/villages/{slug}/plots         official input GeoJSON
GET  /api/villages/{slug}/predictions   saved predictions GeoJSON
GET  /api/villages/{slug}/truths        example truths GeoJSON
GET  /api/villages/{slug}/status        pipeline job status
POST /api/villages/{slug}/run           trigger pipeline (background)
WS   /ws/villages/{slug}/progress       live progress stream
POST /api/explain/                      Gemini plot explanation
GET  /health                            liveness probe
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so bhume/solution are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import explain, predictions, villages

app = FastAPI(
    title="BhuMe Boundary Correction API",
    description="REST + WebSocket API for cadastral plot boundary correction.",
    version="0.1.0",
)

# Allow the React dev server (port 5173) during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(villages.router)
app.include_router(predictions.router)
app.include_router(explain.router)


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}
