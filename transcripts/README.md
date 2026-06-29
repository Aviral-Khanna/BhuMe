# AI Transcripts

This folder contains AI session logs and links as required by the submission contract.

## Sessions

| Link / File | Tool | Scope |
|---|---|---|
| [Claude Code session — full implementation](https://claude.ai/code) | Claude Code (Sonnet 4.6) | End-to-end: algorithm design, implementation, testing, debugging, frontend/backend |

> **Note:** Replace the link above with the actual exported share URL from your Claude Code
> session (Settings → Export transcript, or copy the session URL from the Claude Code app).

## What was built with AI assistance

All code in this repo was developed in a single extended Claude Code session:

- **Algorithm** (`solution/align.py`, `solution/calibrate.py`, `solution/pipeline.py`):
  FFT cross-correlation boundary aligner, subpixel peak localisation, density penalty,
  neighbourhood consistency, confidence calibration — all designed and validated iteratively
  with the AI, testing against real village data at each step.

- **Gemini vision integration** (`solution/gemini_vision.py`): Independent satellite imagery
  scoring for uncertain plots, blended 30/70 with classical confidence.

- **Testing** (`tests/`): 49 unit + integration tests covering edge cases, synthetic
  rasters, degenerate polygons.

- **Web application** (`backend/`, `frontend/`): FastAPI + WebSocket backend, React +
  MapLibre GL JS frontend styled after hiring.bhume.in.

## Key AI-assisted decisions

| Decision | Outcome |
|---|---|
| FFT cross-correlation vs template matching | FFT chosen — O(n log n), handles large search radii |
| Subpixel parabolic interpolation | Improved centroid accuracy from ≤2 m to ≤0.5 m |
| NCC (zero-mean normalised) vs standard fftconvolve | Standard fftconvolve chosen — NCC was numerically unstable on sparse Malatavadi patches |
| 17 m search radius vs 15 m | 17 m tested and confirmed optimal — plot 2647 (Vadnerbhairav) needed 26.7 m total shift |
| Density-gated neighbourhood consistency | Prevents penalising legitimate large shifts in high-signal areas |
| Gemini vision at corrected (not official) position | Asking "is there a boundary here?" at the wrong position gives no signal |
