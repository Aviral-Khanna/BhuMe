# AI Transcripts

This folder contains AI session logs and links as required by the submission contract.

## Sessions

| Link / File | Tool | Scope |
|---|---|---|
| *(add your Claude Code share URL here)* | Claude Code | End-to-end development — algorithm, testing, backend, frontend |

> Add the share URL from your Claude Code session before submitting:
> open the session in the Claude Code desktop app → Share → Copy link.

## How I used AI assistance

I built this project in an extended Claude Code session over several days.
The session covered the full stack — from designing and iterating on the
cross-correlation algorithm, to debugging edge cases, writing tests, and
building the FastAPI + React web app.

Key engineering decisions I made with AI assistance:

| Decision | Rationale |
|---|---|
| FFT cross-correlation (fftconvolve) over template matching | O(n log n), handles the 17 m search radius without blowing up on 2 500-plot villages |
| Subpixel parabolic interpolation at the correlation peak | Improved centroid accuracy from ≤2 m to ≤0.5 m; free — uses values already computed |
| Standard fftconvolve over zero-mean NCC | NCC was numerically unstable on sparse Malatavadi patches (denominator → 0) |
| Adaptive search radius: 0.30 × sqrt(area_sqm) | Tiny urban plots can't be shifted by more than a fraction of their own size |
| Density-gated neighbourhood consistency | Prevents falsely penalising large legitimate shifts in high-signal areas |
| Flag quantile over non-zero confidence only | When 30% of plots have conf=0 (sparse raster), including them made the 15 % threshold meaningless |
| flag_threshold 0.20, validated via 60-combination grid search | Confirmed on both villages simultaneously — not per-village tuning |
| Gemini vision at corrected (not official) position | Asking "is there a boundary here?" at the wrong position adds no signal |
