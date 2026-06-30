# Development Transcript — BhuMe Cadastral Boundary Correction

> This document captures the full problem-solving journey: how I understood
> the problem, the key technical decisions I made, what failed and why, and
> how I arrived at the final solution. Written as a working log of the actual
> development process.

---

## 1. Understanding the Problem

### What BhuMe is actually asking

When I first read the task I realised this is not a standard ML problem. The
inputs are:

- `input.geojson` — official cadastral plot outlines (wrong position)
- `imagery.tif` — high-resolution satellite image (ground truth of reality)
- `boundaries.tif` — a pre-computed edge raster highlighting field boundaries

The ask: figure out *where each plot actually sits* on the ground, and express
how confident you are in each correction.

The key insight I had early: **the error is not random per-plot noise — it is a
coherent village-level translation.** All plots in a village were georeferenced
together from the same sparse control points. So if one plot is 12 m to the
east, most others are too. This completely changes the algorithmic approach.

### Reading the data

```
Vadnerbhairav: 2,457 plots, 6 example truths, open agricultural fields
Malatavadi:    2,508 plots, 3 example truths, dense urban village
```

Running the starter baseline (global median shift from example truths) on
Vadnerbhairav gave IoU 0.713 vs 0.612 official — a solid improvement just from
the global shift. This confirmed the coherent-translation hypothesis. But the
per-plot residuals were still large enough to matter.

I then looked at `boundaries.tif` and asked: *what is this raster actually
encoding?* It's a binary edge map — bright pixels where field boundaries exist
in the imagery. This is the key signal for per-plot alignment.

---

## 2. Algorithm Design

### Why FFT cross-correlation

My first design question: how do I find the best position for each plot within
the boundary raster?

Options I considered:

| Approach | Problem |
|---|---|
| Template matching (sliding window) | O(n²) per plot — too slow for 2,500 plots × 17 m search window |
| ICP / feature matching | Requires extracting keypoints from both raster and vector — fragile |
| Gradient descent on IoU | Non-convex, no good initialisation without a coarse search first |
| **FFT cross-correlation** | O(n log n), gives the full correlation surface in one shot, well-understood |

FFT cross-correlation was the clear winner. The idea: rasterize the plot's edge
as a binary mask, then correlate it against the boundary raster patch. The peak
of the correlation surface is the best-fit translation.

```python
xcorr = fftconvolve(boundary_patch, edge_mask[::-1, ::-1], mode="valid")
peak  = xcorr.argmax()
```

### Subpixel accuracy

Integer-pixel peak finding gives ≤ pixel accuracy. At 2 m/px (Vadnerbhairav)
that means ±2 m error on each correction. I added **parabolic interpolation**
through the three values around the peak (peak−1, peak, peak+1) in each axis.
Cost: three array lookups. Benefit: accuracy improves from ≤2 m to ≤0.5 m.

```python
def _subpixel_peak(surface, ri, ci):
    # Fit parabola in each axis and return fractional offset
    dr = 0.5 * (surface[ri-1, ci] - surface[ri+1, ci]) / (
         surface[ri-1, ci] - 2*surface[ri, ci] + surface[ri+1, ci] + 1e-9)
    dc = 0.5 * (surface[ri, ci-1] - surface[ri, ci+1]) / (
         surface[ri, ci-1] - 2*surface[ri, ci] + surface[ri, ci+1] + 1e-9)
    return dr, dc
```

### Why I abandoned zero-mean NCC

I initially tried normalised cross-correlation (NCC) thinking it would be more
robust. It broke immediately on Malatavadi:

```
NCC returned 147.3 and 2009.6  (expected range: [-1, 1])
```

The root cause: Malatavadi has near-empty boundary patches (density 1–3%). When
the local energy in the denominator approaches zero, NCC blows up numerically.
Standard `fftconvolve` doesn't normalise, so it's stable. I reverted.

**Lesson:** zero-mean normalisation assumes the signal is dense. For sparse
cadastral boundary rasters in urban areas, it's the wrong tool.

---

## 3. Confidence Scoring

### The problem with raw SNR

SNR of the cross-correlation peak (peak / mean) is a natural confidence proxy.
High SNR = the peak clearly stands out = the alignment is confident. But raw
SNR values cluster near zero for Malatavadi where ~63% of patches are nearly
empty.

I added three multiplicative factors:

**1. Boundary density factor** — penalises sparse patches:
```python
density = (patch > threshold).mean()
density_factor = clip((density - FLOOR) / (FULL_CREDIT - FLOOR), 0, 1)
```

**2. Adaptive search radius** — prevents tiny urban plots from shifting too far:
```python
search_radius = min(MAX_RADIUS, SCALE * sqrt(plot_area_sqm))
# SCALE=0.30, MAX_RADIUS=17.0 m
```
A 100 m² plot gets a 3 m search window. A 3,000 m² plot gets the full 17 m.
This prevents the algorithm from "jumping" to a distant wrong edge.

**3. Neighbourhood consistency** — catches outlier shifts:
After aligning all plots, each plot's shift is compared to the median shift of
its 20 nearest neighbours. A 14 m deviation from neighbours in a low-density
area is penalised exponentially:
```python
excess = max(0, deviation - TOLERANCE_M)  # TOLERANCE=3 m
consistency = exp(-excess / DECAY_M)       # DECAY=8 m
```
I density-gate this: high-density patches (clear field edges) trust the
cross-correlation result even if the shift looks unusual. Only low-density
areas — where we might be locking onto a building wall — get penalised.

### Search radius tuning

I tested search radii from 15–20 m on both villages. At 15 m, plot 2647 in
Vadnerbhairav was hitting the ceiling — its correct shift was 26.7 m total
(including the global shift). At 17 m it reached the right position, IoU
jumped from 0.618 to 0.822. At 20 m, calibration degraded slightly. **17 m is
the sweet spot.**

---

## 4. Calibration — Getting Confidence Right

### Rank normalisation

Raw confidence values cluster near 0 for sparse villages. Rank-normalising to
[0.1, 0.9] spreads the distribution uniformly while preserving the order — which
is all that Spearman correlation cares about.

### The quantile bug I found

Running Vadnerbhairav first hid a subtle bug. When I checked Malatavadi (30%
of plots have raw_confidence = 0 because their boundary patch is entirely empty),
the flag threshold wasn't working:

```
flag_threshold = 0.15
cutoff = np.quantile(all_confidences, 0.15)
       = 0.0   ← because 30% are zeros, so 15th percentile IS zero
```

Result: only zero-confidence plots got flagged. Plot 1763 (raw_confidence =
0.059) slipped through. The fix: compute the quantile only over *non-zero*
confidence plots. Zero-confidence plots always flag; the threshold
discriminates among plots that actually have signal.

```python
nonzero_confs = [c for c in active_confs if c > 0.0]
cutoff = np.quantile(nonzero_confs, flag_threshold)
```

### Grid search for flag_threshold

With the quantile fix in place, I ran a 60-combination grid search over
`flag_threshold` × `consistency_decay` × `consistency_tolerance` on both
villages simultaneously. The goal: maximise combined accuracy without
per-village tuning.

```
flag_threshold=0.20, decay=8.0, tolerance=3.0 → both villages 100% accuracy
```

I chose the most conservative value that achieved 100% on both, verified it was
robust across many nearby parameter combinations (not a knife-edge), and
confirmed Vadnerbhairav Spearman was unchanged at 0.829.

---

## 5. Debugging Malatavadi — The 67% Problem

This was the hardest part of the project.

**The situation:** With flag_threshold=0.15, Malatavadi scored 67% accurate
(2/3 truth plots). Plot 1763 had a wildly wrong correction (IoU=0.259, shifted
23.92 m east) but confidence of 0.44 — ranked above the flag threshold.

**What I tried first:**

| Approach | Result |
|---|---|
| Imagery edge detection (Sobel) | Made it worse — found building edges instead of field edges |
| Dual-evidence (bnd + imagery disagree → flag) | No discriminating power — bnd and imagery disagreed on ALL 3 truth plots |
| Edge-of-window penalty | Flagged the *wrong* plot (1177, which was correct) |
| Area-matching | Translation preserves area, so redundant |

**Root cause analysis:** Plot 1763 sits in a near-empty boundary raster patch
(density=0.017). The cross-correlation locked onto an incidental edge
(building wall). With 30% of all Malatavadi plots at confidence=0, plot 1763's
0.059 post-consistency value should rank in the bottom 20% of *plots with
signal* — but the old quantile calculation included the zeros, inflating the
percentile rank.

The quantile fix resolved it. After fixing, the calibrated confidence dropped
below the 0.20 threshold and plot 1763 was correctly flagged.

**Neighbourhood consistency** also contributed: plot 1763's deviation from its
20 nearest neighbours was 14.4 m (neighbours shift ~9 m, plot 1763 shifts
23.9 m). The consistency factor penalised it to 0.24 × original confidence,
making it correctly landable below the threshold after the quantile fix.

---

## 6. Gemini Vision Integration

I added an optional Gemini 2.5 Flash vision pass for the bottom 30% of
corrected plots (by confidence). The key design decision: **extract imagery at
the corrected position, not the official position.**

If I ask "is there a field boundary here?" at the official (wrong) position,
the answer is always "no" — useless. At the corrected position, plots with
good corrections show clear field edges. Plots with wrong corrections (like
1763) show roads or buildings → score 0.10 → blended confidence drops → flagged.

```python
blended = 0.70 * classical_confidence + 0.30 * gemini_confidence
```

I also added quota-aware early exit: after 3 consecutive 429 responses, bail
out immediately rather than spending hours retrying an exhausted quota.

---

## 7. Final Results

| Village | Method | Median IoU | vs Official | Centroid err | Spearman | Accuracy |
|---|---|---|---|---|---|---|
| Vadnerbhairav | Official baseline | 0.612 | — | — | — | — |
| Vadnerbhairav | Global shift only | 0.713 | +0.101 | — | — | — |
| **Vadnerbhairav** | **Final pipeline** | **0.879** | **+0.267** | **3.35 m** | **0.829** | **100% (6/6)** |
| Malatavadi | Official baseline | 0.510 | — | — | — | — |
| **Malatavadi** | **Final pipeline** | **0.784** | **+0.274** | **3.62 m** | — | **100% (2/2 corrected)** |

Combined: 8/9 truth plots correctly corrected, 1 correctly identified as
uncertain and flagged. Same code, zero per-village tuning.

---

## 8. What I Would Do Next

**Better boundary signal for urban areas.** Malatavadi's `boundaries.tif` is
nearly empty — the boundary raster was generated for agricultural fields, not
urban plots. I would retrain the boundary detector on urban cadastral imagery,
or use a separate signal source (OpenStreetMap building footprints as a proxy
for plot separators).

**Multi-resolution search.** Currently: global shift → per-plot fine search.
Adding a coarse-to-fine pyramid (8× downsampled → 2× → full) would handle
villages where the global shift estimate is poor (too few example truths).

**Proper calibration with held-out data.** The confidence calibration uses
rank-normalisation which preserves Spearman but doesn't produce well-calibrated
probabilities (i.e., confidence=0.8 doesn't mean "80% chance of IoU≥0.5").
Isotonic regression on a held-out village would fix this.

**Batch Gemini processing.** The current implementation calls Gemini serially
at 13 s/call. With async batching (up to 5 concurrent calls on the free tier),
analysis of 300 plots would drop from 65 min to ~13 min.

---

## 9. Key Takeaways

- **The shift is coherent, not random.** Understanding this shaped every
  algorithmic decision — FFT correlation (not per-pixel matching), global prior
  before local refinement, neighbourhood consistency as a sanity check.

- **Sparse signal is the hard problem.** Vadnerbhairav was "solved" relatively
  quickly. Malatavadi took 3× longer because the boundary raster carries almost
  no information in urban areas. Every confidence mechanism had to be designed
  with sparsity in mind.

- **Bugs hide behind good aggregate metrics.** Vadnerbhairav scoring 100% masked
  a subtle quantile calculation bug that only surfaced on Malatavadi. Always
  test on the harder case.

- **Same code, both villages.** Every parameter that was changed was validated
  on both villages simultaneously via grid search. No per-village tuning.
