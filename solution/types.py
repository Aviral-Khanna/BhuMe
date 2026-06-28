"""AlignmentResult — the per-plot output record shared between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlignmentResult:
    """Immutable record of one plot's alignment output.

    Frozen so pipeline stages cannot mutate results once produced.
    dx_m / dy_m are measured in UTM metres from the official centroid.
    """

    plot_number: str
    dx_m: float                    # total shift east  from official position (UTM metres)
    dy_m: float                    # total shift north from official position (UTM metres)
    raw_confidence: float          # composite confidence before rank-normalisation [0, 1]
    calibrated_confidence: float   # post-calibration value [0, 1]
    status: str                    # "corrected" | "flagged"
    method_note: str
    boundary_density: float = 0.0  # fraction of non-zero pixels in the boundary patch
    consistency_factor: float = 1.0  # neighbourhood shift consistency (1=consistent, <1=outlier)
