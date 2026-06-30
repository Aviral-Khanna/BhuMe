#!/usr/bin/env python3
"""BhuMe boundary correction pipeline — main entry point.

Usage
-----
::

    # single village
    uv run predict.py data/34855_vadnerbhairav_chandavad_nashik

    # both villages
    uv run predict.py data/34855_vadnerbhairav_chandavad_nashik \\
                      data/12429_malatavadi_chandgad_kolhapur

    # tune parameters
    uv run predict.py data/34855_vadnerbhairav_chandavad_nashik \\
        --search-radius 20 --flag-threshold 0.10

    # suppress self-score
    uv run predict.py data/34855_vadnerbhairav_chandavad_nashik --no-score
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Load .env so GEMINI_API_KEY is available without manual export
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file)
    except ImportError:
        pass

from bhume import load, score, write_predictions
from solution.pipeline import Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("predict")


def _progress_bar(done: int, total: int, width: int = 40) -> None:
    filled = int(width * done / total)
    bar = "█" * filled + "░" * (width - filled)
    pct = 100 * done / total
    print(f"\r  [{bar}] {pct:5.1f}%  {done}/{total}", end="", flush=True)
    if done == total:
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Correct cadastral plot boundaries for a BhuMe village bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "village_dirs",
        nargs="+",
        type=Path,
        metavar="village_dir",
        help="Village bundle directory (must contain input.geojson + imagery.tif).",
    )
    parser.add_argument(
        "--search-radius",
        type=float,
        default=17.0,
        metavar="M",
        help="Translation search radius in metres.",
    )
    parser.add_argument(
        "--flag-threshold",
        type=float,
        default=0.20,
        metavar="F",
        help="Fraction of low-confidence plots to flag.",
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Skip self-scoring against example truths.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bar (useful for scripted runs).",
    )
    args = parser.parse_args(argv)

    predictor = Predictor(
        search_radius_m=args.search_radius,
        flag_threshold=args.flag_threshold,
    )

    exit_code = 0
    for village_dir in args.village_dirs:
        print(f"\n{'═' * 62}")
        print(f"  Village  : {village_dir.name}")
        print(f"  Params   : search_radius={args.search_radius}m  "
              f"flag_threshold={args.flag_threshold}")
        print(f"{'═' * 62}")

        try:
            t0 = time.time()
            village = load(village_dir)
            n_truth = 0 if village.example_truths is None else len(village.example_truths)
            print(f"  plots          : {len(village.plots):,}")
            print(f"  example truths : {n_truth}")
            print(f"  boundaries     : {'yes' if village.boundaries_path else 'MISSING'}")
            print()

            cb = None if args.quiet else _progress_bar
            preds = predictor.predict(village, progress_cb=cb)

            elapsed = time.time() - t0
            n_corrected = int((preds["status"] == "corrected").sum())
            n_flagged   = int((preds["status"] == "flagged").sum())

            out = write_predictions(village_dir / "predictions.geojson", preds)
            print(f"  corrected : {n_corrected:,}  flagged : {n_flagged:,}  "
                  f"({elapsed:.1f}s)")
            print(f"  output    : {out}")

            if not args.no_score and village.example_truths is not None:
                print()
                sc = score(preds, village)
                print(sc)

        except Exception as exc:
            log.error("failed on %s: %s", village_dir, exc)
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
