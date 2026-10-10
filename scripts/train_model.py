# ============================================================================
#  Sabeeq
#  Author : Aseel  (GitHub: aseel-aihub)
#  Copyright (c) 2026 Aseel. All rights reserved.
# ============================================================================
"""
Train the sector model and compare it with the baseline (guessing).

Usage:
    python scripts/train_model.py [--trees 200] [--depth 14]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add the project folder to the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import THREAT_DATA_CSV  # noqa: E402
from src.pattern_model import DEFAULT_DEPTH, DEFAULT_TREES, train  # noqa: E402


def _positive_int(text: str) -> int:
    """Only accept whole numbers bigger than 0."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", type=_positive_int, default=DEFAULT_TREES)
    parser.add_argument("--depth", type=_positive_int, default=DEFAULT_DEPTH)
    args = parser.parse_args()

    # We need the dataset first.
    if not THREAT_DATA_CSV.exists():
        raise SystemExit("No dataset found. Run `python scripts/generate_dataset.py` first.")

    _, m = train(n_estimators=args.trees, max_depth=args.depth)

    print("=" * 58)
    print("  SECTOR PREDICTION MODEL  —  Sabeeq (Aseel)")
    print("=" * 58)
    print(f"  Training samples    : {m.n_train}")
    print(f"  Test samples        : {m.n_test}")
    print("-" * 58)
    print(f"  Test accuracy       : {m.accuracy*100:6.2f}%")
    print(f"  Majority baseline   : {m.baseline_accuracy*100:6.2f}%")
    print(f"  Lift over baseline  : {m.lift_over_baseline*100:+6.2f} points")
    print(f"  Macro F1            : {m.macro_f1:6.3f}")
    print(f"  5-fold CV accuracy  : {m.cv_mean*100:6.2f}% +/- {m.cv_std*100:.2f}")
    print("-" * 58)
    print("  Feature importances:")
    for feat, imp in sorted(m.feature_importances.items(), key=lambda kv: -kv[1]):
        bar = "#" * int(imp * 50)
        print(f"    {feat:<20s} {imp:5.3f}  {bar}")
    print("=" * 58)

    # Warn if the model is almost the same as guessing.
    if m.lift_over_baseline < 0.05:
        print("\n  WARNING: the model barely beats a constant guess.")
        print("  Check the dataset generator before reporting.\n")
    else:
        print(f"\n  Model beats the baseline by {m.lift_over_baseline*100:.1f} points.\n")


if __name__ == "__main__":
    main()
