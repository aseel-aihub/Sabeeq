# ============================================================================
#  Sabeeq
#  Author : Aseel  (GitHub: aseel-aihub)
#  Copyright (c) 2026 Aseel. All rights reserved.
# ============================================================================
"""
Create the fictional facilities file and the training dataset.

Usage:
    python scripts/generate_dataset.py [--samples 3000] [--seed 42] [--noise 0.08]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add the project folder to the path so "config" and "src" can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATASET_SIZE, RANDOM_SEED  # noqa: E402
from src.data_generation import build_and_save  # noqa: E402
from src.facilities import write_facilities_csv  # noqa: E402


def _positive_int(text: str) -> int:
    """Only accept whole numbers bigger than 0."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def _noise_rate(text: str) -> float:
    """Only accept 0.0 to 0.99."""
    value = float(text)
    if not 0.0 <= value < 1.0:
        raise argparse.ArgumentTypeError("must be between 0.0 and 1.0")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=_positive_int, default=DATASET_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--noise", type=_noise_rate, default=None,
                        help="Label noise rate, 0.0-1.0 (default from config).")
    args = parser.parse_args()

    # Step 1: facilities file
    facilities = write_facilities_csv()
    print(f"Asset register written: {len(facilities)} facilities")

    # Step 2: dataset
    kwargs = {"n_samples": args.samples, "seed": args.seed}
    if args.noise is not None:
        kwargs["label_noise"] = args.noise

    df = build_and_save(**kwargs)

    # Step 3: short summary
    print(f"Dataset written: {len(df)} rows")
    print("\nSector distribution:")
    print(df["target_sector"].value_counts().to_string())

    corrupted = (df["target_sector"] != df["true_sector"]).mean()
    print(f"\nLabel noise actually applied: {corrupted*100:.1f}%")
    print(f"Mean distance to nearest facility: {df['nearest_facility_km'].mean():.2f} km")
    print(f"Range: {df['range_km'].min():.1f}-{df['range_km'].max():.1f} km")
    print("\nSabeeq — by Aseel")


if __name__ == "__main__":
    main()
