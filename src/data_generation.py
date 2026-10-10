# ============================================================================
#  Sabeeq
#  Author : Aseel  (GitHub: aseel-aihub)
#  Copyright (c) 2026 Aseel. All rights reserved.
# ============================================================================
"""
data_generation.py — creates the synthetic (fake) training data.

In my first version the label (sector) was chosen randomly, so it had
no link to the inputs, and the model learned nothing (40.0% accuracy vs
40.3% for just guessing the most common sector).
Now every label comes from the physics:
    launch values -> trajectory -> impact point -> nearest facility -> its sector
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    ANGLE_MAX_DEG,
    ANGLE_MIN_DEG,
    DATASET_SIZE,
    HEADING_MAX_DEG,
    HEADING_MIN_DEG,
    LABEL_NOISE_RATE,
    LAUNCH_LAT_MAX,
    LAUNCH_LAT_MIN,
    LAUNCH_LON_MAX,
    LAUNCH_LON_MIN,
    RANDOM_SEED,
    SECTORS,
    SPEED_MAX_MS,
    SPEED_MIN_MS,
    THREAT_DATA_CSV,
)
from src.ballistics import simulate_trajectory
from src.facilities import load_facilities
from src.geo import destination_point, haversine_km

__all__ = ["FEATURE_COLUMNS", "generate_dataset", "build_and_save"]

# The 5 inputs the model learns from.
FEATURE_COLUMNS = [
    "launch_lat",        # start latitude
    "launch_lon",        # start longitude
    "speed_m_s",         # speed
    "launch_angle_deg",  # angle
    "heading_deg",       # direction
]


def generate_dataset(n_samples: int = DATASET_SIZE, seed: int = RANDOM_SEED,
                     label_noise: float = LABEL_NOISE_RATE) -> pd.DataFrame:
    """
    Build the dataset: the label is caused by the inputs, then we add a
    small amount of noise.
    """
    # --- Input checks ---
    if int(n_samples) < 1:
        raise ValueError("n_samples must be at least 1")
    if not 0.0 <= float(label_noise) < 1.0:
        raise ValueError("label_noise must be between 0 and 1")
    n_samples = int(n_samples)

    # Random generator with a seed -> same data every run.
    rng = np.random.default_rng(seed)
    facilities = load_facilities()

    fac_lat = facilities["lat"].to_numpy()
    fac_lon = facilities["lon"].to_numpy()

    # Step 1: random launch values
    launch_lat = rng.uniform(LAUNCH_LAT_MIN, LAUNCH_LAT_MAX, n_samples)
    launch_lon = rng.uniform(LAUNCH_LON_MIN, LAUNCH_LON_MAX, n_samples)
    speed = rng.uniform(SPEED_MIN_MS, SPEED_MAX_MS, n_samples)
    angle = rng.uniform(ANGLE_MIN_DEG, ANGLE_MAX_DEG, n_samples)
    heading = rng.uniform(HEADING_MIN_DEG, HEADING_MAX_DEG, n_samples)

    rows = []
    for i in range(n_samples):
        # Step 2: trajectory -> range
        traj = simulate_trajectory(float(speed[i]), float(angle[i]))
        # Step 3: impact point on the map
        imp_lat, imp_lon = destination_point(
            launch_lat[i], launch_lon[i], heading[i], traj.range_km
        )

        # Step 4: nearest facility
        distances = haversine_km(imp_lat, imp_lon, fac_lat, fac_lon)
        nearest = int(distances.argmin())
        fac = facilities.iloc[nearest]

        rows.append(
            {
                "launch_lat": launch_lat[i],
                "launch_lon": launch_lon[i],
                "speed_m_s": speed[i],
                "launch_angle_deg": angle[i],
                "heading_deg": heading[i],
                # --- extra info for analysis ---
                "range_km": traj.range_km,
                "time_of_flight_s": traj.time_of_flight_s,
                "impact_lat": float(imp_lat),
                "impact_lon": float(imp_lon),
                "nearest_facility_id": fac["facility_id"],
                "nearest_facility_km": float(distances[nearest]),
                # --- the answers the model must learn ---
                "target_facility": fac["name"],
                "target_sector": fac["sector"],
            }
        )

    df = pd.DataFrame(rows)

    # Keep the correct label too, so the tests can check the noise rate.
    df["true_sector"] = df["target_sector"]

    # Step 5: add label noise (fake sensor errors)
    if label_noise > 0:
        n_flip = int(round(label_noise * len(df)))
        flip_idx = rng.choice(len(df), size=n_flip, replace=False)
        for idx in flip_idx:
            alternatives = [s for s in SECTORS if s != df.at[idx, "target_sector"]]
            df.at[idx, "target_sector"] = str(rng.choice(alternatives))

    return df


def build_and_save(path=THREAT_DATA_CSV, **kwargs) -> pd.DataFrame:
    """Generate the dataset and save it as CSV."""
    df = generate_dataset(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
