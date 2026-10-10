# ============================================================================
#  Sabeeq
#  Author : Aseel  (GitHub: aseel-aihub)
#  Copyright (c) 2026 Aseel. All rights reserved.
# ============================================================================
"""
facilities.py — list of protected facilities + find the nearest one (Objective 2).

NOTE:
ALL FACILITIES ARE FICTIONAL. Names and places are invented for the simulation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd

from config import FACILITIES_CSV, SECTORS
from src.geo import haversine_km

__all__ = [
    "Facility",
    "FACILITIES",
    "REQUIRED_COLUMNS",
    "facilities_dataframe",
    "write_facilities_csv",
    "load_facilities",
    "nearest_facility",
    "facilities_within",
]


@dataclass(frozen=True)
class Facility:
    """
    One protected facility. frozen=True means it can't be changed by mistake.
    """

    facility_id: str
    name: str
    sector: str
    facility_type: str
    criticality: int        # 1 = low importance, 5 = very important
    lat: float
    lon: float
    supply_node: str        # main power source
    # Backup power source. Only important facilities have one, on purpose,
    # so we can see who survives when a substation is lost.
    backup_node: str = ""


# ---------------------------------------------------------------------------
# The fictional facilities list (12 facilities, 3 in each sector)
# ---------------------------------------------------------------------------
FACILITIES: list[Facility] = [
    # ---- North Sector ----
    Facility("NTH-01", "Power Plant North Alpha", "North Sector", "Power Generation", 5, 17.58, 42.78, "Substation_North", "Substation_West"),
    Facility("NTH-02", "Desalination Plant North", "North Sector", "Water Supply", 5, 17.64, 43.02, "Substation_North", "Substation_East"),
    Facility("NTH-03", "Comms Relay North", "North Sector", "Communications", 3, 17.53, 43.18, "Substation_North", ""),
    # ---- South Sector ----
    Facility("STH-01", "Power Plant South Bravo", "South Sector", "Power Generation", 5, 17.12, 42.62, "Substation_South", "Substation_West"),
    Facility("STH-02", "Logistics Depot South", "South Sector", "Logistics", 3, 17.16, 42.95, "Substation_South", ""),
    Facility("STH-03", "Comms Relay South", "South Sector", "Communications", 3, 17.08, 43.08, "Substation_South", ""),
    # ---- East Sector ----
    Facility("EST-01", "Substation East", "East Sector", "Power Distribution", 4, 17.30, 43.16, "Substation_East", "Substation_North"),
    Facility("EST-02", "Water Treatment East", "East Sector", "Water Supply", 4, 17.42, 43.30, "Substation_East", "Substation_South"),
    Facility("EST-03", "Forward Operating Base East", "East Sector", "Military", 5, 17.24, 43.33, "Substation_East", "Substation_South"),
    # ---- West Sector ----
    Facility("WST-01", "Substation West", "West Sector", "Power Distribution", 4, 17.33, 42.55, "Substation_West", "Substation_South"),
    Facility("WST-02", "Fuel Storage West", "West Sector", "Fuel Storage", 4, 17.24, 42.45, "Substation_West", "Substation_North"),
    Facility("WST-03", "Radar Station West", "West Sector", "Sensor Site", 4, 17.46, 42.62, "Substation_West", "Substation_North"),
]

# Columns the CSV must have. Used to check the file is not broken.
REQUIRED_COLUMNS = [
    "facility_id", "name", "sector", "facility_type",
    "criticality", "lat", "lon", "supply_node", "backup_node",
]


def facilities_dataframe() -> pd.DataFrame:
    """The list above as a table (DataFrame)."""
    return pd.DataFrame([asdict(f) for f in FACILITIES])


def write_facilities_csv(path=FACILITIES_CSV) -> pd.DataFrame:
    """Save the list to CSV so all parts use the same data."""
    df = facilities_dataframe()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _is_valid_register(df: pd.DataFrame) -> bool:
    """
    Check the CSV data is correct before we trust it.

    If someone edits the CSV and breaks it (missing column, wrong
    number, repeated id), the app will not crash or give wrong results.
    """
    if df.empty or any(col not in df.columns for col in REQUIRED_COLUMNS):
        return False
    try:
        lat = pd.to_numeric(df["lat"], errors="raise")
        lon = pd.to_numeric(df["lon"], errors="raise")
        crit = pd.to_numeric(df["criticality"], errors="raise")
    except (ValueError, TypeError):
        return False

    checks = [
        df["facility_id"].notna().all() and df["facility_id"].is_unique,  # unique ids
        lat.between(-90, 90).all() and lon.between(-180, 180).all(),     # real coordinates
        crit.between(1, 5).all(),                                        # criticality 1..5
        df["sector"].isin(SECTORS).all(),                                # known sectors
        df["supply_node"].notna().all(),                                 # has a power source
    ]
    return all(bool(c) for c in checks)


def load_facilities(path=FACILITIES_CSV) -> pd.DataFrame:
    """
    Load facilities from CSV. If the file is missing or broken,
    use the list written in the code instead.
    """
    if path.exists():
        try:
            # keep_default_na=False so an empty backup_node stays "" (not NaN).
            df = pd.read_csv(path, keep_default_na=False)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            df = None
        if df is not None and _is_valid_register(df):
            df["lat"] = df["lat"].astype(float)
            df["lon"] = df["lon"].astype(float)
            df["criticality"] = df["criticality"].astype(int)
            df["backup_node"] = df["backup_node"].astype(str)
            return df
    return facilities_dataframe()


def _check_point(lat, lon) -> tuple[float, float]:
    """Make sure the point is a real coordinate."""
    lat, lon = float(lat), float(lon)
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise ValueError("lat/lon must be finite numbers")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("lat/lon out of range")
    return lat, lon


def nearest_facility(lat: float, lon: float, facilities: pd.DataFrame | None = None):
    """
    Return (facility_row, distance_km) of the closest facility.

    All distances are calculated in one vectorised call (no loop).
    """
    lat, lon = _check_point(lat, lon)
    df = load_facilities() if facilities is None else facilities

    distances = haversine_km(lat, lon, df["lat"].to_numpy(), df["lon"].to_numpy())
    idx = int(distances.argmin())  # index of the smallest distance

    return df.iloc[idx], float(distances[idx])


def facilities_within(lat: float, lon: float, radius_km: float,
                      facilities: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    All facilities inside a circle of `radius_km`, nearest first.
    """
    lat, lon = _check_point(lat, lon)
    df = (load_facilities() if facilities is None else facilities).copy()

    df["distance_km"] = haversine_km(lat, lon, df["lat"].to_numpy(), df["lon"].to_numpy())

    return df[df["distance_km"] <= radius_km].sort_values("distance_km").reset_index(drop=True)
