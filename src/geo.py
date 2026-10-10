# ============================================================================
#  Sabeeq
#  Author : Aseel  (GitHub: aseel-aihub)
#  Copyright (c) 2026 Aseel. All rights reserved.
# ============================================================================
"""
geo.py — map math (distance, direction, destination point).

We treat the Earth as a sphere. For short distances (< 100 km) the
error is about 0.5%, which is small enough for this project.
"""

from __future__ import annotations

import numpy as np

from config import EARTH_RADIUS_KM

__all__ = ["haversine_km", "initial_bearing_deg", "destination_point"]


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Distance in km between two points on Earth (Haversine formula).

    Works with single numbers OR numpy arrays, so we can measure the
    distance to all facilities in one step (faster than a loop).
    """
    # Convert degrees to radians because numpy trig uses radians.
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    # clip keeps the value between 0 and 1 (avoids tiny rounding errors).
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    return EARTH_RADIUS_KM * c


def initial_bearing_deg(lat1, lon1, lat2, lon2):
    """
    Compass direction (0-360) from point 1 to point 2. 0 = north.
    """
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(np.asarray(lon2) - np.asarray(lon1))

    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    # % 360 keeps the answer between 0 and 360.
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def destination_point(lat, lon, bearing_deg, distance_km):
    """
    Where do we arrive if we move `distance_km` from (lat, lon) in a direction?

    This is better than the simple "divide by 111 km" method, because
    that method is not accurate for longitude away from the equator.
    """
    ang = np.asarray(distance_km, dtype=float) / EARTH_RADIUS_KM  # angle on Earth
    brg = np.radians(bearing_deg)
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)

    sin_lat2 = np.sin(lat1) * np.cos(ang) + np.cos(lat1) * np.sin(ang) * np.cos(brg)
    lat2 = np.arcsin(np.clip(sin_lat2, -1.0, 1.0))

    lon2 = lon1 + np.arctan2(
        np.sin(brg) * np.sin(ang) * np.cos(lat1),
        np.cos(ang) - np.sin(lat1) * sin_lat2,
    )

    # Keep longitude between -180 and 180.
    lon2_deg = (np.degrees(lon2) + 540.0) % 360.0 - 180.0

    return np.degrees(lat2), lon2_deg
