# ============================================================================
#  Sabeeq
#  Author : Aseel  (GitHub: aseel-aihub)
#  Copyright (c) 2026 Aseel. All rights reserved.
# ============================================================================
"""
ballistics.py — predicts the flight path of a falling object (Objective 1).

The simple school formula R = v^2 * sin(2θ) / g ignores air, so it gives
a range that is too long. Here we calculate the path step by step and
include air drag. This also gives us:
  - time of flight  -> how much time we have to evacuate
  - the full path   -> to draw it in the dashboard
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from config import DEFAULT_DRAG_K, GRAVITY, INTEGRATION_DT, MAX_FLIGHT_TIME_S

__all__ = ["TrajectoryResult", "simulate_trajectory", "vacuum_range_km"]


@dataclass
class TrajectoryResult:
    """
    The result of one trajectory calculation.
    """

    range_km: float            # distance to impact
    time_of_flight_s: float    # seconds in the air
    apogee_km: float           # highest point
    impact_speed_ms: float     # speed when it hits
    # List of (distance_km, altitude_km) points used for the chart.
    path: list = field(default_factory=list, repr=False)

    @property
    def range_m(self) -> float:
        return self.range_km * 1000.0


def _check_number(name: str, value) -> float:
    """
    Make sure the input is a real number (not text, not NaN, not infinity).
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def vacuum_range_km(speed_ms: float, angle_deg: float) -> float:
    """
    School formula without air: R = v^2 * sin(2θ) / g.
    """
    theta = np.radians(angle_deg)
    return float((speed_ms ** 2) * np.sin(2.0 * theta) / GRAVITY / 1000.0)


def simulate_trajectory(
    speed_ms: float,
    angle_deg: float,
    drag_k: float = DEFAULT_DRAG_K,
    dt: float = INTEGRATION_DT,
    max_time_s: float = MAX_FLIGHT_TIME_S,
) -> TrajectoryResult:
    """
    Calculate the 2-D path with gravity + air drag.

        acceleration = gravity  -  k * |v| * v

    We use the RK2 (midpoint) method. It is more accurate than the
    simple Euler method with the same step size.

    Parameters:
        speed_ms  : start speed (m/s)
        angle_deg : launch angle (degrees)
        drag_k    : air drag (0 = no air)
        dt        : time step (s)
        max_time_s: safety time limit
    """
    # --- Input validation --------------------------
    # Bad inputs are rejected early with a clear message.
    speed_ms = _check_number("speed_ms", speed_ms)
    angle_deg = _check_number("angle_deg", angle_deg)
    drag_k = _check_number("drag_k", drag_k)
    dt = _check_number("dt", dt)
    max_time_s = _check_number("max_time_s", max_time_s)

    if speed_ms <= 0:
        raise ValueError("speed_ms must be positive")
    if not 0.0 < angle_deg < 90.0:
        raise ValueError("angle_deg must be strictly between 0 and 90")
    if drag_k < 0:
        raise ValueError("drag_k must be non-negative")
    if dt <= 0 or max_time_s <= 0:
        raise ValueError("dt and max_time_s must be positive")

    # --- Start values ---------------------------------
    theta = np.radians(angle_deg)
    pos = np.array([0.0, 0.0])  # (distance, altitude) in metres
    vel = np.array([speed_ms * np.cos(theta), speed_ms * np.sin(theta)])

    def acceleration(v: np.ndarray) -> np.ndarray:
        # Gravity pulls down; drag pushes against the movement.
        a = np.array([0.0, -GRAVITY])
        if drag_k > 0.0:
            a = a - drag_k * np.linalg.norm(v) * v
        return a

    path = [(0.0, 0.0)]
    t = 0.0
    prev_pos = pos.copy()
    prev_vel = vel.copy()
    hit_ground = False

    # --- Step-by-step loop ---------------------------
    while t < max_time_s:
        prev_pos, prev_vel = pos.copy(), vel.copy()

        # RK2 (midpoint method)
        a1 = acceleration(vel)
        v_mid = vel + 0.5 * dt * a1
        a2 = acceleration(v_mid)

        pos = pos + dt * v_mid
        vel = vel + dt * a2
        t += dt

        # Stop when the object reaches the ground.
        if pos[1] <= 0.0:
            hit_ground = True
            break

        path.append((pos[0] / 1000.0, pos[1] / 1000.0))

    # If it never landed, the result would be wrong, so we stop here.
    if not hit_ground:
        raise ValueError("object did not land within max_time_s; increase the limit")

    # Find the exact ground point between the last two steps
    # (linear interpolation), so the range is more accurate.
    if prev_pos[1] != pos[1]:
        frac = prev_pos[1] / (prev_pos[1] - pos[1])
    else:
        frac = 0.0
    impact_pos = prev_pos + frac * (pos - prev_pos)
    impact_vel = prev_vel + frac * (vel - prev_vel)
    impact_time = t - dt + frac * dt

    path.append((impact_pos[0] / 1000.0, 0.0))
    altitudes = [p[1] for p in path]

    return TrajectoryResult(
        range_km=float(impact_pos[0] / 1000.0),
        time_of_flight_s=float(impact_time),
        apogee_km=float(max(altitudes)),
        impact_speed_ms=float(np.linalg.norm(impact_vel)),
        path=path,
    )
