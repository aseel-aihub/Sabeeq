# ============================================================================
#  Sabeeq
#  Author : Aseel  (GitHub: aseel-aihub)
#  Copyright (c) 2026 Aseel. All rights reserved.
# ============================================================================
"""
response_rules.py — gives protective recommendations (Objective 4).

Why rules and not another AI model?
A decision like "evacuate" must be explainable. With rules, we can always
show exactly WHY the system said it. All limits are in config.py.

NOTE: This is advisory text for a simulation only. Nothing is controlled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from config import (
    FACILITY_DANGER_RADIUS_KM,
    FACILITY_WATCH_RADIUS_KM,
    TTI_CRITICAL_S,
    TTI_HIGH_S,
    TTI_MODERATE_S,
)
from src.facilities import facilities_within, nearest_facility

__all__ = ["ThreatAssessment", "assess_threat", "THREAT_LEVELS"]

# Threat levels from lowest to highest.
THREAT_LEVELS = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
_LEVEL_RANK = {name: i for i, name in enumerate(THREAT_LEVELS)}


@dataclass
class ThreatAssessment:
    """The final result of the assessment."""

    threat_level: str
    confidence: float
    time_to_impact_s: float
    nearest_facility_name: str
    nearest_facility_id: str
    nearest_facility_km: float
    nearest_facility_criticality: int
    predicted_sector: str
    facilities_at_risk: pd.DataFrame = field(repr=False)
    actions: list = field(default_factory=list)      # what to do
    reasoning: list = field(default_factory=list)    # why

    @property
    def is_direct_hit(self) -> bool:
        """True if impact is inside the danger radius."""
        return self.nearest_facility_km <= FACILITY_DANGER_RADIUS_KM


def _escalate(current: str, candidate: str) -> str:
    """
    The level can only go UP, never down.
    """
    return candidate if _LEVEL_RANK[candidate] > _LEVEL_RANK[current] else current


def assess_threat(
    impact_lat: float,
    impact_lon: float,
    time_to_impact_s: float,
    predicted_sector: str,
    sector_confidence: float,
) -> ThreatAssessment:
    """
    Turn the predicted impact into a threat level + a list of actions + reasons.
    """
    # --- Input checks -------------------------------
    time_to_impact_s = float(time_to_impact_s)
    sector_confidence = float(sector_confidence)
    if not math.isfinite(time_to_impact_s) or time_to_impact_s < 0:
        raise ValueError("time_to_impact_s must be a non-negative number")
    if not math.isfinite(sector_confidence) or not 0.0 <= sector_confidence <= 1.0:
        raise ValueError("sector_confidence must be between 0 and 1")

    facility, distance_km = nearest_facility(impact_lat, impact_lon)
    at_risk = facilities_within(impact_lat, impact_lon, FACILITY_WATCH_RADIUS_KM)

    level = "LOW"
    reasoning: list[str] = []

    # ---- Rule 1: distance to a facility -----
    if distance_km <= FACILITY_DANGER_RADIUS_KM:
        level = _escalate(level, "HIGH")
        reasoning.append(
            f"Predicted impact is {distance_km:.1f} km from {facility['name']}, "
            f"inside the {FACILITY_DANGER_RADIUS_KM} km danger radius."
        )
    elif distance_km <= FACILITY_WATCH_RADIUS_KM:
        level = _escalate(level, "MODERATE")
        reasoning.append(
            f"Predicted impact is {distance_km:.1f} km from {facility['name']}, "
            f"inside the {FACILITY_WATCH_RADIUS_KM} km watch radius."
        )
    else:
        reasoning.append(
            f"Nearest protected asset ({facility['name']}) is {distance_km:.1f} km "
            "away — outside the watch radius."
        )

    # ---- Rule 2: how important the facility is ----
    criticality = int(facility["criticality"])
    if criticality >= 5 and distance_km <= FACILITY_WATCH_RADIUS_KM:
        level = _escalate(level, "CRITICAL")
        reasoning.append(
            f"{facility['name']} is criticality {criticality}/5 "
            f"({facility['facility_type']}); loss would interrupt an essential service."
        )
    elif criticality >= 4 and distance_km <= FACILITY_DANGER_RADIUS_KM:
        level = _escalate(level, "HIGH")
        reasoning.append(
            f"{facility['name']} is criticality {criticality}/5 and lies in the danger radius."
        )

    # ---- Rule 3: time left before impact -------
    if time_to_impact_s <= TTI_CRITICAL_S:
        if distance_km <= FACILITY_WATCH_RADIUS_KM:
            level = _escalate(level, "CRITICAL")
        reasoning.append(
            f"Only {time_to_impact_s:.0f} s to impact — below the "
            f"{TTI_CRITICAL_S:.0f} s immediate-action threshold."
        )
    elif time_to_impact_s <= TTI_HIGH_S:
        if distance_km <= FACILITY_DANGER_RADIUS_KM:
            level = _escalate(level, "CRITICAL")
        reasoning.append(f"{time_to_impact_s:.0f} s to impact — limited warning time.")
    elif time_to_impact_s <= TTI_MODERATE_S:
        reasoning.append(
            f"{time_to_impact_s:.0f} s to impact — sufficient time for a staged response."
        )
    else:
        reasoning.append(f"{time_to_impact_s:.0f} s to impact — ample warning time.")

    # ---- Rule 4: how sure the model is -----------
    if sector_confidence < 0.50:
        reasoning.append(
            f"Sector prediction confidence is only {sector_confidence*100:.0f}% — "
            "treat the sector attribution as provisional and widen the alert."
        )
    elif sector_confidence >= 0.80:
        reasoning.append(
            f"Sector prediction confidence is {sector_confidence*100:.0f}% — "
            f"{predicted_sector} attribution is well supported."
        )

    # ---- Actions, most urgent first ------------
    if level == "CRITICAL":
        actions = [
            f"Issue immediate impact-area warning for {facility['name']} ({facility['facility_id']}).",
            "Initiate personnel evacuation to hardened shelters in the predicted footprint.",
            f"Alert {predicted_sector} response command and place crews on standby.",
            "Pre-stage the self-healing reroute so supply transfers the moment damage is confirmed.",
            "Escalate to the national operations centre.",
        ]
    elif level == "HIGH":
        actions = [
            f"Issue a warning for {facility['name']} ({facility['facility_id']}).",
            "Move non-essential personnel clear of the predicted footprint.",
            f"Place {predicted_sector} response crews on alert.",
            "Verify alternate supply routing is available for the affected node.",
        ]
    elif level == "MODERATE":
        actions = [
            f"Notify the {facility['name']} duty officer and hold at readiness.",
            f"Increase monitoring across {predicted_sector}.",
            "Confirm backup supply paths are healthy.",
        ]
    else:
        actions = [
            "Log the track and continue routine monitoring.",
            "No protective action required at this time.",
        ]

    # If more than one facility is near, warn all of them.
    if len(at_risk) > 1:
        actions.insert(
            0,
            f"{len(at_risk)} facilities lie within {FACILITY_WATCH_RADIUS_KM} km "
            "of the predicted impact — widen the alert to all of them.",
        )

    return ThreatAssessment(
        threat_level=level,
        confidence=sector_confidence,
        time_to_impact_s=time_to_impact_s,
        nearest_facility_name=str(facility["name"]),
        nearest_facility_id=str(facility["facility_id"]),
        nearest_facility_km=distance_km,
        nearest_facility_criticality=criticality,
        predicted_sector=predicted_sector,
        facilities_at_risk=at_risk,
        actions=actions,
        reasoning=reasoning,
    )
