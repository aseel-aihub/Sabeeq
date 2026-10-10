# ============================================================================
#  Sabeeq
#  Author : Aseel  (GitHub: aseel-aihub)
#  Copyright (c) 2026 Aseel. All rights reserved.
# ============================================================================
"""
config.py — all the settings of the project in one place.

NOTE:
All coordinates, facilities and numbers here are FICTIONAL (synthetic data).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project signature
# My signature, it is shown in the app and saved with the model.
# ---------------------------------------------------------------------------
PROJECT_NAME = "Sabeeq"
PROJECT_AUTHOR = "Aseel"
PROJECT_GITHUB = "https://github.com/aseel-aihub/Sabeeq"
PROJECT_COPYRIGHT = "© 2026 Aseel. All rights reserved."

# ---------------------------------------------------------------------------
# File paths
# Paths are built from this file location, so the project works
# from any folder on any computer.
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

FACILITIES_CSV = DATA_DIR / "facilities.csv"
THREAT_DATA_CSV = DATA_DIR / "synthetic_threat_data.csv"
MODEL_PATH = MODELS_DIR / "sector_classifier.joblib"
MODEL_METRICS_PATH = MODELS_DIR / "model_metrics.json"
# Small file with the model fingerprint (SHA-256) to detect tampering.
MODEL_INFO_PATH = MODELS_DIR / "sector_classifier.info.json"

# ---------------------------------------------------------------------------
# Simulation area (fictional rectangle on the map)
# ---------------------------------------------------------------------------
AO_LAT_MIN, AO_LAT_MAX = 17.05, 17.75
AO_LON_MIN, AO_LON_MAX = 42.40, 43.40
AO_CENTER = (17.35, 42.90)

# Area where the simulated threats start (south of the main area).
LAUNCH_LAT_MIN, LAUNCH_LAT_MAX = 16.95, 17.25
LAUNCH_LON_MIN, LAUNCH_LON_MAX = 42.40, 43.00

# ---------------------------------------------------------------------------
# Threat movement limits
# ---------------------------------------------------------------------------
SPEED_MIN_MS, SPEED_MAX_MS = 300.0, 900.0        # m/s
ANGLE_MIN_DEG, ANGLE_MAX_DEG = 30.0, 60.0        # launch angle
HEADING_MIN_DEG, HEADING_MAX_DEG = 0.0, 60.0     # 0 = north

# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
GRAVITY = 9.80665              # m/s^2
DEFAULT_DRAG_K = 8.0e-6        # air drag constant
INTEGRATION_DT = 0.05          # time step (s)
MAX_FLIGHT_TIME_S = 600.0      # safety limit
EARTH_RADIUS_KM = 6371.0088    # Earth radius in km

# ---------------------------------------------------------------------------
# Response rules thresholds
# ---------------------------------------------------------------------------
TTI_CRITICAL_S = 60.0     # less than 1 min  -> very urgent
TTI_HIGH_S = 120.0        # less than 2 min
TTI_MODERATE_S = 240.0    # less than 4 min

FACILITY_DANGER_RADIUS_KM = 3.0   # inside = danger
FACILITY_WATCH_RADIUS_KM = 8.0    # inside = watch

# ---------------------------------------------------------------------------
# Synthetic dataset settings
# ---------------------------------------------------------------------------
DATASET_SIZE = 3000
RANDOM_SEED = 42           # same seed = same results every time
# 8% of labels are changed on purpose to act like sensor errors,
# so the model can't get an unrealistic 100%.
LABEL_NOISE_RATE = 0.08

SECTORS = ["North Sector", "South Sector", "East Sector", "West Sector"]
