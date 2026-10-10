# ============================================================================
#  Sabeeq
#  Author : Aseel  (GitHub: aseel-aihub)
#  Copyright (c) 2026 Aseel. All rights reserved.
# ============================================================================
"""
pattern_model.py — the AI model that predicts the target sector (Objective 3).

Main ideas:
1. Training only happens when we call train(), not when the file is imported.
2. We compare the model with a "baseline" (always guessing the most common
   sector). If the model can't beat it, it learned nothing.
3. Security — the saved model file is checked with a SHA-256 fingerprint
   before loading, because loading a changed .joblib file can run harmful code.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import cross_val_score, train_test_split

from config import (
    MODEL_METRICS_PATH,
    MODEL_PATH,
    PROJECT_AUTHOR,
    RANDOM_SEED,
    THREAT_DATA_CSV,
)
from src.data_generation import FEATURE_COLUMNS, build_and_save

__all__ = [
    "ModelMetrics",
    "train",
    "load_model",
    "load_metrics",
    "predict_sector",
    "file_sha256",
    "info_path_for",
]

# Settings of the Random Forest (written once, used everywhere).
DEFAULT_TREES = 200
DEFAULT_DEPTH = 14


@dataclass
class ModelMetrics:
    """
    All the evaluation results of the model.
    """

    accuracy: float            # test accuracy
    baseline_accuracy: float   # accuracy of just guessing
    macro_f1: float
    cv_mean: float             # cross-validation average
    cv_std: float
    n_train: int
    n_test: int
    labels: list
    confusion: list            # confusion matrix
    report: dict
    feature_importances: dict  # which input matters most

    @property
    def lift_over_baseline(self) -> float:
        """How much better than guessing."""
        return self.accuracy - self.baseline_accuracy


def info_path_for(model_path):
    """
    The fingerprint file always sits next to its model file
    (sector_classifier.joblib -> sector_classifier.info.json).
    """
    return model_path.with_suffix(".info.json")


def file_sha256(path) -> str:
    """
    Calculate the SHA-256 fingerprint of a file.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        # Read in small pieces so big files don't fill the memory.
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def train(
    data_path=THREAT_DATA_CSV,
    model_path=MODEL_PATH,
    metrics_path=MODEL_METRICS_PATH,
    n_estimators: int = DEFAULT_TREES,
    # Depth 14 gives the same accuracy as unlimited depth but a smaller file.
    max_depth: int | None = DEFAULT_DEPTH,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
    save: bool = True,
) -> tuple[RandomForestClassifier, ModelMetrics]:
    """
    Train the model, test it, and save it.
    """
    # If there is no dataset yet, create it first.
    if not data_path.exists():
        build_and_save(path=data_path)

    df = pd.read_csv(data_path)

    # Make sure the needed columns exist before training.
    missing = [c for c in FEATURE_COLUMNS + ["target_sector"] if c not in df.columns]
    if missing:
        raise ValueError(f"dataset is missing columns: {missing}")

    X = df[FEATURE_COLUMNS]
    y = df["target_sector"]

    # Split 80% train / 20% test. stratify keeps the same sector ratios.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=2,
        class_weight="balanced_subsample",  # fair to small sectors
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Baseline = always guess the most common sector.
    majority_class = y_train.value_counts().idxmax()
    baseline = accuracy_score(y_test, np.full(len(y_test), majority_class))

    # 5-fold cross-validation to make sure the result is stable.
    cv = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    labels = sorted(y.unique().tolist())

    metrics = ModelMetrics(
        accuracy=float(accuracy_score(y_test, y_pred)),
        baseline_accuracy=float(baseline),
        macro_f1=float(f1_score(y_test, y_pred, average="macro")),
        cv_mean=float(cv.mean()),
        cv_std=float(cv.std()),
        n_train=int(len(X_train)),
        n_test=int(len(X_test)),
        labels=labels,
        confusion=confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        report=classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        feature_importances=dict(
            zip(FEATURE_COLUMNS, [float(v) for v in model.feature_importances_])
        ),
    )

    if save:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        metrics_path.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
        # Save the fingerprint + library version + my name with the model.
        info = {
            "sha256": file_sha256(model_path),
            "sklearn_version": sklearn.__version__,
            "author": PROJECT_AUTHOR,
        }
        info_path_for(model_path).write_text(json.dumps(info, indent=2), encoding="utf-8")

    return model, metrics


def _model_is_trusted(model_path) -> bool:
    """
    True only if the model file matches its saved fingerprint and was made
    with the same scikit-learn version.
    """
    info_path = info_path_for(model_path)
    if not (model_path.exists() and info_path.exists()):
        return False
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        info.get("sha256") == file_sha256(model_path)
        and info.get("sklearn_version") == sklearn.__version__
    )


def load_model(model_path=MODEL_PATH) -> RandomForestClassifier:
    """
    Load the saved model. If it is missing, changed, or from another
    scikit-learn version, we train a new clean one instead.
    """
    if _model_is_trusted(model_path):
        return joblib.load(model_path)
    model, _ = train(model_path=model_path)
    return model


def load_metrics(metrics_path=MODEL_METRICS_PATH) -> ModelMetrics | None:
    """Read saved results; return None if missing or broken."""
    if not metrics_path.exists():
        return None
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        # Use only the fields we know (ignore anything extra).
        known = {f.name for f in fields(ModelMetrics)}
        return ModelMetrics(**{k: v for k, v in data.items() if k in known})
    except (OSError, ValueError, TypeError):
        return None


def predict_sector(model, launch_lat, launch_lon, speed, angle_deg, heading_deg):
    """
    Predict the sector AND the probability of every sector.

    The probability matters: 41% and 95% need very different responses.
    """
    values = [launch_lat, launch_lon, speed, angle_deg, heading_deg]
    values = [float(v) for v in values]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("all inputs must be finite numbers")

    row = pd.DataFrame([values], columns=FEATURE_COLUMNS)

    label = str(model.predict(row)[0])
    proba = {str(c): float(p) for c, p in zip(model.classes_, model.predict_proba(row)[0])}

    return label, proba
