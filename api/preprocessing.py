"""
API Preprocessing
===================
Turns a single raw transaction into the exact feature matrix the model
was trained on.

This does NOT reimplement feature engineering. It imports the fitted
pipeline pieces from src/feature_engineering.py directly:

  - apply_feature_engineering()  → same transformations as training
  - apply_transform()            → same fitted OrdinalEncoder, no refitting
  - load_artifacts()             → same bin edges / encoder fitted on TRAIN

If the EDA logic ever changes, it changes in ONE place (src/feature_engineering.py)
and both the training pipeline and the API pick it up automatically.
"""

import os
import pandas as pd

from src.feature_engineering import (
    apply_feature_engineering,
    apply_transform,
    load_artifacts,
    PRODUCTION_FEATURES,
)

ARTIFACTS_PATH = os.environ.get(
    "FEATURE_ARTIFACTS_PATH", "data/engineered/feature_artifacts.pkl"
)

_artifacts_cache = None


def get_artifacts() -> dict:
    """Load feature-engineering artifacts once and cache them in memory."""
    global _artifacts_cache
    if _artifacts_cache is None:
        _artifacts_cache = load_artifacts(ARTIFACTS_PATH)
    return _artifacts_cache


def reload_artifacts() -> dict:
    """Force a reload — useful after retraining without restarting the API."""
    global _artifacts_cache
    _artifacts_cache = None
    return get_artifacts()


def transaction_to_features(transaction: dict) -> pd.DataFrame:
    """Convert a single raw transaction dict into a model-ready feature row.

    Args:
        transaction : dict with keys Time, V1..V28, Amount (raw, unengineered)

    Returns:
        Single-row DataFrame matching the model's expected input columns.
    """
    artifacts = get_artifacts()

    df_raw = pd.DataFrame([transaction])

    # train_stats=artifacts["train_stats"] guarantees we reuse the TRAIN-fitted
    # amount_bin edges rather than refitting on this one row (which would be
    # both statistically meaningless and impossible with n=1).
    df_eng, _ = apply_feature_engineering(df_raw, train_stats=artifacts["train_stats"])

    X = df_eng[PRODUCTION_FEATURES]
    X_proc = apply_transform(X, artifacts["preprocessor"])
    return X_proc