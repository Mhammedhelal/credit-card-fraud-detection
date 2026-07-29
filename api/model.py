"""
Model Loading & Inference
===========================
Loads the trained model bundle from MODEL_PATH (env var) and exposes a
thin predict() wrapper around src.inference_core.run_inference.

Deliberately does NOT import src.testing — that module pulls in matplotlib
and shap at import time for its reporting functions, which would slow down
every API cold start. See src/inference_core.py for why.
"""

import os
import joblib

from src.inference_core import run_inference

MODEL_PATH = os.environ.get("MODEL_PATH", "models/trained_model.pkl")

_bundle_cache = None


def get_model_bundle() -> dict:
    """Load and cache the model bundle saved by training.py.

    Bundle contains: model, model_type, threshold, feature_names,
    scale_pos_weight, metadata.
    """
    global _bundle_cache
    if _bundle_cache is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at MODEL_PATH={MODEL_PATH}. "
                f"Set the MODEL_PATH env var or run training.py first."
            )
        _bundle_cache = joblib.load(MODEL_PATH)
    return _bundle_cache


def reload_model() -> dict:
    """Force a reload — useful after retraining without restarting the API."""
    global _bundle_cache
    _bundle_cache = None
    return get_model_bundle()


def predict(X, threshold: float | None = None):
    """Run inference using the loaded model.

    Args:
        X         : Preprocessed feature matrix (single row for /predict)
        threshold : Optional override; defaults to the threshold saved with
                    the model (chosen via the PR-curve strategy in training.py)

    Returns:
        y_pred, y_proba, threshold_used, bundle
    """
    bundle = get_model_bundle()
    model = bundle["model"]
    threshold_used = threshold if threshold is not None else bundle.get("threshold", 0.5)
    y_pred, y_proba = run_inference(model, X, threshold_used)
    return y_pred, y_proba, threshold_used, bundle