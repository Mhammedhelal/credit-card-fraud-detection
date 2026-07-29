"""
Inference Core
===============
Single source of truth for turning model probabilities into predictions.
"""

import numpy as np
import pandas as pd


def run_inference(model, X: pd.DataFrame, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Return binary predictions and fraud probabilities.

    Args:
        model     : Any fitted classifier exposing predict_proba
        X         : Feature matrix, already engineered + preprocessed
        threshold : Classification threshold applied to the fraud probability

    Returns:
        y_pred  : Binary predictions (0/1)
        y_proba : Raw fraud probability scores
    """
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return y_pred, y_proba