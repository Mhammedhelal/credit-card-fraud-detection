"""
SHAP Explanation
=================
Per-prediction SHAP values for the /predict response.

`shap` is imported lazily (inside the functions, not at module top level)
so that a missing `shap` install degrades gracefully — /predict still
returns a prediction, just with an empty shap_explanation list, instead of
crashing the whole API at import time.

The TreeExplainer is cached per model instance (keyed by id(model)) since
building it is the expensive part; computing shap_values() for a single
row is cheap.
"""

import pandas as pd

_explainer_cache = None
_explainer_model_id = None


def _get_explainer(model):
    import shap  # lazy import — keeps shap optional at API startup

    global _explainer_cache, _explainer_model_id
    if _explainer_cache is None or _explainer_model_id != id(model):
        _explainer_cache = shap.TreeExplainer(model)
        _explainer_model_id = id(model)
    return _explainer_cache


def explain_single(model, X_row: pd.DataFrame, top_n: int = 10) -> list[dict]:
    """Return the top_n features by |SHAP value| for a single-row prediction.

    Args:
        model  : Trained XGBoost/LightGBM model (TreeExplainer-compatible)
        X_row  : Single-row, already-engineered feature DataFrame
        top_n  : How many top contributing features to return

    Returns:
        List of {"feature", "value", "shap_value"} dicts, sorted by
        descending absolute SHAP contribution. Empty list if shap isn't
        installed or the model type isn't tree-based.
    """
    try:
        explainer = _get_explainer(model)
    except ImportError:
        return []
    except Exception:
        # Non-tree models (e.g. logistic/knn) aren't TreeExplainer-compatible.
        # Fail soft — the prediction itself is still valid and useful.
        return []

    shap_values = explainer.shap_values(X_row)

    # Binary classifiers may return a list [class0_shap, class1_shap]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    row_values = shap_values[0]
    contributions = [
        {
            "feature": col,
            "value": float(X_row.iloc[0][col]),
            "shap_value": float(val),
        }
        for col, val in zip(X_row.columns, row_values)
    ]
    contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
    return contributions[:top_n]