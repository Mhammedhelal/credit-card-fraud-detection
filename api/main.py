"""
FastAPI App — Credit Card Fraud Detection
============================================
Endpoints:
    POST /predict        Score a single raw transaction
    POST /predict/batch   Score a list of raw transactions
    GET  /health          Liveness/readiness + model metadata

Config (env vars):
    MODEL_PATH               default: models/trained_model.pkl
    FEATURE_ARTIFACTS_PATH   default: data/engineered/feature_artifacts.pkl
    PREDICTION_LOG_PATH      default: logs/predictions.jsonl

Run:
    uvicorn api.main:app --reload --port 8000
"""

import os
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    TransactionInput,
    BatchPredictionRequest,
    PredictionResponse,
    HealthResponse,
    ShapContribution,
)
from api.preprocessing import transaction_to_features
from api.model import get_model_bundle, predict as run_model_predict
from api.explain import explain_single

APP_VERSION = "1.0.0"
LOG_PATH = os.environ.get("PREDICTION_LOG_PATH", "logs/predictions.jsonl")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud_api")

app = FastAPI(
    title="Credit Card Fraud Detection API",
    version=APP_VERSION,
    description="Serves the trained XGBoost/LightGBM fraud model with SHAP explanations.",
)

# Loosen for local dev (Streamlit on a different port). Tighten before
# deploying — restrict allow_origins to the actual frontend origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)


def _log_prediction(request_id: str, transaction: dict, prediction: int,
                     fraud_probability: float, threshold: float) -> None:
    """Append one JSON line per prediction — timestamp, features, result.

    JSONL (not a DB) keeps this dependency-free; swap for a real sink
    (Kafka, a logging service, a table) when this goes past a prototype.
    """
    record = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": transaction,
        "prediction": prediction,
        "fraud_probability": fraud_probability,
        "threshold": threshold,
    }
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        # Logging failure should never take down a prediction response.
        logger.warning(f"Could not write prediction log: {e}")


def _score_transaction(transaction: TransactionInput, threshold: Optional[float]):
    """Shared path for single and batch prediction."""
    try:
        X = transaction_to_features(transaction.model_dump())
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Feature artifacts unavailable: {e}")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature engineering failed: {e}")

    try:
        y_pred, y_proba, used_threshold, bundle = run_model_predict(X, threshold)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {e}")

    proba = float(y_proba[0])
    pred = int(y_pred[0])
    confidence = "high" if abs(proba - used_threshold) > 0.3 else "low"
    shap_contribs = explain_single(bundle["model"], X, top_n=10)

    request_id = str(uuid.uuid4())
    _log_prediction(request_id, transaction.model_dump(), pred, proba, used_threshold)

    return PredictionResponse(
        prediction=pred,
        fraud_probability=proba,
        threshold_used=used_threshold,
        confidence=confidence,
        shap_explanation=[ShapContribution(**c) for c in shap_contribs],
        request_id=request_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.on_event("startup")
def _warm_start():
    """Load model + artifacts at boot so the first real request isn't slow.

    Deliberately doesn't crash the app if they're missing — /health will
    report 'degraded' and /predict will return a clear 503, which is more
    debuggable in a container than a boot-loop crash.
    """
    try:
        get_model_bundle()
        logger.info("Model loaded at startup.")
    except FileNotFoundError as e:
        logger.warning(f"Startup warning — model not loaded: {e}")


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        bundle = get_model_bundle()
        return HealthResponse(
            status="ok",
            model_type=bundle.get("model_type", "unknown"),
            model_version=APP_VERSION,
            threshold=bundle.get("threshold", 0.5),
        )
    except FileNotFoundError:
        return HealthResponse(
            status="degraded",
            model_type="not_loaded",
            model_version=APP_VERSION,
            threshold=0.0,
        )


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionInput, threshold: Optional[float] = None):
    """Score a single transaction. Pydantic validates every field before
    this function body even runs — a string in a numeric field returns a
    422 automatically, never reaches the model."""
    return _score_transaction(transaction, threshold)


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(request: BatchPredictionRequest):
    return [_score_transaction(tx, request.threshold) for tx in request.transactions]