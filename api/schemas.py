"""
Pydantic Schemas
=================
Request/response models for the fraud API.

TransactionInput mirrors the RAW columns feature_engineering.py expects
(Time, V1-V28, Amount) — NOT the engineered feature set. Engineering happens
inside api/preprocessing.py so the client only ever sends what the original
Kaggle-style dataset provides.

Pydantic gives us input validation for free: a string where a float is
expected, a missing field, or an out-of-range value all become a 422 with
a clear error message instead of a silent NaN propagating into the model.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class TransactionInput(BaseModel):
    """Raw transaction as it would appear in the source dataset."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Time": 406.0, "V1": -2.31, "V2": 1.95, "V3": -1.61, "V4": 3.99,
                "V5": -0.52, "V6": -1.43, "V7": -2.54, "V8": 1.39, "V9": -2.77,
                "V10": -2.77, "V11": 3.20, "V12": -2.90, "V13": -0.60, "V14": -4.29,
                "V15": 0.38, "V16": -1.14, "V17": -2.83, "V18": -0.02, "V19": 0.42,
                "V20": 0.13, "V21": 0.52, "V22": -0.15, "V23": 0.08, "V24": -0.21,
                "V25": -0.56, "V26": 0.32, "V27": 0.11, "V28": 0.30, "Amount": 0.0,
            }
        }
    )

    Time: float = Field(..., ge=0, description="Seconds elapsed since first transaction in dataset")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., ge=0, description="Transaction amount in dataset currency units")


class BatchPredictionRequest(BaseModel):
    transactions: List[TransactionInput]
    threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Override the model's saved threshold"
    )


class ShapContribution(BaseModel):
    feature: str
    value: float = Field(..., description="The feature's actual (engineered) value for this transaction")
    shap_value: float = Field(..., description="Contribution to the fraud log-odds; positive pushes toward fraud")


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = normal, 1 = fraud")
    fraud_probability: float
    threshold_used: float
    confidence: str = Field(..., description="'high' or 'low', based on distance from threshold")
    shap_explanation: List[ShapContribution]
    request_id: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' or 'degraded'")
    model_type: str
    model_version: str
    threshold: float