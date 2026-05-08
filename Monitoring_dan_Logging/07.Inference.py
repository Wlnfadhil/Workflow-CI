from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import uvicorn

from fastapi import FastAPI, HTTPException

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest
)

from pydantic import BaseModel, Field
from starlette.responses import Response

# =========================================================
# PATH CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TARGET_ENCODER_PATH = Path(
    os.getenv(
        "TARGET_ENCODER_PATH",
        str(
            BASE_DIR
            / "preprocessing"
            / "artifacts"
            / "encoders"
            / "target_label_encoder.pkl"
        )
    )
)

ENCODER_PATH = Path(
    os.getenv(
        "ENCODER_PATH",
        str(
            BASE_DIR
            / "preprocessing"
            / "artifacts"
            / "encoders"
            / "feature_label_encoders.pkl"
        )
    )
)

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(
            BASE_DIR
            / "preprocessing"
            / "artifacts"
            / "models"
            / "best_xgboost.pkl"
        )
    )
)

METADATA_PATH = Path(
    os.getenv(
        "PREPROCESSING_METADATA_PATH",
        str(
            BASE_DIR
            / "preprocessing"
            / "artifacts"
            / "metadata"
            / "preprocessing_metadata.json"
        )
    )
)
# =========================================================
# PROMETHEUS METRICS
# =========================================================

REQUEST_COUNT = Counter(
    "smsml_inference_requests_total",
    "Total inference requests received by API.",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "smsml_inference_latency_seconds",
    "Inference request latency in seconds.",
    ["endpoint"],
)

PREDICTION_COUNT = Counter(
    "smsml_prediction_count_total",
    "Total predictions by predicted grade.",
    ["grade"],
)

MODEL_LOADED = Gauge(
    "smsml_model_loaded",
    "Model loading status.",
)

FEATURE_COUNT = Gauge(
    "smsml_feature_count",
    "Number of feature columns used for inference.",
)

# =========================================================
# REQUEST SCHEMA
# =========================================================

class StudentPayload(BaseModel):

    study_hours_per_day: float = Field(..., ge=0)

    attendance_percentage: float = Field(
        ...,
        ge=0,
        le=100
    )

    assignment_score: float = Field(
        ...,
        ge=0,
        le=100
    )

    midterm_score: float = Field(
        ...,
        ge=0,
        le=100
    )

    final_exam_score: float = Field(
        ...,
        ge=0,
        le=100
    )

    participation_score: float = Field(
        ...,
        ge=0,
        le=100
    )

    sleep_hours: float = Field(..., ge=0)

    gender: str = "Male"

    internet_access: str = "Yes"

    extra_classes: str = "No"

    parent_education: str = "High School"

# =========================================================
# RESPONSE SCHEMA
# =========================================================

class PredictionResponse(BaseModel):

    prediction: str

    model_path: str

    latency_seconds: float

# =========================================================
# LOAD METADATA
# =========================================================

def load_metadata() -> dict[str, Any]:

    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            f"Metadata tidak ditemukan: {METADATA_PATH}"
        )

    with METADATA_PATH.open(
        encoding="utf-8"
    ) as file:

        return json.load(file)

# =========================================================
# LOAD MODEL
# =========================================================

def load_model() -> Any:

    if not MODEL_PATH.exists():

        MODEL_LOADED.set(0)

        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    MODEL_LOADED.set(1)

    return model

# =========================================================
# LOAD ASSETS
# =========================================================

metadata = load_metadata()

model = load_model()

feature_encoders = joblib.load(
    ENCODER_PATH
)

target_encoder = joblib.load(
    TARGET_ENCODER_PATH
)

feature_columns = metadata["feature_columns"]

numeric_columns = metadata["numeric_columns"]

categorical_columns = metadata["categorical_columns"]

FEATURE_COUNT.set(
    len(feature_columns)
)

print("\nModel berhasil dimuat.")
print(f"Model path : {MODEL_PATH}")

print("\nMetadata berhasil dimuat.")
print(f"Metadata path : {METADATA_PATH}")

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="SMSML Student Performance API",
    description="Inference API with Prometheus Monitoring",
    version="1.0.0",
)

# =========================================================
# PREPROCESS INPUT
# =========================================================

def payload_to_features(
    payload: StudentPayload
) -> pd.DataFrame:

    payload_dict = payload.dict()

    row: dict[str, Any] = {}

    for column in feature_columns:

        value = payload_dict[column]

        if column in categorical_columns:

            encoder = feature_encoders[column]

            value = encoder.transform(
                [value]
            )[0]

        row[column] = value

    return pd.DataFrame(
        [row],
        columns=feature_columns
    )

# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
def root():

    REQUEST_COUNT.labels(
        endpoint="/",
        status="success"
    ).inc()

    return {
        "service": (
            "SMSML Student Performance "
            "Inference API"
        ),
        "status": "running",
        "docs": "/docs",
        "metrics": "/metrics",
        "health": "/health"
    }

# =========================================================
# HEALTH ENDPOINT
# =========================================================

@app.get("/health")
def health():

    REQUEST_COUNT.labels(
        endpoint="/health",
        status="success"
    ).inc()

    return {
        "status": "ok",
        "model_loaded": int(
            MODEL_LOADED._value.get()
        ),
        "model_path": str(MODEL_PATH),
        "feature_count": len(feature_columns),
    }

# =========================================================
# PREDICT ENDPOINT
# =========================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    payload: StudentPayload
) -> PredictionResponse:

    start_time = time.perf_counter()

    endpoint = "/predict"

    try:

        with REQUEST_LATENCY.labels(
            endpoint=endpoint
        ).time():

            features = payload_to_features(
                payload
            )

            prediction_encoded = model.predict(
                features
            )[0]

            prediction = target_encoder.inverse_transform(
                [prediction_encoded]
            )[0]

        latency = (
            time.perf_counter()
            - start_time
        )

        REQUEST_COUNT.labels(
            endpoint=endpoint,
            status="success"
        ).inc()

        PREDICTION_COUNT.labels(
            grade=prediction
        ).inc()

        return PredictionResponse(
            prediction=str(prediction),
            model_path=str(MODEL_PATH),
            latency_seconds=round(
                latency,
                4
            ),
        )

    except Exception as error:

        REQUEST_COUNT.labels(
            endpoint=endpoint,
            status="error"
        ).inc()

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

# =========================================================
# METRICS ENDPOINT
# =========================================================

@app.get("/metrics")
def metrics():

    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("\nStarting FastAPI inference service...")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                "8000"
            )
        )
    )