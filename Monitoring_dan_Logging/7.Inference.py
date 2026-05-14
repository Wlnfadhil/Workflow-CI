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
# PATH CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACT_DIR = (
    BASE_DIR
    / "preprocessing"
    / "artifacts"
)

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(
            ARTIFACT_DIR
            / "models"
            / "best_model.joblib"
        )
    )
)

FEATURE_ENCODER_PATH = Path(
    os.getenv(
        "FEATURE_ENCODER_PATH",
        str(
            ARTIFACT_DIR
            / "encoders"
            / "feature_label_encoders.pkl"
        )
    )
)

TARGET_ENCODER_PATH = Path(
    os.getenv(
        "TARGET_ENCODER_PATH",
        str(
            ARTIFACT_DIR
            / "encoders"
            / "target_label_encoder.pkl"
        )
    )
)

SCALER_PATH = Path(
    os.getenv(
        "SCALER_PATH",
        str(
            ARTIFACT_DIR
            / "encoders"
            / "scaler.joblib"
        )
    )
)

METADATA_PATH = Path(
    os.getenv(
        "METADATA_PATH",
        str(
            ARTIFACT_DIR
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
    "Total inference requests.",
    ["endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "smsml_inference_latency_seconds",
    "Inference latency in seconds.",
    ["endpoint"]
)

PREDICTION_COUNT = Counter(
    "smsml_prediction_count_total",
    "Total predictions by class.",
    ["grade"]
)

MODEL_LOADED = Gauge(
    "smsml_model_loaded",
    "Model loading status."
)

FEATURE_COUNT = Gauge(
    "smsml_feature_count",
    "Total feature columns."
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

    latency_seconds: float

    model_path: str

# =========================================================
# LOAD METADATA
# =========================================================

def load_metadata() -> dict[str, Any]:

    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            f"Metadata tidak ditemukan:\n{METADATA_PATH}"
        )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    return metadata

# =========================================================
# LOAD MODEL
# =========================================================

def load_model():

    if not MODEL_PATH.exists():

        MODEL_LOADED.set(0)

        raise FileNotFoundError(
            f"Model tidak ditemukan:\n{MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    MODEL_LOADED.set(1)

    return model

# =========================================================
# LOAD ASSETS
# =========================================================

print("\nLoading artifacts...")

metadata = load_metadata()

model = load_model()

feature_encoders = joblib.load(
    FEATURE_ENCODER_PATH
)

target_encoder = joblib.load(
    TARGET_ENCODER_PATH
)

scaler = joblib.load(
    SCALER_PATH
)

feature_columns = metadata["feature_columns"]

numeric_columns = metadata["numeric_columns"]

categorical_columns = metadata["categorical_columns"]

FEATURE_COUNT.set(
    len(feature_columns)
)

print("\nArtifacts berhasil dimuat.")

print(f"Model path : {MODEL_PATH}")
print(f"Metadata path : {METADATA_PATH}")

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="SMSML Student Performance API",
    description="Inference API with Prometheus Monitoring",
    version="1.0.0"
)

# =========================================================
# PREPROCESS INPUT
# =========================================================

def payload_to_features(
    payload: StudentPayload
) -> pd.DataFrame:

    payload_dict = payload.model_dump()

    row: dict[str, Any] = {}

    for column in feature_columns:

        value = payload_dict[column]

        # categorical encoding
        if column in categorical_columns:

            encoder = feature_encoders[column]

            value = encoder.transform(
                [value]
            )[0]

        row[column] = value

    features = pd.DataFrame(
        [row],
        columns=feature_columns
    )

    # scaling numeric features
    features[numeric_columns] = scaler.transform(
        features[numeric_columns]
    )

    return features

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
        "service":
            "SMSML Student Performance API",

        "status":
            "running",

        "docs":
            "/docs",

        "health":
            "/health",

        "metrics":
            "/metrics"
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

        "status":
            "healthy",

        "model_loaded":
            True,

        "model_path":
            str(MODEL_PATH),

        "feature_count":
            len(feature_columns)
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
):

    endpoint = "/predict"

    start_time = time.perf_counter()

    try:

        with REQUEST_LATENCY.labels(
            endpoint=endpoint
        ).time():

            # preprocessing
            features = payload_to_features(
                payload
            )

            # prediction
            prediction_encoded = model.predict(
                features
            )[0]

            # inverse transform target
            prediction = (
                target_encoder.inverse_transform(
                    [prediction_encoded]
                )[0]
            )

        latency = (
            time.perf_counter()
            - start_time
        )

        REQUEST_COUNT.labels(
            endpoint=endpoint,
            status="success"
        ).inc()

        PREDICTION_COUNT.labels(
            grade=str(prediction)
        ).inc()

        return PredictionResponse(

            prediction=str(prediction),

            latency_seconds=round(
                latency,
                4
            ),

            model_path=str(MODEL_PATH)
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