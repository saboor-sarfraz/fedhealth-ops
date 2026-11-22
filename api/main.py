"""FastAPI inference service for the federated global ECG autoencoder."""

import csv
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, FiniteFloat

from hospital_clients.training import get_reconstruction_errors
from models.autoencoder import ECGAutoencoder


ROOT_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT_DIR / "outputs" / "federated_global_autoencoder.pth"
EVALUATION_METRICS_PATH = (
    ROOT_DIR / "outputs" / "evaluation" / "model_comparison_metrics.csv"
)
FEDERATED_MODEL_LABEL = "Federated global"
INPUT_SAMPLES = 200
DEVICE = torch.device("cpu")


class PredictionRequest(BaseModel):
    """One ECG beat represented by 200 numeric samples."""

    samples: list[FiniteFloat] = Field(
        min_length=INPUT_SAMPLES,
        max_length=INPUT_SAMPLES,
    )


class PredictionResponse(BaseModel):
    prediction: Literal["normal", "anomalous"]
    reconstruction_error: float
    threshold: float


def load_federated_threshold() -> float:
    """Read the federated model's calibrated threshold from evaluation results."""
    if not EVALUATION_METRICS_PATH.is_file():
        raise FileNotFoundError(
            f"Evaluation metrics file not found: {EVALUATION_METRICS_PATH}"
        )

    with EVALUATION_METRICS_PATH.open(
        newline="",
        encoding="utf-8",
    ) as csv_file:
        rows = [
            row
            for row in csv.DictReader(csv_file)
            if row.get("model") == FEDERATED_MODEL_LABEL
        ]

    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one '{FEDERATED_MODEL_LABEL}' row in "
            f"{EVALUATION_METRICS_PATH}; found {len(rows)}"
        )
    try:
        threshold = float(rows[0]["threshold"])
    except (KeyError, ValueError) as error:
        raise ValueError(
            f"The federated threshold in {EVALUATION_METRICS_PATH} is invalid"
        ) from error

    if not math.isfinite(threshold):
        raise ValueError("The federated anomaly threshold must be finite")
    return threshold


def load_inference_model() -> ECGAutoencoder:
    """Load the existing federated checkpoint onto the CPU."""
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Federated checkpoint not found: {CHECKPOINT_PATH}")

    model = ECGAutoencoder().to(DEVICE)
    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_inference_model()
    app.state.threshold = load_federated_threshold()
    yield


app = FastAPI(
    title="FedHealth-Ops ECG Inference API",
    description="CPU inference using the federated global ECG autoencoder.",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str | int]:
    return {
        "service": "FedHealth-Ops ECG inference API",
        "model": "Federated global ECG autoencoder",
        "input_samples": INPUT_SAMPLES,
        "health": "/health",
        "predict": "/predict",
        "docs": "/docs",
    }


@app.get("/health")
def health(request: Request) -> dict[str, bool | str]:
    model_loaded = getattr(request.app.state, "model", None) is not None
    threshold_loaded = getattr(request.app.state, "threshold", None) is not None
    return {
        "status": "ok" if model_loaded and threshold_loaded else "not_ready",
        "model_loaded": model_loaded,
        "threshold_loaded": threshold_loaded,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    model = getattr(request.app.state, "model", None)
    threshold = getattr(request.app.state, "threshold", None)
    if model is None or threshold is None:
        raise HTTPException(
            status_code=503,
            detail="The federated model or its threshold is not loaded",
        )

    beat = np.asarray([payload.samples], dtype=np.float32)
    reconstruction_error = float(
        get_reconstruction_errors(
            beat,
            model,
            batch_size=1,
            device=DEVICE,
        )[0]
    )
    prediction = (
        "anomalous" if reconstruction_error > threshold else "normal"
    )
    return PredictionResponse(
        prediction=prediction,
        reconstruction_error=reconstruction_error,
        threshold=threshold,
    )
