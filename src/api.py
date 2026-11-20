
from pathlib import Path
from contextlib import asynccontextmanager
import json
import sqlite3
import time
import uuid

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "fraud_model.joblib"
DB_PATH = ROOT / "data" / "predictions.db"

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

artifact = None


def initialize_database():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT
                    (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                fraud_probability REAL NOT NULL,
                predicted_class INTEGER NOT NULL,
                threshold REAL NOT NULL,
                latency_ms REAL NOT NULL,
                features_json TEXT NOT NULL,
                actual_class INTEGER
            )
        """)
        conn.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global artifact

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. "
            "Run python src/train.py first."
        )

    artifact = joblib.load(MODEL_PATH)
    initialize_database()

    yield


app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="Fraud prediction and monitoring service",
    version="1.0.0",
    lifespan=lifespan,
)


class TransactionRequest(BaseModel):
    # The dataset has 30 input features.
    # Extra fields are rejected to catch schema mistakes.
    transaction: dict[str, float] = Field(
        ...,
        description="Transaction features: Time, V1-V28, Amount",
    )


def save_prediction(
    prediction_id,
    probability,
    predicted_class,
    threshold,
    latency_ms,
    features,
):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO predictions (
                prediction_id,
                fraud_probability,
                predicted_class,
                threshold,
                latency_ms,
                features_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            prediction_id,
            probability,
            predicted_class,
            threshold,
            latency_ms,
            json.dumps(features),
        ))
        conn.commit()


@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": artifact is not None,
    }


@app.post("/predict")
def predict(request: TransactionRequest):
    if artifact is None:
        raise HTTPException(503, "Model is not loaded")

    features = request.transaction
    expected = artifact["feature_names"]

    missing = sorted(set(expected) - set(features))
    extra = sorted(set(features) - set(expected))

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Invalid transaction feature schema",
                "missing_features": missing,
                "unexpected_features": extra,
            },
        )

    ordered_features = {
        name: features[name] for name in expected
    }

    # Reject NaN and infinite values.
    import math
    if not all(math.isfinite(v) for v in ordered_features.values()):
        raise HTTPException(
            status_code=422,
            detail="All features must be finite numbers",
        )

    start = time.perf_counter()

    try:
        X = pd.DataFrame([ordered_features], columns=expected)
        probability = float(
            artifact["model"].predict_proba(X)[0, 1]
        )
        threshold = float(artifact["threshold"])
        predicted_class = int(probability >= threshold)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed; check server logs",
        ) from exc

    latency_ms = (time.perf_counter() - start) * 1000
    prediction_id = str(uuid.uuid4())

    save_prediction(
        prediction_id,
        probability,
        predicted_class,
        threshold,
        latency_ms,
        ordered_features,
    )

    return {
        "prediction_id": prediction_id,
        "fraud_probability": round(probability, 6),
        "predicted_class": predicted_class,
        "risk": "HIGH" if predicted_class else "LOW",
        "threshold": threshold,
        "latency_ms": round(latency_ms, 3),
    }


@app.get("/metrics/summary")
def metrics_summary():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("""
            SELECT
                COUNT(*) AS total_predictions,
                COALESCE(SUM(predicted_class), 0) AS flagged_fraud,
                COALESCE(AVG(fraud_probability), 0)
                    AS avg_fraud_probability,
                COALESCE(AVG(latency_ms), 0) AS avg_latency_ms
            FROM predictions
        """).fetchone()

    return dict(row)


@app.get("/predictions")
def recent_predictions(limit: int = 50):
    if not 1 <= limit <= 500:
        raise HTTPException(
            status_code=422,
            detail="limit must be between 1 and 500",
        )

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT prediction_id, created_at, fraud_probability,
                   predicted_class, threshold, latency_ms, actual_class
            FROM predictions
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return {"predictions": [dict(row) for row in rows]}
