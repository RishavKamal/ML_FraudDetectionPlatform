
from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient

from src.api import app


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"


def get_valid_payload():
    df = pd.read_csv(DATA_PATH, nrows=1)
    row = df.drop(columns=["Class"]).iloc[0]

    return {
        "transaction": {
            key: float(value)
            for key, value in row.to_dict().items()
        }
    }


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_prediction():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=get_valid_payload(),
        )

    assert response.status_code == 200

    result = response.json()

    assert "prediction_id" in result
    assert 0 <= result["fraud_probability"] <= 1
    assert result["predicted_class"] in (0, 1)
    assert result["risk"] in ("LOW", "HIGH")
    assert result["latency_ms"] >= 0


def test_missing_feature_is_rejected():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"transaction": {"Time": 0.0}},
        )

    assert response.status_code == 422


def test_summary_endpoint():
    with TestClient(app) as client:
        response = client.get("/metrics/summary")

    assert response.status_code == 200

    result = response.json()

    assert "total_predictions" in result
    assert "flagged_fraud" in result
    assert result["total_predictions"] >= 0


def test_prediction_history_endpoint():
    with TestClient(app) as client:
        response = client.get("/predictions?limit=5")

    assert response.status_code == 200
    assert "predictions" in response.json()
    assert len(response.json()["predictions"]) <= 5
