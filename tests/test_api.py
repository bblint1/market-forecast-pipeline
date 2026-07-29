import sys
from pathlib import Path

from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.serving.app import app  # noqa: E402

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "model_name" in response.json()


def test_predict_endpoint():
    sample_payload = {
        "open": 50000.0,
        "high": 51000.0,
        "low": 49500.0,
        "close": 50500.0,
        "volume": 1200.0,
        "return_1h": 0.005,
        "return_24h": 0.021,
        "lag_return_1h": 0.002,
        "lag_return_24h": 0.018,
        "volatility_24h": 0.012,
        "volume_mean_24h": 1500.5,
        "rsi_14": 55.4,
        "bollinger_hband": 52000.0,
        "bollinger_lband": 48000.0,
        "target_next_24h_return": 0.005,
    }

    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_crypto_24h_return" in data
    assert "trading_signal" in data
    assert data["trading_signal"] in ["BUY", "SELL", "HOLD"]
