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
        "crypto_open": 50000.0,
        "crypto_high": 51000.0,
        "crypto_low": 49500.0,
        "crypto_close": 50500.0,
        "crypto_volume": 1200.0,
        "crypto_return_1h": 0.005,
        "crypto_return_24h": 0.021,
        "crypto_lag_return_1h": 0.002,
        "crypto_lag_return_24h": 0.018,
        "crypto_volatility_24h": 0.012,
        "crypto_volume_mean_24h": 1500.5,
        "crypto_rsi_14": 55.4,
        "crypto_bollinger_hband": 52000.0,
        "crypto_bollinger_lband": 48000.0,
        "stocks_open": 500.0,
        "stocks_high": 505.0,
        "stocks_low": 498.0,
        "stocks_close": 502.0,
        "stocks_volume": 50000.0,
        "stocks_return_1h": 0.001,
        "stocks_return_24h": 0.004,
        "stocks_lag_return_1h": 0.001,
        "stocks_lag_return_24h": 0.003,
        "stocks_volatility_24h": 0.005,
        "stocks_volume_mean_24h": 52000.0,
        "stocks_rsi_14": 52.1,
        "stocks_bollinger_hband": 510.0,
        "stocks_bollinger_lband": 490.0,
    }

    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    assert "predicted_crypto_24h_return" in response.json()
    assert "trading_signal" in response.json()
