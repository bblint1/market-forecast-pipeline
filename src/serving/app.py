import os
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

MODEL_PATH = BASE_DIR / "models" / "xgboost_model.json"
FEATURES_DATA_DIR = BASE_DIR / "data" / "features"

from contextlib import asynccontextmanager  # noqa: E402

from src.serving.schemas import MarketFeaturesRequest, PredictionResponse  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_or_train_model()
    yield


app = FastAPI(
    title="Crypto & Stock Volatility Forecast API",
    description="Serving endpoint predicting 24-hour Bitcoin return based on market signals.",
    version="1.0.0",
    lifespan=lifespan,
)

model = None
feature_names = []


def load_or_train_model():
    global model, feature_names
    if os.path.exists(FEATURES_DATA_DIR):
        files = sorted(FEATURES_DATA_DIR.glob("*.parquet"))
        if files:
            df = pd.read_parquet(files[-1])
            target_cols = [c for c in df.columns if "target" in c]
            if target_cols:
                target_col = target_cols[0]
                df = df.dropna(subset=[target_col])
                drop_cols = ["datetime", target_col]
            else:
                drop_cols = ["datetime"]
                target_col = None

            feature_names = [c for c in df.columns if c not in drop_cols]

            if MODEL_PATH.exists():
                model = xgb.XGBRegressor()
                model.load_model(str(MODEL_PATH))
                return

            if target_col:
                X = df[feature_names]
                y = df[target_col]

                model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
                model.fit(X, y)

                MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                model.save_model(str(MODEL_PATH))
                return


from fastapi.responses import HTMLResponse  # noqa: E402


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.get("/drift-report", response_class=HTMLResponse)
def drift_report():
    report_file = BASE_DIR / "monitoring" / "reports" / "data_drift_report.html"
    if not report_file.exists():
        raise HTTPException(
            status_code=440,
            detail="Drift report has not been generated yet. Run pipeline first.",
        )
    return report_file.read_text(encoding="utf-8")


@app.get("/model-info")
def model_info():
    return {
        "model_name": "energy-forecast-xgboost-optuna",
        "framework": "XGBoost",
        "target": "crypto_target_next_24h_return",
        "status": "Active",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: MarketFeaturesRequest):
    if model is None:
        load_or_train_model()
        if model is None:
            raise HTTPException(
                status_code=500, detail="Model is not trained or loaded."
            )

    features_dict = (
        features.model_dump() if hasattr(features, "model_dump") else features.dict()
    )

    if feature_names:
        input_df = pd.DataFrame([features_dict])[feature_names]
    else:
        input_df = pd.DataFrame([features_dict])

    pred = float(model.predict(input_df)[0])

    if pred > 0.01:
        signal = "BUY"
    elif pred < -0.01:
        signal = "SELL"
    else:
        signal = "HOLD"

    return PredictionResponse(
        predicted_crypto_24h_return=round(pred, 5),
        trading_signal=signal,
        model_version="1.0.0",
    )
