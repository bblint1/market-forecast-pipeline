import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FEATURES_DATA_DIR = BASE_DIR / "data" / "features"

file_path = FEATURES_DATA_DIR

if not os.path.exists(file_path):
    print("Local parquet not found. Ensure features parquet is present.")

df = pd.read_parquet(file_path)
df = df.dropna(subset=["crypto_target_next_24h_return"]).reset_index(drop=True)

df = df.sort_values("datetime").reset_index(drop=True)
split_idx = int(len(df) * 0.8)

train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

drop_cols = ["datetime", "crypto_target_next_24h_return"]
feature_cols = [c for c in df.columns if c not in drop_cols]

X_train, y_train = train_df[feature_cols], train_df["crypto_target_next_24h_return"]
X_test, y_test = test_df[feature_cols], test_df["crypto_target_next_24h_return"]

# 1. Enable MLflow Autologging
mlflow.xgboost.autolog()

with mlflow.start_run(run_name="xgboost_crypto_return_forecast") as run:
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # 2. Log extra metrics manually: RMSE, MAE, MAPE
    rmse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)

    mlflow.log_metric("test_rmse", rmse)
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_mape", mape)

    # 3. Log feature importance as artifact (PNG plot)
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, max_num_features=10)
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plot_path = "feature_importance.png"
    plt.savefig(plot_path)
    plt.close()

    mlflow.log_artifact(plot_path)
    if os.path.exists(plot_path):
        os.remove(plot_path)

    # 4. Register model in MLflow Registry
    model_uri = f"runs:/{run.info.run_id}/model"
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name="energy-forecast-xgboost",
    )

    print(
        f"Model Registered: {registered_model.name}, Version: {registered_model.version}"
    )
    print(f"Metrics - RMSE: {rmse:.5f}, MAE: {mae:.5f}, MAPE: {mape:.5f}")
