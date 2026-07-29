import os

import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Locate features data file
# In local/Git setup, adjust path relative to repo root
file_path = "../../data/features/features_20260729.parquet"

if not os.path.exists(file_path):
    print("Local parquet not found. Ensure features parquet is present.")

df = pd.read_parquet(file_path)

# 2. Time-Based Train/Test Split
df = df.sort_values("datetime").reset_index(drop=True)
split_idx = int(len(df) * 0.8)

train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

drop_cols = ["datetime", "crypto_target_next_24h_return"]
feature_cols = [c for c in df.columns if c not in drop_cols]

X_train, y_train = train_df[feature_cols], train_df["crypto_target_next_24h_return"]
X_test, y_test = test_df[feature_cols], test_df["crypto_target_next_24h_return"]

# 3. Train XGBoost with MLflow Autologging
mlflow.xgboost.autolog()

with mlflow.start_run(run_name="xgboost_crypto_return_forecast"):
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = mean_squared_error(y_test, predictions, squared=False)
    mae = mean_absolute_error(y_test, predictions)

    mlflow.log_metric("test_rmse", rmse)
    mlflow.log_metric("test_mae", mae)

    print(f"Model Training Complete. Test RMSE: {rmse:.5f}, MAE: {mae:.5f}")
