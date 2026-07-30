import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

drop_cols = [
    "datetime",
    "crypto_target_next_24h_return",
    "stocks_target_next_24h_return",
]
feature_cols = [c for c in df.columns if c not in drop_cols]

X_train, y_train = train_df[feature_cols], train_df["crypto_target_next_24h_return"]
X_test, y_test = test_df[feature_cols], test_df["crypto_target_next_24h_return"]


# 1. Define Optuna Objective Function
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
    }

    with mlflow.start_run(nested=True):
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        val_rmse = mean_squared_error(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metric("val_rmse", val_rmse)

    return val_rmse


# 2. Run Optuna Study
mlflow.set_experiment("xgboost_optuna_tuning")

with mlflow.start_run(run_name="optuna_parent_run"):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    best_params = study.best_params
    print(f"Best Hyperparameters: {best_params}")

    # 3. Train & Log Final Model with Best Hyperparameters
    with mlflow.start_run(run_name="best_tuned_xgboost_model", nested=True) as best_run:
        best_model = xgb.XGBRegressor(**best_params, random_state=42)
        best_model.fit(X_train, y_train)

        final_preds = best_model.predict(X_test)
        final_rmse = mean_squared_error(y_test, final_preds)
        final_mae = mean_absolute_error(y_test, final_preds)

        mlflow.log_params(best_params)
        mlflow.log_metric("final_test_rmse", final_rmse)
        mlflow.log_metric("final_test_mae", final_mae)

        # Log Model explicitly for Registry
        mlflow.xgboost.log_model(best_model, artifact_path="model")

        # 4. Feature Importance Artifact
        plt.figure(figsize=(10, 6))
        xgb.plot_importance(best_model, max_num_features=10)
        plt.title("Best Model Feature Importance")
        plt.tight_layout()
        plot_path = "best_feature_importance.png"
        plt.savefig(plot_path)
        plt.close()

        mlflow.log_artifact(plot_path)
        if os.path.exists(plot_path):
            os.remove(plot_path)

        # 5. Save model locally for serving API
        model_save_path = BASE_DIR / "models" / "xgboost_model.json"
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        best_model.save_model(str(model_save_path))

        # 6. Register Best Model
        model_uri = f"runs:/{best_run.info.run_id}/model"
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name="energy-forecast-xgboost-optuna",
        )

        print(
            f"Tuned Model Registered: {registered_model.name}, Version: {registered_model.version}"
        )
        print(f"Best Model Test RMSE: {final_rmse:.5f}, MAE: {final_mae:.5f}")
