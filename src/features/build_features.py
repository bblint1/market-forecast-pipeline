from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
FEATURES_DATA_DIR = BASE_DIR / "data" / "features"
FEATURES_DATA_DIR.mkdir(parents=True, exist_ok=True)


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_market_features(file_type: str = "crypto") -> str:
    date_str = pd.Timestamp.now().strftime("%Y%m%d")
    cleaned_path = PROCESSED_DATA_DIR / f"cleaned_{file_type}_{date_str}.parquet"

    if not cleaned_path.exists():
        files = sorted(PROCESSED_DATA_DIR.glob(f"cleaned_{file_type}_*.parquet"))
        if not files:
            raise FileNotFoundError(f"Cleaned file not found for {file_type}")
        cleaned_path = files[-1]

    df = pd.read_parquet(cleaned_path)

    df["return_1h"] = df["close"].pct_change(1)
    df["return_24h"] = df["close"].pct_change(24)
    df["lag_return_1h"] = df["return_1h"].shift(1)
    df["lag_return_24h"] = df["return_1h"].shift(24)

    df["volatility_24h"] = df["return_1h"].rolling(window=24).std()
    df["volume_mean_24h"] = df["volume"].rolling(window=24).mean()

    df["rsi_14"] = calculate_rsi(df["close"], period=14)

    rolling_mean_20 = df["close"].rolling(window=20).mean()
    rolling_std_20 = df["close"].rolling(window=20).std()
    df["bollinger_hband"] = rolling_mean_20 + (rolling_std_20 * 2)
    df["bollinger_lband"] = rolling_mean_20 - (rolling_std_20 * 2)

    df["target_next_24h_return"] = df["close"].pct_change(24).shift(-24)

    df = df.dropna().reset_index(drop=True)

    features_path = FEATURES_DATA_DIR / f"features_{file_type}_{date_str}.parquet"
    df.to_parquet(features_path, index=False)
    return str(features_path)


if __name__ == "__main__":
    build_market_features("crypto")
    build_market_features("stocks")
