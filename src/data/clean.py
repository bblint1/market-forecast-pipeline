from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def clean_market_data(file_type: str = "crypto") -> str:
    date_str = pd.Timestamp.now().strftime("%Y%m%d")
    raw_path = RAW_DATA_DIR / f"{file_type}_{date_str}.parquet"

    if not raw_path.exists():
        files = sorted(RAW_DATA_DIR.glob(f"{file_type}_*.parquet"))
        if not files:
            raise FileNotFoundError(f"Raw file not found for {file_type}")
        raw_path = files[-1]

    df = pd.read_parquet(raw_path)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    if "datetime" not in [str(c).lower() for c in df.columns]:
        df = df.reset_index()

    df.columns = [str(c).lower().strip() for c in df.columns]

    date_col_candidates = ["date", "datetime", "index", "timestamp"]
    for cand in date_col_candidates:
        if cand in df.columns:
            df = df.rename(columns={cand: "datetime"})
            break

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.drop_duplicates(subset=["datetime"]).reset_index(drop=True)

    numeric_cols = [
        c for c in ["open", "high", "low", "close", "volume"] if c in df.columns
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    processed_path = PROCESSED_DATA_DIR / f"cleaned_{file_type}_{date_str}.parquet"
    df.to_parquet(processed_path, index=False)
    return str(processed_path)


if __name__ == "__main__":
    clean_market_data("crypto")
    clean_market_data("stocks")
