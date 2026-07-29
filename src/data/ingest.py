from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_crypto_data(
    symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 1000
) -> pd.DataFrame:
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]
    df = pd.DataFrame(data, columns=columns)
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df[["datetime", "open", "high", "low", "close", "volume"]]

    date_str = pd.Timestamp.now().strftime("%Y%m%d")
    output_path = RAW_DATA_DIR / f"crypto_{date_str}.parquet"
    df.to_parquet(output_path, index=False)
    return df


def fetch_stock_data(
    ticker: str = "SPY", period: str = "1mo", interval: str = "1h"
) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df.columns = [str(c).lower().strip() for c in df.columns]
    date_col = "date" if "date" in df.columns else df.columns[0]
    df = df.rename(columns={date_col: "datetime"})

    df = df[["datetime", "open", "high", "low", "close", "volume"]]

    date_str = pd.Timestamp.now().strftime("%Y%m%d")
    output_path = RAW_DATA_DIR / f"stocks_{date_str}.parquet"
    df.to_parquet(output_path, index=False)
    return df


if __name__ == "__main__":
    fetch_crypto_data()
    fetch_stock_data()
