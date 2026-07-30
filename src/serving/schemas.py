from pydantic import BaseModel, ConfigDict, Field


class MarketFeaturesRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

    crypto_open: float = Field(...)
    crypto_high: float = Field(...)
    crypto_low: float = Field(...)
    crypto_close: float = Field(...)
    crypto_volume: float = Field(...)
    crypto_return_1h: float = Field(...)
    crypto_return_24h: float = Field(...)
    crypto_lag_return_1h: float = Field(...)
    crypto_lag_return_24h: float = Field(...)
    crypto_volatility_24h: float = Field(...)
    crypto_volume_mean_24h: float = Field(...)
    crypto_rsi_14: float = Field(...)
    crypto_bollinger_hband: float = Field(...)
    crypto_bollinger_lband: float = Field(...)

    stocks_open: float = Field(...)
    stocks_high: float = Field(...)
    stocks_low: float = Field(...)
    stocks_close: float = Field(...)
    stocks_volume: float = Field(...)
    stocks_return_1h: float = Field(...)
    stocks_return_24h: float = Field(...)
    stocks_lag_return_1h: float = Field(...)
    stocks_lag_return_24h: float = Field(...)
    stocks_volatility_24h: float = Field(...)
    stocks_volume_mean_24h: float = Field(...)
    stocks_rsi_14: float = Field(...)
    stocks_bollinger_hband: float = Field(...)
    stocks_bollinger_lband: float = Field(...)


class PredictionResponse(BaseModel):
    predicted_crypto_24h_return: float
    trading_signal: str
    model_version: str = "1.0.0"
