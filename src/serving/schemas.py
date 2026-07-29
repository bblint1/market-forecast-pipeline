from pydantic import BaseModel, ConfigDict, Field


class MarketFeaturesRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

    open: float = Field(...)
    high: float = Field(...)
    low: float = Field(...)
    close: float = Field(...)
    volume: float = Field(...)
    return_1h: float = Field(...)
    return_24h: float = Field(...)
    lag_return_1h: float = Field(...)
    lag_return_24h: float = Field(...)
    volatility_24h: float = Field(...)
    volume_mean_24h: float = Field(...)
    rsi_14: float = Field(...)
    bollinger_hband: float = Field(...)
    bollinger_lband: float = Field(...)
    target_next_24h_return: float = Field(...)


class PredictionResponse(BaseModel):
    predicted_crypto_24h_return: float
    trading_signal: str
    model_version: str = "1.0.0"
