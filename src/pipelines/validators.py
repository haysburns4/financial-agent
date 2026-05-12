from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class PriceBarIn(BaseModel):
    ticker: str
    timestamp: datetime
    open: float = Field(ge=0)
    high: float = Field(ge=0)
    low: float = Field(ge=0)
    close: float = Field(ge=0)
    volume: float = Field(ge=0)
    adjusted_close: float = Field(ge=0)
    data_quality: str = "ok"

    @field_validator("ticker")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()

    def is_consistent(self) -> bool:
        return self.low <= min(self.open, self.close) and self.high >= max(self.open, self.close)


class PositionIn(BaseModel):
    account_id: str
    ticker: str
    quantity: float
    cost_basis: float
    market_value: float
    last_updated: datetime

    @field_validator("ticker")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()
