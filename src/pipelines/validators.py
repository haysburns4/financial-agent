"""Pydantic validators for the price pipeline.

`RawPriceBar` rejects structurally-invalid quotes (negative prices, inverted
high/low, etc.). `ValidatedPriceBar` adds soft anomaly flags that don't reject
but are persisted alongside the bar so downstream consumers can decide what
to do.
"""
from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator, model_validator


class RawPriceBar(BaseModel):
    ticker: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjusted_close: float | None = None

    @field_validator("ticker")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("timestamp")
    @classmethod
    def _tz_aware_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware UTC")
        return v.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _consistency(self):
        for f in ("open", "high", "low", "close"):
            if getattr(self, f) < 0:
                raise ValueError(f"{f} cannot be negative")
        if self.volume < 0:
            raise ValueError("volume cannot be negative")
        if self.close <= 0:
            raise ValueError("close must be > 0")
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) < low ({self.low})")
        if self.high < self.open or self.high < self.close:
            raise ValueError("high must be >= open and close")
        if self.low > self.open or self.low > self.close:
            raise ValueError("low must be <= open and close")
        if self.adjusted_close is None:
            self.adjusted_close = self.close
        return self


class ValidatedPriceBar(RawPriceBar):
    data_quality: str = "ok"
    anomalies: list[str] = Field(default_factory=list)

    @classmethod
    def from_raw(
        cls, raw: RawPriceBar, *, prev_close: float | None = None
    ) -> "ValidatedPriceBar":
        anomalies: list[str] = []

        if raw.volume == 0:
            anomalies.append("zero_volume")

        if prev_close is not None and prev_close > 0:
            pct_change = abs(raw.close - prev_close) / prev_close
            if pct_change > 0.15:
                anomalies.append("large_move")

        if raw.timestamp.weekday() >= 5:
            anomalies.append("weekend_bar")

        return cls(
            **raw.model_dump(),
            anomalies=anomalies,
            data_quality="flagged" if anomalies else "ok",
        )
