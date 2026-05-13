from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class PriceBar(Base):
    __tablename__ = "price_bars"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    open: Mapped[float]
    high: Mapped[float]
    low: Mapped[float]
    close: Mapped[float]
    volume: Mapped[float]
    adjusted_close: Mapped[float]
    data_quality: Mapped[str] = mapped_column(String(20), default="ok")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("ticker", "timestamp", name="uq_price_bars_ticker_timestamp"),
        Index("ix_price_bars_ticker_timestamp", "ticker", "timestamp"),
    )


class Indicator(Base):
    __tablename__ = "indicators"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    rsi_14: Mapped[float | None]
    macd_line: Mapped[float | None]
    macd_signal: Mapped[float | None]
    macd_hist: Mapped[float | None]
    ema_9: Mapped[float | None]
    ema_21: Mapped[float | None]
    price_bar_id: Mapped[int | None] = mapped_column(ForeignKey("price_bars.id"))

    __table_args__ = (
        UniqueConstraint("ticker", "timestamp", name="uq_indicators_ticker_timestamp"),
    )


class Position(Base):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    account_id: Mapped[str] = mapped_column(String(50))
    ticker: Mapped[str] = mapped_column(String(10))
    quantity: Mapped[float]
    cost_basis: Mapped[float]
    market_value: Mapped[float]
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("account_id", "ticker", name="uq_positions_account_ticker"),
    )


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    signal_type: Mapped[str] = mapped_column(String(30))
    direction: Mapped[str | None] = mapped_column(String(10))
    confidence: Mapped[float]
    reasoning: Mapped[str] = mapped_column(Text)
    delivered: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    pipeline: Mapped[str] = mapped_column(String(50))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(20))
    tickers_processed: Mapped[int] = mapped_column(default=0)
    errors: Mapped[str | None] = mapped_column(Text)


class ETradeCredentials(Base):
    __tablename__ = "etrade_credentials"

    id: Mapped[int] = mapped_column(primary_key=True)  # always 1 (singleton row)
    oauth_token_ct: Mapped[bytes] = mapped_column(LargeBinary)
    oauth_token_secret_ct: Mapped[bytes] = mapped_column(LargeBinary)
    authenticated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
