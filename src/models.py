from datetime import date, datetime, timezone

from sqlalchemy import (
    Date,
    DateTime,
    ForeignKey,
    Index,
    JSON,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def as_utc(moment: datetime) -> datetime:
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


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


class PositionSnapshot(Base):
    __tablename__ = "position_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    snapshot_date: Mapped[date] = mapped_column(Date, index=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    capture_source: Mapped[str] = mapped_column(String(20))  # startup | manual | scheduled
    market_state: Mapped[str] = mapped_column(String(20))
    account_id: Mapped[str] = mapped_column(String(50), index=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True)
    quantity: Mapped[float]
    cost_basis: Mapped[float]
    market_value: Mapped[float]
    pnl: Mapped[float]
    pnl_pct: Mapped[float | None]
    positions_as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "snapshot_date", "account_id", "ticker", name="uq_position_snapshots_date_account_ticker"
        ),
        Index("ix_position_snapshots_date_account", "snapshot_date", "account_id"),
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
    # Set by the Discord tick (src/alerts/discord.py) once a signal is posted.
    # TODO: One flag covers one channel; a second channel would never see what
    # the first already claimed. Before adding one, move delivery to a
    # signal_deliveries table keyed (signal_id, channel), claim rows with an
    # UPDATE ... WHERE delivered IS false RETURNING so two workers can't both
    # send, and drop this column.
    delivered: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class AgentOutput(Base):
    """Everything the agent said (and was asked), kept with its provenance.

    Rows are immutable: content is never updated in place, so `content_hash`
    always describes `content` — a re-embedding pass can skip rows whose hash
    it has seen, and find the ones a new embedding model has not. Changed
    content is a new row. Nothing assumes one row is one retrieval chunk.
    """

    __tablename__ = "agent_outputs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    output_type: Mapped[str] = mapped_column(String(20))  # synthesis | chat
    role: Mapped[str | None] = mapped_column(String(20))  # user | assistant; None for synthesis
    content: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)  # sha256 hex of content
    provider: Mapped[str] = mapped_column(String(30))
    model: Mapped[str] = mapped_column(String(120))
    conversation_id: Mapped[str | None] = mapped_column(String(40), index=True)
    context_ref: Mapped[dict | None] = mapped_column(JSON)
    latency_ms: Mapped[int | None]
    # For a future delivery channel to claim syntheses, like signals.delivered.
    # TODO: Nothing sets this yet, so /agent/history/undelivered and /health's
    # undelivered_syntheses just mirror the full synthesis history. When a
    # channel (Discord, email) starts sending syntheses, have it read
    # list_outputs(undelivered=True) and set the flag right after each
    # successful send, as discord.py does for signals. If syntheses ever go to
    # more than one channel, use a per-channel deliveries table instead, as
    # noted on Signal.delivered.
    delivered: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (Index("ix_agent_outputs_type_timestamp", "output_type", "timestamp"),)


class AgentOutputRef(Base):
    """What an agent output refers to, one row per reference, for filtering.

    Derived from the output's context_ref on insert. Retrieval over the archive
    filters by ticker (and date) before or alongside any vector search, so
    "every narrative mentioning NVDA in September" is an indexed join here
    rather than a scan of JSON.
    """

    __tablename__ = "agent_output_refs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    output_id: Mapped[int] = mapped_column(ForeignKey("agent_outputs.id"), index=True)
    ref_type: Mapped[str] = mapped_column(String(20))  # ticker | signal_id | account_id
    ref_value: Mapped[str] = mapped_column(String(50))

    __table_args__ = (Index("ix_agent_output_refs_type_value", "ref_type", "ref_value"),)


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
