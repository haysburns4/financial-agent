"""Signal evaluation: entry/exit/risk rules over price+indicator+position state."""
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd
from loguru import logger
from sqlalchemy import and_, insert, select
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from src.backtest import confidence_config
from src.models import Indicator, Position, PriceBar, Signal


def _conf(rule_name: str, fallback: float = 0.5) -> float:
    """Look up the current calibrated weight for a rule. Read by attribute so
    POST /backtest/apply takes effect immediately (no module reload needed)."""
    return confidence_config.CONFIDENCE_WEIGHTS.get(rule_name, fallback)


SIGNAL_CATEGORIES: dict[str, str] = {
    "oversold_reversal": "entry",
    "golden_cross": "entry",
    "breakout": "entry",
    "overbought_reversal": "exit",
    "death_cross": "exit",
    "stop_loss_warning": "exit",
    "concentration_risk": "risk",
    "drawdown_alert": "risk",
}


def signal_types_for_category(category: str) -> list[str]:
    return [k for k, v in SIGNAL_CATEGORIES.items() if v == category]


@dataclass
class SignalCandidate:
    ticker: str
    timestamp: datetime
    signal_type: str
    direction: str | None
    confidence: float
    reasoning: str

    def to_row(self) -> dict:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp,
            "signal_type": self.signal_type,
            "direction": self.direction,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "delivered": False,
        }


def _val(series: pd.Series, key: str) -> float | None:
    """Read a value from a pandas Series, returning None for missing/NaN."""
    v = series.get(key)
    if v is None:
        return None
    try:
        missing = pd.isna(v)
    except (TypeError, ValueError):
        # Non-scalar or unorderable: treat as present and let float() rule on it.
        missing = False
    if missing:
        return None
    return float(v)


def _ts(value) -> datetime:
    """Coerce a pandas Timestamp / naive datetime / None into tz-aware UTC datetime."""
    if value is None:
        return datetime.now(timezone.utc)
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if not isinstance(value, datetime):
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class SignalEngine:
    DEDUP_WINDOW = timedelta(hours=4)
    HISTORY_LIMIT = 30
    BREAKOUT_LOOKBACK = 20
    PORTFOLIO_SENTINEL = "PORTFOLIO"

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def evaluate(self, ticker: str) -> list[dict]:
        ticker = ticker.strip().upper()
        async with self._engine.connect() as conn:
            history = await self._load_history(conn, ticker, self.HISTORY_LIMIT)
            position = await self._load_position(conn, ticker)
            portfolio = await self._load_portfolio(conn)
        candidates = self._build_candidates(ticker, history, position, portfolio)
        return [asdict(c) for c in candidates]

    async def run_all(self, tickers: list[str]) -> list[dict]:
        inserted: list[dict] = []
        emitted_portfolio: set[str] = set()  # within-run dedup for portfolio-wide types

        async with self._engine.begin() as conn:
            portfolio = await self._load_portfolio(conn)
            for ticker in tickers:
                ticker = ticker.strip().upper()
                history = await self._load_history(conn, ticker, self.HISTORY_LIMIT)
                position = await self._load_position(conn, ticker)
                candidates = self._build_candidates(ticker, history, position, portfolio)

                for c in candidates:
                    # Portfolio-wide signals: emit once per run regardless of carrier ticker.
                    if SIGNAL_CATEGORIES.get(c.signal_type) == "risk" and c.signal_type == "drawdown_alert":
                        if c.signal_type in emitted_portfolio:
                            continue
                        emitted_portfolio.add(c.signal_type)

                    if await self._is_duplicate(conn, c):
                        logger.debug(
                            "dedup'd recent {} for {} (within {})",
                            c.signal_type,
                            c.ticker,
                            self.DEDUP_WINDOW,
                        )
                        continue

                    res = await conn.execute(insert(Signal).values(**c.to_row()))
                    sig_id = res.inserted_primary_key[0]
                    inserted.append({**asdict(c), "id": sig_id})
                    logger.info(
                        "signal: {} {} ({:.2f}) — {}",
                        c.ticker,
                        c.signal_type,
                        c.confidence,
                        c.reasoning,
                    )

        logger.info("signal engine: {} new signal(s) across {} ticker(s)", len(inserted), len(tickers))
        return inserted

    # ---------- candidate construction ----------

    def _build_candidates(
        self,
        ticker: str,
        history: pd.DataFrame,
        position: dict | None,
        portfolio: list[dict],
    ) -> list[SignalCandidate]:
        out: list[SignalCandidate] = []
        if not history.empty and len(history) >= 2:
            out.extend(self._entry_signals(ticker, history))
            out.extend(self._exit_signals(ticker, history))
        if position is not None:
            out.extend(self._stop_loss(ticker, position, history))
            out.extend(self._concentration(ticker, position, portfolio))
        out.extend(self._drawdown(portfolio))
        return out

    def _entry_signals(self, ticker: str, history: pd.DataFrame) -> list[SignalCandidate]:
        out: list[SignalCandidate] = []
        last = history.iloc[-1]
        prev = history.iloc[-2]
        ts = _ts(last["timestamp"])

        rsi = _val(last, "rsi_14")
        hist_now = _val(last, "macd_hist")
        hist_prev = _val(prev, "macd_hist")

        # oversold_reversal
        if rsi is not None and rsi < 30 and hist_now is not None and hist_prev is not None:
            if hist_now > 0 and hist_prev <= 0:
                out.append(SignalCandidate(
                    ticker=ticker, timestamp=ts,
                    signal_type="oversold_reversal", direction="long",
                    confidence=_conf("oversold_reversal"),
                    reasoning=(
                        f"RSI={rsi:.1f} (<30), MACD histogram flipped positive "
                        f"({hist_prev:.3f} → {hist_now:.3f})"
                    ),
                ))

        # golden_cross
        e9, e21 = _val(last, "ema_9"), _val(last, "ema_21")
        pe9, pe21 = _val(prev, "ema_9"), _val(prev, "ema_21")
        if None not in (e9, e21, pe9, pe21):
            if pe9 <= pe21 and e9 > e21:
                out.append(SignalCandidate(
                    ticker=ticker, timestamp=ts,
                    signal_type="golden_cross", direction="long",
                    confidence=_conf("golden_cross"),
                    reasoning=f"EMA9 ({e9:.2f}) crossed above EMA21 ({e21:.2f})",
                ))

        # breakout: close above 20-day high on above-average volume
        if len(history) >= self.BREAKOUT_LOOKBACK + 1:
            lookback = history.iloc[-(self.BREAKOUT_LOOKBACK + 1):-1]
            prior_high = float(lookback["high"].max())
            vol_avg = float(lookback["volume"].mean())
            last_close = float(last["close"])
            last_vol = float(last["volume"])
            if last_close > prior_high and vol_avg > 0 and last_vol > vol_avg:
                out.append(SignalCandidate(
                    ticker=ticker, timestamp=ts,
                    signal_type="breakout", direction="long",
                    confidence=_conf("breakout"),
                    reasoning=(
                        f"Close {last_close:.2f} > {self.BREAKOUT_LOOKBACK}-day high "
                        f"{prior_high:.2f} on volume {last_vol:.0f} (avg {vol_avg:.0f})"
                    ),
                ))

        return out

    def _exit_signals(self, ticker: str, history: pd.DataFrame) -> list[SignalCandidate]:
        out: list[SignalCandidate] = []
        last = history.iloc[-1]
        prev = history.iloc[-2]
        ts = _ts(last["timestamp"])

        rsi = _val(last, "rsi_14")
        hist_now = _val(last, "macd_hist")
        hist_prev = _val(prev, "macd_hist")

        # overbought_reversal
        if rsi is not None and rsi > 70 and hist_now is not None and hist_prev is not None:
            if hist_now < 0 and hist_prev >= 0:
                out.append(SignalCandidate(
                    ticker=ticker, timestamp=ts,
                    signal_type="overbought_reversal", direction=None,
                    confidence=_conf("overbought_reversal"),
                    reasoning=(
                        f"RSI={rsi:.1f} (>70), MACD histogram flipped negative "
                        f"({hist_prev:.3f} → {hist_now:.3f})"
                    ),
                ))

        # death_cross
        e9, e21 = _val(last, "ema_9"), _val(last, "ema_21")
        pe9, pe21 = _val(prev, "ema_9"), _val(prev, "ema_21")
        if None not in (e9, e21, pe9, pe21):
            if pe9 >= pe21 and e9 < e21:
                out.append(SignalCandidate(
                    ticker=ticker, timestamp=ts,
                    signal_type="death_cross", direction=None,
                    confidence=_conf("death_cross"),
                    reasoning=f"EMA9 ({e9:.2f}) crossed below EMA21 ({e21:.2f})",
                ))

        return out

    def _stop_loss(
        self, ticker: str, position: dict, history: pd.DataFrame
    ) -> list[SignalCandidate]:
        if position["cost_basis"] <= 0:
            return []
        pnl_pct = (position["market_value"] - position["cost_basis"]) / position["cost_basis"]
        if pnl_pct >= -0.05:
            return []
        ts = _ts(history.iloc[-1]["timestamp"]) if not history.empty else datetime.now(timezone.utc)
        return [SignalCandidate(
            ticker=ticker, timestamp=ts,
            signal_type="stop_loss_warning", direction=None,
            confidence=_conf("stop_loss_warning"),
            reasoning=(
                f"Position P&L {pnl_pct:.1%} "
                f"(cost ${position['cost_basis']:.0f} → mkt ${position['market_value']:.0f})"
            ),
        )]

    def _concentration(
        self, ticker: str, position: dict, portfolio: list[dict]
    ) -> list[SignalCandidate]:
        total_value = sum(p["market_value"] for p in portfolio)
        if total_value <= 0:
            return []
        pos_pct = position["market_value"] / total_value
        if pos_pct <= 0.15:
            return []
        return [SignalCandidate(
            ticker=ticker, timestamp=datetime.now(timezone.utc),
            signal_type="concentration_risk", direction=None,
            confidence=_conf("concentration_risk"),
            reasoning=(
                f"{ticker} is {pos_pct:.1%} of portfolio "
                f"(${position['market_value']:.0f} / ${total_value:.0f})"
            ),
        )]

    def _drawdown(self, portfolio: list[dict]) -> list[SignalCandidate]:
        # NOTE: True drawdown requires a peak-tracking history we don't yet store.
        # As a proxy we report unrealized loss against aggregate cost basis.
        if not portfolio:
            return []
        total_cost = sum(p["cost_basis"] for p in portfolio)
        total_value = sum(p["market_value"] for p in portfolio)
        if total_cost <= 0:
            return []
        drawdown = (total_cost - total_value) / total_cost
        if drawdown <= 0.10:
            return []
        return [SignalCandidate(
            ticker=self.PORTFOLIO_SENTINEL,
            timestamp=datetime.now(timezone.utc),
            signal_type="drawdown_alert", direction=None,
            confidence=_conf("drawdown_alert"),
            reasoning=(
                f"Portfolio unrealized drawdown {drawdown:.1%} "
                f"(cost ${total_cost:.0f} → mkt ${total_value:.0f})"
            ),
        )]

    # ---------- loaders ----------

    async def _load_history(
        self, conn: AsyncConnection, ticker: str, limit: int
    ) -> pd.DataFrame:
        stmt = (
            select(
                PriceBar.timestamp,
                PriceBar.open,
                PriceBar.high,
                PriceBar.low,
                PriceBar.close,
                PriceBar.volume,
                Indicator.rsi_14,
                Indicator.macd_line,
                Indicator.macd_signal,
                Indicator.macd_hist,
                Indicator.ema_9,
                Indicator.ema_21,
            )
            .outerjoin(
                Indicator,
                and_(
                    PriceBar.ticker == Indicator.ticker,
                    PriceBar.timestamp == Indicator.timestamp,
                ),
            )
            .where(PriceBar.ticker == ticker)
            .order_by(PriceBar.timestamp.desc())
            .limit(limit)
        )
        rows = (await conn.execute(stmt)).all()
        if not rows:
            return pd.DataFrame()
        records = [
            {
                "timestamp": r.timestamp,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
                "rsi_14": r.rsi_14,
                "macd_line": r.macd_line,
                "macd_signal": r.macd_signal,
                "macd_hist": r.macd_hist,
                "ema_9": r.ema_9,
                "ema_21": r.ema_21,
            }
            for r in reversed(rows)  # chronological
        ]
        return pd.DataFrame(records)

    async def _load_position(
        self, conn: AsyncConnection, ticker: str
    ) -> dict | None:
        stmt = (
            select(Position)
            .where(Position.ticker == ticker)
            .order_by(Position.last_updated.desc())
            .limit(1)
        )
        row = (await conn.execute(stmt)).first()
        if not row:
            return None
        return {
            "account_id": row.account_id,
            "ticker": row.ticker,
            "quantity": row.quantity,
            "cost_basis": row.cost_basis,
            "market_value": row.market_value,
            "last_updated": row.last_updated,
        }

    async def _load_portfolio(self, conn: AsyncConnection) -> list[dict]:
        stmt = select(Position)
        rows = (await conn.execute(stmt)).all()
        return [
            {
                "ticker": r.ticker,
                "market_value": r.market_value,
                "cost_basis": r.cost_basis,
            }
            for r in rows
        ]

    async def _is_duplicate(
        self, conn: AsyncConnection, candidate: SignalCandidate
    ) -> bool:
        cutoff = datetime.now(timezone.utc) - self.DEDUP_WINDOW
        direction_filter = (
            Signal.direction.is_(None)
            if candidate.direction is None
            else Signal.direction == candidate.direction
        )
        stmt = (
            select(Signal.id)
            .where(
                Signal.ticker == candidate.ticker,
                Signal.signal_type == candidate.signal_type,
                direction_filter,
                Signal.created_at >= cutoff,
            )
            .limit(1)
        )
        return (await conn.execute(stmt)).first() is not None
