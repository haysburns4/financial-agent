"""Signal evaluation: entry/exit/risk rules over price+indicator+position state.

Entry/exit rules run on the 5-minute series. Before a signal is recorded it
must be confirmed by the daily trend (src/signals/confirmation.py): long
entries need a daily uptrend, exits a daily downtrend; stop-loss and risk
alerts always fire. Filtered-vs-fired counts are kept for GET /health.
"""
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import datetime, time, timedelta, timezone

import pandas as pd
from loguru import logger
from sqlalchemy import and_, insert, select
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from src.backtest import confidence_config
from src.models import Indicator, Position, PriceBar, Signal
from src.signals.confirmation import (
    CONFIRMATION_RULES,  # re-exported: the per-rule confirmation policy
    MAX_STALE_TRADING_DAYS,
    DailyBar,
    DailyTrend,
    Decision,
    decide,
    is_daily,
    reasoning_note,
    required_trend,
    trend_from_bar,
)


def _conf(rule_name: str, ticker: str | None = None, fallback: float = 0.5) -> float:
    """Look up the current calibrated weight for a rule: the ticker's override
    when calibration wrote one, otherwise the rule's `_default`. Read by
    attribute so POST /backtest/apply takes effect immediately (no reload)."""
    weights = confidence_config.CONFIDENCE_WEIGHTS
    override = weights.get(ticker, {}).get(rule_name) if ticker else None
    if override is not None:
        return override
    return weights.get("_default", {}).get(rule_name, fallback)


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


# Calendar days to look back for the last completed daily bar: enough to span
# MAX_STALE_TRADING_DAYS plus a weekend and a holiday.
_DAILY_LOOKBACK_DAYS = MAX_STALE_TRADING_DAYS + 5


async def get_daily_trend(conn: AsyncConnection, ticker: str, as_of: datetime) -> DailyTrend:
    """The daily trend a signal at `as_of` is confirmed against: the most recent
    daily bar from a trading day *before* as_of's (the same day's bar already
    holds that day's close), or "unknown" if there is none recent enough."""
    as_of = as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
    signal_day = as_of.astimezone(timezone.utc).date()
    # Daily bars sit exactly at midnight UTC, so ask for those instants by
    # value: portable, and it skips the intraday bars in between.
    midnights = [
        datetime.combine(signal_day - timedelta(days=n), time(0), tzinfo=timezone.utc)
        for n in range(1, _DAILY_LOOKBACK_DAYS + 1)
    ]
    stmt = (
        select(PriceBar.timestamp, PriceBar.close, Indicator.ema_9, Indicator.ema_21, Indicator.rsi_14)
        .outerjoin(
            Indicator,
            and_(PriceBar.ticker == Indicator.ticker, PriceBar.timestamp == Indicator.timestamp),
        )
        .where(PriceBar.ticker == ticker, PriceBar.timestamp.in_(midnights))
        .order_by(PriceBar.timestamp.desc())
        .limit(1)
    )
    row = (await conn.execute(stmt)).first()
    bar = None
    if row is not None:
        bar = DailyBar(
            day=_ts(row.timestamp).date(), close=float(row.close),
            ema_9=row.ema_9, ema_21=row.ema_21, rsi_14=row.rsi_14,
        )
    daily = trend_from_bar(bar, signal_day)
    if daily.trend == "unknown":
        logger.warning(
            "{}: no daily trend for {} ({}); signals fire unconfirmed",
            ticker, signal_day, daily.reason or f"none in the last {_DAILY_LOOKBACK_DAYS} days",
        )
    return daily


class FilterStats:
    """Fired vs filtered counts for confirmation-gated rules, last 24 hours.

    In memory: counts start from zero when the API restarts.
    """

    WINDOW = timedelta(hours=24)

    def __init__(self, clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self._clock = clock
        self._events: deque[tuple[datetime, str, bool]] = deque()

    def record(self, rule: str, fired: bool) -> None:
        self._events.append((self._clock(), rule, fired))

    def last_24h(self) -> dict[str, dict[str, float]]:
        cutoff = self._clock() - self.WINDOW
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()
        counts: dict[str, list[int]] = {}
        for _, rule, fired in self._events:
            tally = counts.setdefault(rule, [0, 0])
            tally[0 if fired else 1] += 1
        return {
            rule: {"fired": fired, "filtered": filtered, "filter_rate": round(filtered / (fired + filtered), 2)}
            for rule, (fired, filtered) in sorted(counts.items())
        }


FILTER_STATS = FilterStats()


@dataclass
class SignalCandidate:
    ticker: str
    timestamp: datetime
    signal_type: str
    direction: str | None
    confidence: float
    reasoning: str

    def to_row(self, delivered: bool = False) -> dict:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp,
            "signal_type": self.signal_type,
            "direction": self.direction,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "delivered": delivered,
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

    def __init__(self, engine: AsyncEngine, stats: FilterStats = FILTER_STATS) -> None:
        self._engine = engine
        self._stats = stats

    async def evaluate(self, ticker: str) -> list[dict]:
        """Candidates that would be recorded now (confirmed), without recording them."""
        ticker = ticker.strip().upper()
        async with self._engine.connect() as conn:
            history = await self._load_history(conn, ticker, self.HISTORY_LIMIT)
            position = await self._load_position(conn, ticker)
            portfolio = await self._load_portfolio(conn)
            candidates = self._build_candidates(ticker, history, position, portfolio)
            confirmed = [c for c, _ in [await self._confirm(conn, c) for c in candidates] if c is not None]
        return [asdict(c) for c in confirmed]

    async def _confirm(
        self, conn: AsyncConnection, candidate: SignalCandidate
    ) -> tuple[SignalCandidate | None, Decision]:
        """Gate a candidate on the daily trend. None means filtered out."""
        if required_trend(candidate.signal_type) is None:
            return candidate, Decision.NOT_REQUIRED
        daily = await get_daily_trend(conn, candidate.ticker, candidate.timestamp)
        decision = decide(candidate.signal_type, daily)
        if decision is Decision.FILTERED:
            logger.debug(
                "{} {}: filtered by daily trend confirmation (daily {}, needs {})",
                candidate.ticker, candidate.signal_type, daily.trend, required_trend(candidate.signal_type),
            )
            return None, decision
        if decision is Decision.UNCONFIRMED:
            logger.info("{} {}: unconfirmed — no daily data", candidate.ticker, candidate.signal_type)
        note = reasoning_note(daily, decision)
        if note:
            candidate = replace(candidate, reasoning=f"{candidate.reasoning.rstrip()} {note}")
        return candidate, decision

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
                    confirmed, decision = await self._confirm(conn, c)
                    if confirmed is None:
                        self._stats.record(c.signal_type, fired=False)
                        continue
                    c = confirmed

                    # Portfolio-wide signals: emit once per run regardless of carrier ticker.
                    if SIGNAL_CATEGORIES.get(c.signal_type) == "risk" and c.signal_type == "drawdown_alert":
                        if c.signal_type in emitted_portfolio:
                            continue
                        emitted_portfolio.add(c.signal_type)

                    sig_id = await self.record(conn, c)
                    if sig_id is None:
                        continue
                    if decision is not Decision.NOT_REQUIRED:
                        self._stats.record(c.signal_type, fired=True)
                    inserted.append({**asdict(c), "id": sig_id})

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
                    confidence=_conf("oversold_reversal", ticker),
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
                    confidence=_conf("golden_cross", ticker),
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
                    confidence=_conf("breakout", ticker),
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
                    confidence=_conf("overbought_reversal", ticker),
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
                    confidence=_conf("death_cross", ticker),
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
            confidence=_conf("stop_loss_warning", ticker),
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
            confidence=_conf("concentration_risk", ticker),
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
            .limit(limit * 3)
        )
        # Rules run on the 5-minute series; daily bars are confirmation context
        # (get_daily_trend), not the previous bar. Over-fetch, they get skipped.
        rows = [r for r in (await conn.execute(stmt)).all() if not is_daily(_ts(r.timestamp))][:limit]
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

    async def record(
        self, conn: AsyncConnection, candidate: SignalCandidate, delivered: bool = False
    ) -> int | None:
        """Insert a confirmed candidate unless it duplicates one already recorded.

        Shared by the live run and the post-backfill catch-up
        (src/signals/catchup.py). Returns the new id, or None if deduplicated.
        """
        if await self._is_duplicate(conn, candidate):
            logger.debug(
                "dedup'd {} for {} (another within {} of {})",
                candidate.signal_type, candidate.ticker, self.DEDUP_WINDOW, candidate.timestamp,
            )
            return None
        res = await conn.execute(insert(Signal).values(**candidate.to_row(delivered=delivered)))
        logger.info(
            "signal: {} {} ({:.2f}) — {}",
            candidate.ticker, candidate.signal_type, candidate.confidence, candidate.reasoning,
        )
        return res.inserted_primary_key[0]

    async def _is_duplicate(
        self, conn: AsyncConnection, candidate: SignalCandidate
    ) -> bool:
        # By when the signal happened, not when the row was written: a
        # catch-up records days of signals at once, and those must only be
        # deduplicated against signals near their own time.
        window = (candidate.timestamp - self.DEDUP_WINDOW, candidate.timestamp + self.DEDUP_WINDOW)
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
                Signal.timestamp > window[0],
                Signal.timestamp < window[1],
            )
            .limit(1)
        )
        return (await conn.execute(stmt)).first() is not None
