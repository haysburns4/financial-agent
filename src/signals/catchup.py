"""Catch up on signals the live engine missed while the app wasn't running.

Live signals are only checked while the app runs during market hours with
E-Trade logged in, and each check looks at the newest bar only. After every
backfill (which fetches 5-minute bars from Yahoo, no login needed) this replays
the same rules over the bars since the last check — at most
MAX_CATCHUP_TRADING_DAYS back — with the same daily-trend confirmation, and
records what fired at the bar's real time.

Rule evaluation and confirmation are the backtest's (src/backtest/runner.py,
src/signals/confirmation.py); recording, dedup and confidence are the live
engine's (`SignalEngine.record`, `_conf`). Caught-up signals are stored as
already delivered, so a backfill never floods Discord with old signals.
"""
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from loguru import logger
from sqlalchemy import func, insert, select
from sqlalchemy.ext.asyncio import AsyncEngine

from src.backtest.runner import RULE_WINDOW_BARS, BacktestRunner, HistoryBar, evaluate_rules
from src.models import PipelineRun
from src.signals.confirmation import (
    DailyBar,
    Decision,
    decide,
    last_completed_bar,
    minus_trading_days,
    reasoning_note,
    trend_from_bar,
)
from src.signals.engine import SignalCandidate, SignalEngine, _conf

CATCHUP_PIPELINE = "signal_catchup"
MAX_CATCHUP_TRADING_DAYS = 5
# Runs that evaluated signals: the live price pipeline and earlier catch-ups.
_CHECKING_PIPELINES = ("price_pipeline", CATCHUP_PIPELINE)
_DIRECTIONS = {"oversold_reversal": "long", "golden_cross": "long", "breakout": "long"}


@dataclass
class CatchUpResult:
    since: datetime
    until: datetime
    recorded: dict[str, int] = field(default_factory=dict)  # per rule
    filtered: int = 0  # dropped by daily-trend confirmation
    duplicates: int = 0  # already recorded near the same time


def describe(rule: str, window: list[HistoryBar]) -> str:
    """The rule's reasoning, in the live engine's words."""
    last, prev = window[-1], window[-2]
    if rule in ("oversold_reversal", "overbought_reversal"):
        side = "<30" if rule == "oversold_reversal" else ">70"
        turn = "positive" if rule == "oversold_reversal" else "negative"
        return (
            f"RSI={last.rsi_14:.1f} ({side}), MACD histogram flipped {turn} "
            f"({prev.macd_hist:.3f} → {last.macd_hist:.3f})"
        )
    if rule in ("golden_cross", "death_cross"):
        way = "above" if rule == "golden_cross" else "below"
        return f"EMA9 ({last.ema_9:.2f}) crossed {way} EMA21 ({last.ema_21:.2f})"
    if rule == "breakout":
        lookback = window[:-1]
        prior_high = max(b.high for b in lookback)
        vol_avg = sum(b.volume for b in lookback) / len(lookback)
        return (
            f"Close {last.close:.2f} > {len(lookback)}-bar high {prior_high:.2f} "
            f"on volume {last.volume:.0f} (avg {vol_avg:.0f})"
        )
    return rule


async def last_check(engine: AsyncEngine) -> datetime | None:
    async with engine.connect() as conn:
        last = await conn.scalar(
            select(func.max(PipelineRun.completed_at)).where(
                PipelineRun.pipeline.in_(_CHECKING_PIPELINES), PipelineRun.status == "ok",
            )
        )
    if last is None:
        return None
    return last if last.tzinfo else last.replace(tzinfo=timezone.utc)


async def catch_up(
    engine: AsyncEngine,
    tickers: list[str],
    now: datetime | None = None,
    signal_engine: SignalEngine | None = None,
) -> CatchUpResult:
    now = now or datetime.now(timezone.utc)
    floor = minus_trading_days(now, MAX_CATCHUP_TRADING_DAYS)
    last = await last_check(engine)
    since = max(last, floor) if last else floor
    result = CatchUpResult(since=since, until=now)
    runner = BacktestRunner(engine)
    recorder = signal_engine or SignalEngine(engine)
    symbols = [t.strip().upper() for t in tickers]

    # Read everything first, then write in one transaction. A week of calendar
    # days before `since` covers the rules' 20-bar lookback.
    load_from = (since - timedelta(days=7)).date()
    loaded: dict[str, tuple[list[HistoryBar], list[DailyBar]]] = {}
    for ticker in symbols:
        loaded[ticker] = (
            await runner.load_history(ticker, load_from, now.date(), "intraday"),
            await runner.load_daily_context(ticker, load_from, now.date()),
        )

    recorded: Counter[str] = Counter()
    async with engine.begin() as conn:
        for ticker, (history, daily) in loaded.items():
            for i, bar in enumerate(history):
                if bar.timestamp <= since or bar.timestamp > now:
                    continue
                window = history[max(0, i - RULE_WINDOW_BARS + 1): i + 1]
                fired = evaluate_rules(window)
                if not fired:
                    continue
                trend = trend_from_bar(last_completed_bar(daily, bar.timestamp), bar.timestamp.date())
                for rule in fired:
                    decision = decide(rule, trend)
                    if decision is Decision.FILTERED:
                        result.filtered += 1
                        continue
                    reasoning = " ".join(
                        part for part in (
                            describe(rule, window) + ".",
                            "Caught up after a backfill.",
                            reasoning_note(trend, decision),
                        ) if part
                    )
                    candidate = SignalCandidate(
                        ticker=ticker, timestamp=bar.timestamp, signal_type=rule,
                        direction=_DIRECTIONS.get(rule), confidence=_conf(rule, ticker),
                        reasoning=reasoning,
                    )
                    if await recorder.record(conn, candidate, delivered=True) is None:
                        result.duplicates += 1
                    else:
                        recorded[rule] += 1
        # completed_at marks how far bars were checked, so the next catch-up
        # starts exactly where this one stopped.
        await conn.execute(insert(PipelineRun).values(
            pipeline=CATCHUP_PIPELINE, started_at=now, completed_at=now,
            status="ok", tickers_processed=len(symbols),
        ))

    result.recorded = dict(sorted(recorded.items()))
    logger.info(
        "signal catch-up {} → {}: {} recorded, {} filtered by daily trend, {} duplicates",
        since, now, sum(recorded.values()), result.filtered, result.duplicates,
    )
    return result
