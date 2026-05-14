"""Offline backtest runner.

Replays the signal rules bar-by-bar over historical data with strict no-
lookahead semantics: at each bar i the rule sees only bars [0..i] and is
graded against the forward window [i+1 .. i+forward_window_days].

The output is a per-rule hit rate (`wins / evaluated`) that the dev can
review and — via POST /backtest/apply — push back into
[src/backtest/confidence_config.py](src/backtest/confidence_config.py) as
the calibrated confidence weight used by the live signal engine.

This module deliberately re-implements rule evaluation against a list of
Row objects instead of reusing `SignalEngine._entry_signals` /
`_exit_signals` (which take a pandas DataFrame and would push us toward
loading per-bar history from the DB). Keeping a local `evaluate_rules`
makes the no-lookahead slicing explicit and self-contained.
"""
import statistics
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from loguru import logger
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncEngine

from src.models import Indicator, PriceBar
from src.signals.engine import SIGNAL_CATEGORIES


_BREAKOUT_LOOKBACK = 20
_MIN_EVALUATED_SAMPLES = 30
_CONFIDENCE_FLOOR = 0.3
_CONFIDENCE_CEILING = 0.95
_PORTFOLIO_RULES = {"concentration_risk", "drawdown_alert", "stop_loss_warning"}
_ENTRY_RULES = {"oversold_reversal", "golden_cross", "breakout"}
_EXIT_RULES = {"overbought_reversal", "death_cross"}


@dataclass
class HistoryBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    rsi_14: float | None
    macd_line: float | None
    macd_signal: float | None
    macd_hist: float | None
    ema_9: float | None
    ema_21: float | None


@dataclass
class BacktestEvent:
    ticker: str
    rule_name: str
    signal_bar_timestamp: datetime
    signal_bar_close: float
    forward_close: float | None
    outcome: str  # "win" | "loss" | "insufficient_data" | "skipped"


@dataclass
class RuleStats:
    rule_name: str
    occurrences: int
    evaluated: int
    wins: int
    losses: int
    hit_rate: float | None
    avg_forward_return: float
    suggested_confidence: float
    sample_size_warning: bool


@dataclass
class BacktestReport:
    start_date: date
    end_date: date
    tickers: list[str]
    forward_window_days: int
    outcome_threshold_pct: float
    rules: list[RuleStats]
    total_events: int
    generated_at: datetime


def evaluate_rules(history: list[HistoryBar]) -> list[str]:
    """Return the names of rules that fire at history[-1] given only history[:].

    Mirrors src/signals/engine.py for the price-only rules and intentionally
    skips position/portfolio-conditional rules (stop_loss_warning,
    concentration_risk, drawdown_alert) — they can't be evaluated in a
    per-ticker price backtest.
    """
    if len(history) < 2:
        return []
    last = history[-1]
    prev = history[-2]
    fired: list[str] = []

    rsi = last.rsi_14
    hist_now = last.macd_hist
    hist_prev = prev.macd_hist

    # oversold_reversal
    if (
        rsi is not None and rsi < 30
        and hist_now is not None and hist_prev is not None
        and hist_now > 0 and hist_prev <= 0
    ):
        fired.append("oversold_reversal")

    # golden_cross
    e9, e21, pe9, pe21 = last.ema_9, last.ema_21, prev.ema_9, prev.ema_21
    if None not in (e9, e21, pe9, pe21):
        if pe9 <= pe21 and e9 > e21:
            fired.append("golden_cross")

    # breakout: close above N-day high on above-average volume
    if len(history) >= _BREAKOUT_LOOKBACK + 1:
        lookback = history[-(_BREAKOUT_LOOKBACK + 1):-1]
        prior_high = max(b.high for b in lookback)
        vol_avg = sum(b.volume for b in lookback) / len(lookback)
        if last.close > prior_high and vol_avg > 0 and last.volume > vol_avg:
            fired.append("breakout")

    # overbought_reversal
    if (
        rsi is not None and rsi > 70
        and hist_now is not None and hist_prev is not None
        and hist_now < 0 and hist_prev >= 0
    ):
        fired.append("overbought_reversal")

    # death_cross
    if None not in (e9, e21, pe9, pe21):
        if pe9 >= pe21 and e9 < e21:
            fired.append("death_cross")

    return fired


def _classify(rule: str, signal_close: float, forward_close: float, threshold: float) -> str:
    if rule in _ENTRY_RULES:
        return "win" if forward_close > signal_close * (1 + threshold) else "loss"
    if rule in _EXIT_RULES:
        return "win" if forward_close < signal_close * (1 - threshold) else "loss"
    return "skipped"


def _aggregate(events: list[BacktestEvent]) -> list[RuleStats]:
    stats: list[RuleStats] = []
    # Include every known rule so the report shape is stable even when a rule
    # never fired in the backtest window.
    for rule in SIGNAL_CATEGORIES.keys():
        rule_events = [e for e in events if e.rule_name == rule]
        evaluable = [e for e in rule_events if e.outcome in ("win", "loss")]
        wins = [e for e in evaluable if e.outcome == "win"]
        losses = [e for e in evaluable if e.outcome == "loss"]
        hit_rate = (len(wins) / len(evaluable)) if evaluable else None
        win_returns = [
            (e.forward_close - e.signal_bar_close) / e.signal_bar_close
            for e in wins
            if e.forward_close is not None and e.signal_bar_close
        ]
        avg_forward_return = statistics.fmean(win_returns) if win_returns else 0.0

        if hit_rate is None:
            suggested = _CONFIDENCE_FLOOR  # No data; we won't apply this anyway.
        else:
            suggested = max(_CONFIDENCE_FLOOR, min(_CONFIDENCE_CEILING, hit_rate))

        stats.append(RuleStats(
            rule_name=rule,
            occurrences=len(rule_events),
            evaluated=len(evaluable),
            wins=len(wins),
            losses=len(losses),
            hit_rate=hit_rate,
            avg_forward_return=avg_forward_return,
            suggested_confidence=suggested,
            sample_size_warning=len(evaluable) < _MIN_EVALUATED_SAMPLES,
        ))
    return stats


class BacktestRunner:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def run(
        self,
        tickers: list[str],
        start_date: date,
        end_date: date,
        forward_window_days: int = 5,
        outcome_threshold_pct: float = 0.01,
    ) -> BacktestReport:
        if forward_window_days < 1:
            raise ValueError("forward_window_days must be >= 1")
        if not tickers:
            raise ValueError("at least one ticker required")

        logger.info(
            "backtest starting: {} ticker(s), {} → {}, fwd={}, threshold={:.1%}",
            len(tickers), start_date, end_date, forward_window_days, outcome_threshold_pct,
        )

        events: list[BacktestEvent] = []
        for ticker in tickers:
            ticker = ticker.strip().upper()
            try:
                history = await self._load_history(ticker, start_date, end_date)
            except Exception:
                logger.exception("backtest: failed to load history for {}", ticker)
                continue
            if not history:
                logger.info("backtest {}: no bars in range, skipping", ticker)
                continue

            ticker_events = self._replay_one(
                ticker, history, forward_window_days, outcome_threshold_pct,
            )
            events.extend(ticker_events)
            logger.info(
                "backtest {}: {} bars, {} signal events",
                ticker, len(history), len(ticker_events),
            )

        rules = _aggregate(events)
        report = BacktestReport(
            start_date=start_date,
            end_date=end_date,
            tickers=[t.strip().upper() for t in tickers],
            forward_window_days=forward_window_days,
            outcome_threshold_pct=outcome_threshold_pct,
            rules=rules,
            total_events=len(events),
            generated_at=datetime.now(timezone.utc),
        )
        logger.info(
            "backtest complete: {} total events across {} ticker(s)",
            len(events), len(tickers),
        )
        for rs in rules:
            if rs.evaluated == 0:
                continue
            logger.info(
                "  {}: occ={} eval={} wins={} hit_rate={:.1%} suggested_conf={:.2f}{}",
                rs.rule_name, rs.occurrences, rs.evaluated, rs.wins,
                rs.hit_rate, rs.suggested_confidence,
                " (low sample)" if rs.sample_size_warning else "",
            )
        return report

    def _replay_one(
        self,
        ticker: str,
        history: list[HistoryBar],
        forward_window_days: int,
        threshold: float,
    ) -> list[BacktestEvent]:
        events: list[BacktestEvent] = []
        n = len(history)
        for i in range(n):
            visible = history[: i + 1]
            fired = evaluate_rules(visible)
            if not fired:
                continue
            signal_bar = visible[-1]
            forward_idx = i + forward_window_days
            has_forward = forward_idx < n
            forward_close = history[forward_idx].close if has_forward else None

            for rule in fired:
                if rule in _PORTFOLIO_RULES:
                    rule_outcome = "skipped"
                    fwd = None
                elif not has_forward:
                    rule_outcome = "insufficient_data"
                    fwd = None
                else:
                    fwd = forward_close
                    rule_outcome = _classify(rule, signal_bar.close, fwd, threshold)
                events.append(BacktestEvent(
                    ticker=ticker,
                    rule_name=rule,
                    signal_bar_timestamp=signal_bar.timestamp,
                    signal_bar_close=signal_bar.close,
                    forward_close=fwd,
                    outcome=rule_outcome,
                ))
        return events

    async def _load_history(
        self, ticker: str, start_date: date, end_date: date,
    ) -> list[HistoryBar]:
        start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
        end_dt = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc)
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
            .where(PriceBar.timestamp >= start_dt)
            .where(PriceBar.timestamp <= end_dt)
            .order_by(PriceBar.timestamp.asc())
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).all()
        out: list[HistoryBar] = []
        for r in rows:
            ts = r.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            out.append(HistoryBar(
                timestamp=ts,
                open=float(r.open), high=float(r.high), low=float(r.low),
                close=float(r.close), volume=float(r.volume),
                rsi_14=r.rsi_14, macd_line=r.macd_line, macd_signal=r.macd_signal,
                macd_hist=r.macd_hist, ema_9=r.ema_9, ema_21=r.ema_21,
            ))
        return out
