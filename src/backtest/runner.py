"""Offline backtest runner.

Replays the signal rules bar-by-bar over historical data with strict no-
lookahead semantics: at each bar i the rule sees only bars [0..i] and is
graded against the forward window [i+1 .. i+forward_window_days].

The output is a per-rule hit rate (`wins / evaluated`) that the dev can
review and — via POST /backtest/apply, or automatically after a backfill —
apply as the calibrated confidence weight used by the live signal engine
(saved to data/confidence_weights.json; defaults in
[src/backtest/confidence_config.py](src/backtest/confidence_config.py)).

`run_walkforward` is the more honest variant: it splits the range into
rolling train/test windows and reports each rule's out-of-sample hit rate per
test window, how much that swings between windows, and a suggested confidence
discounted for the swing. Rules are fixed, not fitted, so the train window
only guarantees the ticker had enough history for its indicators to settle.

Both runners take a `timeframe`: "daily" (the default; years of history, so
walk-forward works) or "intraday" (the 5-minute bars live signals fire on;
Yahoo serves ~60 days, more as backfills accumulate). Either way
`forward_window_days` is in trading days, and every signal must pass the same
daily-trend confirmation as live (src/signals/confirmation.py), judged on the
last *completed* day — so occurrence counts drop versus unconfirmed replays.
In the daily timeframe that rules out golden/death crosses entirely: the day
before a daily cross always has the opposite EMA order.

This module deliberately re-implements rule evaluation against a list of
Row objects instead of reusing `SignalEngine._entry_signals` /
`_exit_signals` (which take a pandas DataFrame and would push us toward
loading per-bar history from the DB). Keeping a local `evaluate_rules`
makes the no-lookahead slicing explicit and self-contained.
"""
import bisect
import calendar
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal

from loguru import logger
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncEngine

from src.models import Indicator, PriceBar
from src.signals.confirmation import (
    DailyBar,
    Decision,
    decide,
    is_daily,
    last_completed_bar,
    trend_from_bar,
)
from src.signals.engine import SIGNAL_CATEGORIES

Timeframe = Literal["daily", "intraday"]
_TIMEFRAMES = ("daily", "intraday")


_BREAKOUT_LOOKBACK = 20
# Bars a rule reads: the breakout lookback plus the bar being evaluated.
RULE_WINDOW_BARS = _BREAKOUT_LOOKBACK + 1
_MIN_EVALUATED_SAMPLES = 30
_CONFIDENCE_FLOOR = 0.3
_CONFIDENCE_CEILING = 0.95
_PORTFOLIO_RULES = {"concentration_risk", "drawdown_alert", "stop_loss_warning"}
_ENTRY_RULES = {"oversold_reversal", "golden_cross", "breakout"}
_EXIT_RULES = {"overbought_reversal", "death_cross"}

# Walk-forward
_MIN_WINDOW_OCCURRENCES = 20
# How hard instability (std/mean of per-window hit rates) discounts confidence:
# a rule whose hit rate swings wildly loses up to 30% of its pooled hit rate.
_STABILITY_WEIGHT = 0.3
# A ticker counts in a window only if its history starts by train_start (plus
# this slack for weekends/holidays), so its indicators had the train window to
# warm up.
_TRAIN_COVERAGE_SLACK_DAYS = 7


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
    outcome: str  # "win" | "loss" | "insufficient_data" | "skipped" | "filtered"


@dataclass
class TickerRuleStats:
    """One rule's single-window result for one ticker."""

    occurrences: int
    evaluated: int
    wins: int
    hit_rate: float | None
    suggested_confidence: float
    filtered: int = 0  # dropped by daily-trend confirmation; not in occurrences


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
    filtered: int = 0  # dropped by daily-trend confirmation; not in occurrences
    by_ticker: dict[str, TickerRuleStats] = field(default_factory=dict)


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
    timeframe: Timeframe = "daily"


@dataclass
class WalkforwardWindow:
    """Half-open windows: [train_start, train_end) then [test_start, test_end)."""

    train_start: date
    train_end: date
    test_start: date
    test_end: date


@dataclass
class WalkforwardStats:
    """Out-of-sample results across test windows. Hit-rate fields are None when
    nothing was graded; `window_hit_rates` covers only windows with graded
    events, in chronological order."""

    windows_evaluated: int
    total_occurrences: int  # confirmed signals only
    total_filtered: int  # dropped by daily-trend confirmation
    total_evaluated: int
    aggregate_hit_rate: float | None  # pooled: total wins / total evaluated
    window_hit_rates: list[float]
    hit_rate_mean: float | None
    hit_rate_std: float | None  # population std of window_hit_rates
    hit_rate_min: float | None
    hit_rate_max: float | None
    stability_score: float | None  # 1 - std/mean, clamped to [0, 1]
    sample_size_warning: bool  # any window with < 20 occurrences
    suggested_confidence: float


@dataclass
class WalkforwardTickerStats(WalkforwardStats):
    ticker: str = ""


@dataclass
class WalkforwardRuleStats(WalkforwardStats):
    rule_name: str = ""
    by_ticker: dict[str, WalkforwardTickerStats] = field(default_factory=dict)


@dataclass
class WalkforwardReport:
    start_date: date
    end_date: date
    train_window_months: int
    test_window_months: int
    step_months: int
    forward_window_days: int
    outcome_threshold_pct: float
    windows: list[WalkforwardWindow]
    rules: list[WalkforwardRuleStats]
    tickers: list[str]
    generated_at: datetime
    timeframe: Timeframe = "daily"


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


def _clamp_confidence(value: float) -> float:
    return max(_CONFIDENCE_FLOOR, min(_CONFIDENCE_CEILING, value))


def _ticker_rule_stats(all_events: list[BacktestEvent]) -> TickerRuleStats:
    events = [e for e in all_events if e.outcome != "filtered"]
    evaluable = [e for e in events if e.outcome in ("win", "loss")]
    wins = sum(e.outcome == "win" for e in evaluable)
    hit_rate = wins / len(evaluable) if evaluable else None
    return TickerRuleStats(
        occurrences=len(events),
        evaluated=len(evaluable),
        wins=wins,
        hit_rate=hit_rate,
        suggested_confidence=_CONFIDENCE_FLOOR if hit_rate is None else _clamp_confidence(hit_rate),
        filtered=len(all_events) - len(events),
    )


def _aggregate(events: list[BacktestEvent]) -> list[RuleStats]:
    stats: list[RuleStats] = []
    # Include every known rule so the report shape is stable even when a rule
    # never fired in the backtest window.
    for rule in SIGNAL_CATEGORIES.keys():
        all_rule_events = [e for e in events if e.rule_name == rule]
        rule_events = [e for e in all_rule_events if e.outcome != "filtered"]
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

        by_ticker: dict[str, list[BacktestEvent]] = defaultdict(list)
        for e in all_rule_events:
            by_ticker[e.ticker].append(e)

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
            filtered=len(all_rule_events) - len(rule_events),
            by_ticker={t: _ticker_rule_stats(es) for t, es in sorted(by_ticker.items())},
        ))
    return stats


# ---------- walk-forward ----------


def _add_months(d: date, months: int) -> date:
    """`d` moved by whole months, clamped to the target month's last day."""
    year, month0 = divmod(d.month - 1 + months, 12)
    year += d.year
    day = min(d.day, calendar.monthrange(year, month0 + 1)[1])
    return date(year, month0 + 1, day)


def walkforward_windows(
    start_date: date,
    end_date: date,
    train_window_months: int,
    test_window_months: int,
    step_months: int,
) -> list[WalkforwardWindow]:
    """Rolling windows from start_date, stepping until a test window would end
    after end_date.

    Every boundary is an offset from start_date itself, not from the previous
    boundary: chaining month additions drifts after a clamp (Jan 31 -> Feb 28
    -> Mar 28) and would leave days between consecutive test windows.
    """
    windows: list[WalkforwardWindow] = []
    i = 0
    while True:
        offset = step_months * i
        train_start = _add_months(start_date, offset)
        train_end = _add_months(start_date, offset + train_window_months)
        test_end = _add_months(start_date, offset + train_window_months + test_window_months)
        if test_end > end_date:
            return windows
        windows.append(WalkforwardWindow(train_start, train_end, train_end, test_end))
        i += 1


def _add_trading_days(ts: datetime, days: int) -> datetime:
    """`ts` moved forward `days` weekdays, same time of day (holidays ignored)."""
    while days > 0:
        ts += timedelta(days=1)
        if ts.weekday() < 5:
            days -= 1
    return ts


@dataclass
class _Tally:
    occurrences: int = 0
    evaluated: int = 0
    wins: int = 0
    filtered: int = 0

    def add(self, event: BacktestEvent) -> None:
        if event.outcome == "filtered":
            self.filtered += 1
            return
        self.occurrences += 1
        if event.outcome in ("win", "loss"):
            self.evaluated += 1
            self.wins += event.outcome == "win"


def summarize_windows(tallies: list[_Tally]) -> WalkforwardStats:
    """Aggregate one rule's per-window tallies (chronological) into stats."""
    graded = [t for t in tallies if t.evaluated > 0]
    rates = [t.wins / t.evaluated for t in graded]
    total_evaluated = sum(t.evaluated for t in tallies)
    pooled = sum(t.wins for t in tallies) / total_evaluated if total_evaluated else None

    mean = statistics.fmean(rates) if rates else None
    std = statistics.pstdev(rates) if rates else None
    if mean is None or std is None:
        stability = None
    elif mean == 0:
        stability = 0.0  # never right in any window: nothing stable to trust
    else:
        stability = max(0.0, min(1.0, 1 - std / mean))

    if pooled is None or stability is None:
        suggested = _CONFIDENCE_FLOOR  # No data; apply won't use it.
    else:
        # A stable 65% rule should outrank a 75% rule that swings by regime.
        suggested = _clamp_confidence(pooled * (1 - _STABILITY_WEIGHT * (1 - stability)))

    return WalkforwardStats(
        windows_evaluated=len(graded),
        total_occurrences=sum(t.occurrences for t in tallies),
        total_filtered=sum(t.filtered for t in tallies),
        total_evaluated=total_evaluated,
        aggregate_hit_rate=pooled,
        window_hit_rates=rates,
        hit_rate_mean=mean,
        hit_rate_std=std,
        hit_rate_min=min(rates) if rates else None,
        hit_rate_max=max(rates) if rates else None,
        stability_score=stability,
        sample_size_warning=any(t.occurrences < _MIN_WINDOW_OCCURRENCES for t in tallies),
        suggested_confidence=suggested,
    )


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
        timeframe: Timeframe = "daily",
    ) -> BacktestReport:
        if forward_window_days < 1:
            raise ValueError("forward_window_days must be >= 1")
        if not tickers:
            raise ValueError("at least one ticker required")
        if timeframe not in _TIMEFRAMES:
            raise ValueError(f"timeframe must be one of {_TIMEFRAMES}")

        logger.info(
            "backtest starting: {} ticker(s), {} → {}, fwd={}, threshold={:.1%}",
            len(tickers), start_date, end_date, forward_window_days, outcome_threshold_pct,
        )

        events: list[BacktestEvent] = []
        for ticker in tickers:
            ticker = ticker.strip().upper()
            try:
                history = await self.load_history(ticker, start_date, end_date, timeframe)
                daily = await self.load_daily_context(ticker, start_date, end_date)
            except Exception:
                logger.exception("backtest: failed to load history for {}", ticker)
                continue
            if not history:
                logger.info("backtest {}: no {} bars in range, skipping", ticker, timeframe)
                continue

            ticker_events = self._replay_one(
                ticker, history, daily, forward_window_days, outcome_threshold_pct, timeframe,
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
            timeframe=timeframe,
        )
        logger.info(
            "backtest complete: {} total events across {} ticker(s)",
            len(events), len(tickers),
        )
        for rs in rules:
            if rs.evaluated == 0:
                continue
            logger.info(
                "  {}: occ={} filtered={} eval={} wins={} hit_rate={:.1%} suggested_conf={:.2f}{}",
                rs.rule_name, rs.occurrences, rs.filtered, rs.evaluated, rs.wins,
                rs.hit_rate, rs.suggested_confidence,
                " (low sample)" if rs.sample_size_warning else "",
            )
        return report

    async def run_walkforward(
        self,
        tickers: list[str],
        start_date: date,
        end_date: date,
        train_window_months: int = 12,
        test_window_months: int = 1,
        step_months: int = 1,
        forward_window_days: int = 5,
        outcome_threshold_pct: float = 0.01,
        timeframe: Timeframe = "daily",
    ) -> WalkforwardReport:
        """Out-of-sample hit rates per rolling test window, aggregated per rule.

        Each ticker is replayed once over the whole range — a rule at bar i
        sees only bars [0..i], so that is the same as replaying per window —
        and every event is then credited to the test window its signal bar
        falls in. Grading may read bars past test_end; the rule never does.
        """
        if forward_window_days < 1:
            raise ValueError("forward_window_days must be >= 1")
        if min(train_window_months, test_window_months, step_months) < 1:
            raise ValueError("train_window_months, test_window_months and step_months must be >= 1")
        if not tickers:
            raise ValueError("at least one ticker required")
        if timeframe not in _TIMEFRAMES:
            raise ValueError(f"timeframe must be one of {_TIMEFRAMES}")
        windows = walkforward_windows(
            start_date, end_date, train_window_months, test_window_months, step_months,
        )
        if not windows:
            raise ValueError(
                f"{start_date} → {end_date} is too short for one window of "
                f"{train_window_months} train + {test_window_months} test month(s)"
            )
        symbols = [t.strip().upper() for t in tickers]
        logger.info(
            "walk-forward starting: {} ticker(s), {} → {}, {} window(s) of {}m train / {}m test, step {}m",
            len(symbols), start_date, end_date, len(windows),
            train_window_months, test_window_months, step_months,
        )

        # tallies[(rule, ticker)][window index]; None where the ticker lacked
        # history covering that window's train period.
        tallies: dict[tuple[str, str], list[_Tally | None]] = {}
        for ticker in symbols:
            try:
                history = await self.load_history(ticker, start_date, end_date, timeframe)
                daily = await self.load_daily_context(ticker, start_date, end_date)
            except Exception:
                logger.exception("walk-forward: failed to load history for {}", ticker)
                continue
            if not history:
                logger.info("walk-forward {}: no {} bars in range, skipping", ticker, timeframe)
                continue
            first_day = history[0].timestamp.date()
            eligible = [
                first_day <= w.train_start + timedelta(days=_TRAIN_COVERAGE_SLACK_DAYS)
                for w in windows
            ]
            events = self._replay_one(
                ticker, history, daily, forward_window_days, outcome_threshold_pct, timeframe,
            )
            for rule in SIGNAL_CATEGORIES:
                row: list[_Tally | None] = [_Tally() if ok else None for ok in eligible]
                for e in events:
                    if e.rule_name != rule:
                        continue
                    day = e.signal_bar_timestamp.date()
                    for idx, w in enumerate(windows):
                        tally = row[idx]
                        if tally is not None and w.test_start <= day < w.test_end:
                            tally.add(e)
                tallies[(rule, ticker)] = row
            logger.info(
                "walk-forward {}: {} bars, {} events, in {}/{} window(s)",
                ticker, len(history), len(events), sum(eligible), len(windows),
            )

        rules: list[WalkforwardRuleStats] = []
        for rule in SIGNAL_CATEGORIES:
            pooled = [_Tally() for _ in windows]
            by_ticker: dict[str, WalkforwardTickerStats] = {}
            for (name, ticker), row in sorted(tallies.items()):
                if name != rule:
                    continue
                for idx, tally in enumerate(row):
                    if tally is not None:
                        pooled[idx].occurrences += tally.occurrences
                        pooled[idx].evaluated += tally.evaluated
                        pooled[idx].wins += tally.wins
                        pooled[idx].filtered += tally.filtered
                own = [t for t in row if t is not None]
                if sum(t.occurrences + t.filtered for t in own):
                    by_ticker[ticker] = WalkforwardTickerStats(
                        **asdict(summarize_windows(own)), ticker=ticker,
                    )
            rules.append(WalkforwardRuleStats(
                **asdict(summarize_windows(pooled)), rule_name=rule, by_ticker=by_ticker,
            ))

        report = WalkforwardReport(
            start_date=start_date,
            end_date=end_date,
            train_window_months=train_window_months,
            test_window_months=test_window_months,
            step_months=step_months,
            forward_window_days=forward_window_days,
            outcome_threshold_pct=outcome_threshold_pct,
            windows=windows,
            rules=rules,
            tickers=symbols,
            generated_at=datetime.now(timezone.utc),
            timeframe=timeframe,
        )
        for rs in rules:
            if rs.total_evaluated == 0:
                continue
            logger.info(
                "  {}: windows={} filtered={} eval={} pooled={:.1%} mean={:.1%} std={:.1%} stability={:.2f} suggested_conf={:.2f}{}",
                rs.rule_name, rs.windows_evaluated, rs.total_filtered, rs.total_evaluated, rs.aggregate_hit_rate,
                rs.hit_rate_mean, rs.hit_rate_std, rs.stability_score, rs.suggested_confidence,
                " (low sample)" if rs.sample_size_warning else "",
            )
        return report

    def _replay_one(
        self,
        ticker: str,
        history: list[HistoryBar],
        daily: list[DailyBar],
        forward_window_days: int,
        threshold: float,
        timeframe: Timeframe = "daily",
    ) -> list[BacktestEvent]:
        """Every rule firing in `history`, graded `forward_window_days` trading
        days later, after the same daily-trend confirmation live applies."""
        events: list[BacktestEvent] = []
        n = len(history)
        timestamps = [b.timestamp for b in history]
        for i in range(n):
            # A rule reads at most the breakout lookback plus the current bar.
            fired = evaluate_rules(history[max(0, i - _BREAKOUT_LOOKBACK): i + 1])
            if not fired:
                continue
            signal_bar = history[i]
            if timeframe == "daily":
                forward_idx = i + forward_window_days
            else:
                target = _add_trading_days(signal_bar.timestamp, forward_window_days)
                forward_idx = bisect.bisect_left(timestamps, target)
            forward_close = history[forward_idx].close if forward_idx < n else None
            daily_trend = trend_from_bar(
                last_completed_bar(daily, signal_bar.timestamp), signal_bar.timestamp.date(),
            )

            for rule in fired:
                fwd = None
                if rule in _PORTFOLIO_RULES:
                    rule_outcome = "skipped"
                elif decide(rule, daily_trend) is Decision.FILTERED:
                    rule_outcome = "filtered"
                elif forward_close is None:
                    rule_outcome = "insufficient_data"
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

    async def load_daily_context(self, ticker: str, start_date: date, end_date: date) -> list[DailyBar]:
        """Daily bars for confirmation, from a little before start_date so the
        first signals have a previous completed day to be judged on."""
        bars = await self.load_history(ticker, start_date - timedelta(days=15), end_date, "daily")
        return [
            DailyBar(day=b.timestamp.date(), close=b.close, ema_9=b.ema_9, ema_21=b.ema_21, rsi_14=b.rsi_14)
            for b in bars
        ]

    async def load_history(
        self, ticker: str, start_date: date, end_date: date, timeframe: Timeframe = "daily",
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
            # One timeframe per series: daily bars (stamped at UTC midnight by
            # the backfill) or the 5-minute bars between them, never both.
            if is_daily(ts) != (timeframe == "daily"):
                continue
            out.append(HistoryBar(
                timestamp=ts,
                open=float(r.open), high=float(r.high), low=float(r.low),
                close=float(r.close), volume=float(r.volume),
                rsi_14=r.rsi_14, macd_line=r.macd_line, macd_signal=r.macd_signal,
                macd_hist=r.macd_hist, ema_9=r.ema_9, ema_21=r.ema_21,
            ))
        return out
