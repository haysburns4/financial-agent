"""Multi-timeframe confirmation: gate 5-minute signals on the daily trend.

Shared by the live signal engine (src/signals/engine.py) and the backtests
(src/backtest/runner.py), so a rule is confirmed the same way in both — the
backtest's hit rates are only worth calibrating on if they match live.

Two rules keep it honest:
  - The daily trend is read from the last *completed* trading day before the
    signal. Daily bars are stamped at midnight UTC but carry the whole day's
    close, so the same day's bar would let a 10:30 signal see 16:00's price.
  - A daily bar more than MAX_STALE_TRADING_DAYS old says nothing about today:
    the trend is "unknown" and the signal fires unconfirmed rather than being
    filtered on stale evidence.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import StrEnum
from typing import Literal

Trend = Literal["up", "down", "flat", "unknown"]

# None means the rule always fires: stop-loss and risk alerts protect capital,
# they are not trying to find an edge.
CONFIRMATION_RULES: dict[str, dict[str, Trend | None]] = {
    "oversold_reversal": {"require_daily_trend": "up"},
    "golden_cross": {"require_daily_trend": "up"},
    "breakout": {"require_daily_trend": "up"},
    "overbought_reversal": {"require_daily_trend": "down"},
    "death_cross": {"require_daily_trend": "down"},
    "stop_loss_warning": {"require_daily_trend": None},
    "concentration_risk": {"require_daily_trend": None},
    "drawdown_alert": {"require_daily_trend": None},
}

MAX_STALE_TRADING_DAYS = 5


def required_trend(signal_type: str) -> Trend | None:
    return CONFIRMATION_RULES.get(signal_type, {}).get("require_daily_trend")


def is_daily(ts: datetime) -> bool:
    """Daily bars are the ones the backfill stamps at midnight UTC."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).timetz() == time(0, tzinfo=timezone.utc)


def trading_days_between(earlier: date, later: date) -> int:
    """Weekdays in (earlier, later]: Friday's bar seen on Monday is 1 day old.
    Ignores exchange holidays, so a holiday week reads one day staler."""
    days, d = 0, earlier
    while d < later:
        d += timedelta(days=1)
        if d.weekday() < 5:
            days += 1
    return days


def minus_trading_days(ts: datetime, days: int) -> datetime:
    """`ts` moved back `days` weekdays, same time of day (holidays ignored)."""
    while days > 0:
        ts -= timedelta(days=1)
        if ts.weekday() < 5:
            days -= 1
    return ts


def classify_trend(ema_9: float | None, ema_21: float | None, close: float | None) -> Trend:
    if ema_9 is None or ema_21 is None or close is None:
        return "unknown"
    if ema_9 > ema_21 and close > ema_21:
        return "up"
    if ema_9 < ema_21 and close < ema_21:
        return "down"
    return "flat"


@dataclass(frozen=True)
class DailyBar:
    """The daily fields confirmation needs; `day` is the bar's trading date."""

    day: date
    close: float
    ema_9: float | None
    ema_21: float | None
    rsi_14: float | None


@dataclass(frozen=True)
class DailyTrend:
    trend: Trend
    ema_9: float | None
    ema_21: float | None
    rsi_14: float | None
    last_close: float | None
    bars_since_update: int | None  # trading days between that bar and the signal
    reason: str | None = None  # why the trend is "unknown"


def trend_from_bar(bar: DailyBar | None, as_of: date) -> DailyTrend:
    """The trend a signal on `as_of` should be confirmed against."""
    if bar is None:
        return DailyTrend("unknown", None, None, None, None, None, "no daily bar")
    age = trading_days_between(bar.day, as_of)
    if age > MAX_STALE_TRADING_DAYS:
        return DailyTrend(
            "unknown", bar.ema_9, bar.ema_21, bar.rsi_14, bar.close, age,
            f"daily bar is {age} trading days old",
        )
    trend = classify_trend(bar.ema_9, bar.ema_21, bar.close)
    reason = "daily bar has no EMAs" if trend == "unknown" else None
    return DailyTrend(trend, bar.ema_9, bar.ema_21, bar.rsi_14, bar.close, age, reason)


def last_completed_bar(daily: list[DailyBar], when: datetime) -> DailyBar | None:
    """The newest bar in `daily` (chronological) from a day before `when`'s."""
    idx = bisect.bisect_left([b.day for b in daily], when.astimezone(timezone.utc).date())
    return daily[idx - 1] if idx > 0 else None


class Decision(StrEnum):
    NOT_REQUIRED = "not_required"
    CONFIRMED = "confirmed"
    UNCONFIRMED = "unconfirmed"  # no usable daily data: fires anyway
    FILTERED = "filtered"


def decide(signal_type: str, daily: DailyTrend) -> Decision:
    required = required_trend(signal_type)
    if required is None:
        return Decision.NOT_REQUIRED
    if daily.trend == "unknown":
        return Decision.UNCONFIRMED
    return Decision.CONFIRMED if daily.trend == required else Decision.FILTERED


def reasoning_note(daily: DailyTrend, decision: Decision) -> str | None:
    """The sentence appended to a persisted signal's reasoning."""
    if decision is Decision.CONFIRMED and daily.trend == "up":
        return "Confirmed by daily uptrend (EMA9 > EMA21, close above EMA21)."
    if decision is Decision.CONFIRMED and daily.trend == "down":
        return "Confirmed by daily downtrend (EMA9 < EMA21, close below EMA21)."
    if decision is Decision.UNCONFIRMED:
        return f"Unconfirmed: {daily.reason or 'no daily data'}."
    return None
