"""Daily-trend confirmation of 5-minute signals, live and in the backtest replay."""
from datetime import date, datetime, time, timedelta, timezone

import pytest
from sqlalchemy import insert, select

from src.backtest.runner import BacktestRunner, HistoryBar
from src.models import Indicator, PriceBar, Signal
from src.pipelines.backfill_pipeline import BackfillPipeline
from src.signals.confirmation import (
    DailyBar,
    Decision,
    classify_trend,
    decide,
    last_completed_bar,
    trading_days_between,
    trend_from_bar,
)
from src.signals.engine import FilterStats, SignalEngine, get_daily_trend

UTC = timezone.utc
MON, TUE, WED, THU, FRI = (date(2026, 10, d) for d in range(5, 10))
UP = dict(ema_9=3.0, ema_21=2.0)
FLAT = dict(ema_9=1.0, ema_21=2.0)  # EMA9 below EMA21 but close (100) above it


def _at(day: date, hour: int = 0, minute: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=UTC)


# ---------- the policy ----------


@pytest.mark.parametrize(
    ("ema_9", "ema_21", "close", "trend"),
    [(3, 2, 2.5, "up"), (1, 2, 1.5, "down"), (3, 2, 1.5, "flat"), (1, 2, 2.5, "flat"), (None, 2, 1, "unknown")],
)
def test_classify_trend(ema_9, ema_21, close, trend):
    assert classify_trend(ema_9, ema_21, close) == trend


def test_staleness_counts_trading_days():
    assert trading_days_between(FRI - timedelta(days=7), MON) == 1  # Friday -> Monday
    fresh = trend_from_bar(DailyBar(MON - timedelta(days=7), 100, 3, 2, 50), MON)  # 5 trading days
    stale = trend_from_bar(DailyBar(MON - timedelta(days=8), 100, 3, 2, 50), MON)  # 6
    assert fresh.trend == "up" and fresh.bars_since_update == 5
    assert stale.trend == "unknown" and "6 trading days old" in (stale.reason or "")


def test_decisions():
    up, down = trend_from_bar(DailyBar(MON, 100, **UP, rsi_14=50), TUE), trend_from_bar(None, TUE)
    assert decide("golden_cross", up) is Decision.CONFIRMED
    assert decide("death_cross", up) is Decision.FILTERED
    assert decide("breakout", down) is Decision.UNCONFIRMED  # no daily bar: fires unconfirmed
    assert decide("stop_loss_warning", up) is Decision.NOT_REQUIRED


def test_confirmation_never_reads_the_signal_days_own_bar():
    daily = [DailyBar(MON, 100, **UP, rsi_14=None), DailyBar(TUE, 100, **FLAT, rsi_14=None)]
    assert last_completed_bar(daily, _at(TUE, 15)) == daily[0]
    assert last_completed_bar(daily, _at(MON, 15)) is None


# ---------- backtest replay ----------


def _bar(ts: datetime, close: float = 100.0, volume: float = 1000.0, **ind: float) -> HistoryBar:
    return HistoryBar(
        timestamp=ts, open=100.0, high=100.0, low=97.0, close=close, volume=volume,
        rsi_14=None, macd_line=None, macd_signal=None, macd_hist=None,
        ema_9=ind.get("ema_9", 3.0), ema_21=ind.get("ema_21", 2.0),
    )


def _day_of_5min(day: date, start_hour: int = 14, count: int = 24) -> list[HistoryBar]:
    first = _at(day, start_hour)
    return [_bar(first + timedelta(minutes=5 * i)) for i in range(count)]


def _breakout_on(bars: list[HistoryBar], idx: int) -> None:
    bars[idx] = _bar(bars[idx].timestamp, close=101.0, volume=2000.0)


def test_intraday_replay_confirms_on_the_previous_day_and_grades_trading_days_ahead():
    # Monday up, Tuesday flat: a Tuesday signal is judged on Monday (up ->
    # fires), a Wednesday signal on Tuesday (flat -> filtered), whatever the
    # signal day's own daily bar says.
    daily = [DailyBar(MON, 100, **UP, rsi_14=None), DailyBar(TUE, 100, **FLAT, rsi_14=None),
             DailyBar(WED, 100, **UP, rsi_14=None)]
    bars = _day_of_5min(TUE) + _day_of_5min(WED) + _day_of_5min(THU) + _day_of_5min(FRI)
    tue, wed = 22, 24 + 22
    _breakout_on(bars, tue)
    _breakout_on(bars, wed)
    # Two trading days after Tuesday 15:50 is Thursday 15:50.
    thu_same_time = next(i for i, b in enumerate(bars) if b.timestamp == bars[tue].timestamp + timedelta(days=2))
    bars[thu_same_time] = _bar(bars[thu_same_time].timestamp, close=103.0)

    events = BacktestRunner(engine=None)._replay_one("AAPL", bars, daily, 2, 0.01, "intraday")
    breakouts = [e for e in events if e.rule_name == "breakout"]

    assert [(e.signal_bar_timestamp.date(), e.outcome) for e in breakouts] == [(TUE, "win"), (WED, "filtered")]
    assert breakouts[0].forward_close == 103.0


def test_daily_crosses_can_never_be_confirmed():
    # The day before a daily golden cross has EMA9 <= EMA21, so it is never an
    # uptrend: in the daily timeframe confirmation filters every cross.
    days = [MON - timedelta(days=7) + timedelta(days=i) for i in range(12)]
    days = [d for d in days if d.weekday() < 5]
    bars = [_bar(_at(d), **(UP if i >= 4 else FLAT)) for i, d in enumerate(days)]
    daily = [DailyBar(b.timestamp.date(), b.close, b.ema_9, b.ema_21, None) for b in bars]

    events = BacktestRunner(engine=None)._replay_one("AAPL", bars, daily, 2, 0.01, "daily")

    assert [e.outcome for e in events if e.rule_name == "golden_cross"] == ["filtered"]


# ---------- live engine ----------


async def _daily(engine, ticker: str, day: date, **ind: float) -> None:
    async with engine.begin() as conn:
        await conn.execute(insert(PriceBar), [{
            "ticker": ticker, "timestamp": _at(day), "open": 100.0, "high": 100.0, "low": 100.0,
            "close": 100.0, "volume": 1000.0, "adjusted_close": 100.0,
        }])
        await conn.execute(insert(Indicator), [{"ticker": ticker, "timestamp": _at(day), "rsi_14": 55.0, **ind}])


async def _intraday_golden_cross(engine, ticker: str, day: date) -> None:
    """30 five-minute bars whose last one crosses EMA9 above EMA21."""
    first = _at(day, 14)
    bars, indicators = [], []
    for i in range(30):
        ts = first + timedelta(minutes=5 * i)
        bars.append({"ticker": ticker, "timestamp": ts, "open": 100.0, "high": 100.0, "low": 100.0,
                     "close": 100.0, "volume": 1000.0, "adjusted_close": 100.0})
        indicators.append({"ticker": ticker, "timestamp": ts, "ema_9": 3.0 if i == 29 else 1.0, "ema_21": 2.0})
    async with engine.begin() as conn:
        await conn.execute(insert(PriceBar), bars)
        await conn.execute(insert(Indicator), indicators)


async def test_get_daily_trend_uses_the_last_completed_day(engine):
    await _daily(engine, "AAPL", MON, **UP)
    await _daily(engine, "AAPL", TUE, **FLAT)  # today's bar: ignored
    async with engine.connect() as conn:
        daily = await get_daily_trend(conn, "AAPL", _at(TUE, 15))
    assert daily.trend == "up"
    assert daily.bars_since_update == 1
    assert (daily.ema_9, daily.ema_21, daily.rsi_14, daily.last_close) == (3.0, 2.0, 55.0, 100.0)


async def test_get_daily_trend_is_unknown_without_recent_daily_data(engine):
    await _daily(engine, "OLD", MON - timedelta(days=7), **UP)  # previous Monday: 6 trading days before Tuesday
    async with engine.connect() as conn:
        stale = await get_daily_trend(conn, "OLD", _at(TUE, 15))
        missing = await get_daily_trend(conn, "NONE", _at(TUE, 15))
    assert stale.trend == "unknown" and stale.bars_since_update == 6
    assert missing.trend == "unknown" and missing.reason == "no daily bar"


async def test_confirmed_signal_is_recorded_with_its_daily_context(engine):
    await _daily(engine, "AAPL", MON, **UP)
    await _intraday_golden_cross(engine, "AAPL", TUE)
    stats = FilterStats()

    inserted = await SignalEngine(engine, stats).run_all(["AAPL"])

    assert [s["signal_type"] for s in inserted] == ["golden_cross"]
    async with engine.connect() as conn:
        reasoning = (await conn.execute(select(Signal.reasoning))).scalar_one()
    assert reasoning.endswith("Confirmed by daily uptrend (EMA9 > EMA21, close above EMA21).")
    assert stats.last_24h() == {"golden_cross": {"fired": 1, "filtered": 0, "filter_rate": 0.0}}


async def test_signal_against_the_daily_trend_is_filtered(engine):
    await _daily(engine, "AAPL", MON, **FLAT)
    await _intraday_golden_cross(engine, "AAPL", TUE)
    stats = FilterStats()

    assert await SignalEngine(engine, stats).run_all(["AAPL"]) == []
    assert await SignalEngine(engine, stats).evaluate("AAPL") == []
    assert stats.last_24h() == {"golden_cross": {"fired": 0, "filtered": 1, "filter_rate": 1.0}}


async def test_signal_without_daily_data_fires_unconfirmed(engine):
    await _intraday_golden_cross(engine, "AAPL", TUE)

    inserted = await SignalEngine(engine, FilterStats()).run_all(["AAPL"])

    assert inserted[0]["reasoning"].endswith("Unconfirmed: no daily bar.")


def test_filter_stats_keep_only_the_last_24_hours():
    now = [datetime(2026, 10, 7, 12, tzinfo=UTC)]
    stats = FilterStats(clock=lambda: now[0])
    stats.record("breakout", fired=False)
    now[0] += timedelta(hours=23)
    stats.record("breakout", fired=True)
    stats.record("breakout", fired=False)
    assert stats.last_24h()["breakout"] == {"fired": 1, "filtered": 2, "filter_rate": 0.67}
    now[0] += timedelta(hours=2)
    assert stats.last_24h()["breakout"] == {"fired": 1, "filtered": 1, "filter_rate": 0.5}


# ---------- indicators per timeframe ----------


async def test_backfill_computes_daily_indicators_from_daily_bars_only(engine):
    days = [date(2026, 8, 3) + timedelta(days=i) for i in range(60)]
    days = [d for d in days if d.weekday() < 5]
    rows = [{"ticker": "AAPL", "timestamp": _at(d), "open": 100.0, "high": 100.0, "low": 100.0,
             "close": 100.0, "volume": 1000.0, "adjusted_close": 100.0} for d in days]
    # 5-minute bars at a wildly different price, interleaved with the daily ones.
    rows += [{"ticker": "AAPL", "timestamp": _at(d, 15, 5 * k), "open": 500.0, "high": 500.0, "low": 500.0,
              "close": 500.0, "volume": 10.0, "adjusted_close": 500.0} for d in days[-10:] for k in range(6)]
    async with engine.begin() as conn:
        await conn.execute(insert(PriceBar), rows)

    await BackfillPipeline(engine)._recompute_indicators("AAPL")

    async with engine.connect() as conn:
        ema = (await conn.execute(
            select(Indicator.ema_9).where(Indicator.ticker == "AAPL", Indicator.timestamp == _at(days[-1]))
        )).scalar_one()
    assert ema == pytest.approx(100.0)
