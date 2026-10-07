"""Post-backfill catch-up: record the signals missed while the app was off."""
from datetime import date, datetime, time, timedelta, timezone

from sqlalchemy import insert, select

from src.models import Indicator, PipelineRun, PriceBar, Signal
from src.signals.catchup import CATCHUP_PIPELINE, catch_up
from src.signals.engine import FilterStats, SignalEngine

UTC = timezone.utc
MON, TUE, WED, THU, FRI = (date(2026, 10, d) for d in range(5, 10))
NOW = datetime(2026, 10, 9, 21, tzinfo=UTC)  # Friday, after the close


def _at(day: date, hour: int = 0, minute: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=UTC)


def _bar(ticker: str, ts: datetime, close: float = 100.0, volume: float = 1000.0) -> dict:
    return {"ticker": ticker, "timestamp": ts, "open": 100.0, "high": 100.0, "low": 97.0,
            "close": close, "volume": volume, "adjusted_close": close}


async def _seed(engine, ticker: str, breakouts: list[datetime], daily_ema_9: float = 3.0) -> None:
    """5-minute bars Mon–Fri 14:00–16:00 UTC, flat at 100 except the given
    breakouts (close 101 on double volume); daily bars for confirmation."""
    rows, indicators = [], []
    for day in (MON, TUE, WED, THU, FRI):
        for k in range(24):
            ts = _at(day, 14) + timedelta(minutes=5 * k)
            hit = ts in breakouts
            rows.append(_bar(ticker, ts, 101.0 if hit else 100.0, 2000.0 if hit else 1000.0))
            indicators.append({"ticker": ticker, "timestamp": ts, "ema_9": 3.0, "ema_21": 2.0})
        rows.append(_bar(ticker, _at(day)))  # the day's daily bar
        indicators.append({"ticker": ticker, "timestamp": _at(day), "ema_9": daily_ema_9, "ema_21": 2.0})
    async with engine.begin() as conn:
        await conn.execute(insert(PriceBar), rows)
        await conn.execute(insert(Indicator), indicators)


async def _last_check(engine, at: datetime) -> None:
    async with engine.begin() as conn:
        await conn.execute(insert(PipelineRun).values(
            pipeline="price_pipeline", started_at=at, completed_at=at, status="ok", tickers_processed=1,
        ))


async def _signals(engine) -> list:
    async with engine.connect() as conn:
        return (await conn.execute(select(Signal).order_by(Signal.timestamp))).all()


def _engine(engine) -> SignalEngine:
    return SignalEngine(engine, FilterStats())


async def test_records_what_fired_since_the_last_check_at_its_real_time(engine):
    before, after = _at(TUE, 15), _at(THU, 15)
    await _seed(engine, "AAPL", [before, after])
    await _last_check(engine, _at(WED, 12))

    result = await catch_up(engine, ["aapl"], now=NOW, signal_engine=_engine(engine))

    signals = await _signals(engine)
    assert [(s.ticker, s.signal_type) for s in signals] == [("AAPL", "breakout")]
    assert signals[0].timestamp.replace(tzinfo=UTC) == after  # the bar's time, not now
    assert signals[0].delivered  # never pushed to Discord after the fact
    assert "Caught up after a backfill." in signals[0].reasoning
    assert signals[0].reasoning.endswith("Confirmed by daily uptrend (EMA9 > EMA21, close above EMA21).")
    assert result.recorded == {"breakout": 1}
    assert result.since == _at(WED, 12)


async def test_never_reaches_back_more_than_five_trading_days(engine):
    await _seed(engine, "AAPL", [_at(MON, 15), _at(FRI, 15)])
    await _last_check(engine, _at(MON, 12) - timedelta(days=21))  # weeks ago

    result = await catch_up(engine, ["AAPL"], now=_at(date(2026, 10, 12), 21), signal_engine=_engine(engine))

    # Monday Oct 12 21:00 minus 5 trading days is Monday Oct 5 21:00: Monday's
    # 15:00 breakout is older than that, Friday's is not.
    assert result.since == _at(MON, 21)
    assert [s.timestamp.replace(tzinfo=UTC) for s in await _signals(engine)] == [_at(FRI, 15)]


async def test_applies_daily_trend_confirmation(engine):
    await _seed(engine, "AAPL", [_at(THU, 15)], daily_ema_9=1.0)  # daily trend flat
    await _last_check(engine, _at(WED, 12))

    result = await catch_up(engine, ["AAPL"], now=NOW, signal_engine=_engine(engine))

    assert await _signals(engine) == []
    assert result.filtered == 1


async def test_deduplicates_like_the_live_engine(engine):
    # Two breakouts 30 minutes apart, and one already recorded live on Friday.
    await _seed(engine, "AAPL", [_at(THU, 14, 50), _at(THU, 15, 20), _at(FRI, 15)])
    await _last_check(engine, _at(WED, 12))
    async with engine.begin() as conn:
        await conn.execute(insert(Signal), [{
            "ticker": "AAPL", "timestamp": _at(FRI, 15), "signal_type": "breakout", "direction": "long",
            "confidence": 0.4, "reasoning": "live", "delivered": False,
        }])

    result = await catch_up(engine, ["AAPL"], now=NOW, signal_engine=_engine(engine))

    assert result.recorded == {"breakout": 1}
    assert result.duplicates == 2
    assert len(await _signals(engine)) == 2


async def test_a_second_catch_up_starts_where_the_first_stopped(engine):
    await _seed(engine, "AAPL", [_at(THU, 15)])
    await _last_check(engine, _at(WED, 12))
    await catch_up(engine, ["AAPL"], now=NOW, signal_engine=_engine(engine))

    again = await catch_up(engine, ["AAPL"], now=NOW, signal_engine=_engine(engine))

    assert again.since == NOW
    assert again.recorded == {}
    async with engine.connect() as conn:
        runs = (await conn.execute(select(PipelineRun).where(PipelineRun.pipeline == CATCHUP_PIPELINE))).all()
    assert len(runs) == 2
