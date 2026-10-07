"""Recalibrate signal confidences from all stored history.

Runs after every backfill (POST /pipeline/backfill/run, so also
`./start --backfill`): a daily walk-forward over everything stored, applied
straight to the live weights. The apply policy decides what may change —
portfolio rules never, rules with too few confirmed signals keep their default,
ticker overrides need enough evidence of their own — so automatic is safe to
repeat. Signals already recorded keep the confidence they fired with.
"""
from datetime import date, datetime, timezone

from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine

from src.backtest.persistence import STORE, CalibrationStore
from src.backtest.runner import BacktestRunner
from src.models import PriceBar


async def earliest_bar_date(engine: AsyncEngine, tickers: list[str]) -> date | None:
    async with engine.connect() as conn:
        first = await conn.scalar(
            select(func.min(PriceBar.timestamp)).where(PriceBar.ticker.in_([t.upper() for t in tickers]))
        )
    return first.date() if first is not None else None


async def calibrate(
    engine: AsyncEngine,
    tickers: list[str],
    store: CalibrationStore = STORE,
    today: date | None = None,
) -> dict:
    """Walk-forward over all stored history for `tickers`, then apply it.

    Raises ValueError when there is too little history for one window.
    """
    start = await earliest_bar_date(engine, tickers)
    if start is None:
        raise ValueError("no price history stored for these tickers")
    end = today or datetime.now(timezone.utc).date()
    report = await BacktestRunner(engine).run_walkforward(tickers, start, end)
    store.save_walkforward_report(report)
    payload = store.load_walkforward_json()
    assert payload is not None  # just written
    applied = store.apply_report(payload, walkforward=True)
    logger.info(
        "calibration: walk-forward {} → {} over {} window(s); {} default(s) and {} ticker override(s) updated",
        start, end, len(report.windows), len(applied["updated"]),
        sum(len(rules) for rules in applied["overrides"].values()),
    )
    return {
        "windows": len(report.windows),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        **applied,
    }
