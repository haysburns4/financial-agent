from datetime import datetime, timezone

from loguru import logger
from sqlalchemy import insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from src.config import settings
from src.db import get_connection
from src.etrade.market import fetch_ohlcv
from src.models import PipelineRun, PriceBar
from src.pipelines.validators import PriceBarIn


async def run_price_pipeline() -> None:
    started = datetime.now(timezone.utc)
    processed = 0
    errors: list[str] = []

    async with get_connection() as conn:
        run_id = (
            await conn.execute(
                insert(PipelineRun).values(
                    pipeline="price_pipeline",
                    started_at=started,
                    status="running",
                )
            )
        ).inserted_primary_key[0]

        for ticker in settings.WATCHLIST:
            try:
                raw_bars = await fetch_ohlcv(ticker)
                bars = [PriceBarIn(**b).model_dump() for b in raw_bars]
                if not bars:
                    continue

                stmt = sqlite_insert(PriceBar).values(bars)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["ticker", "timestamp"],
                    set_={
                        "open": stmt.excluded.open,
                        "high": stmt.excluded.high,
                        "low": stmt.excluded.low,
                        "close": stmt.excluded.close,
                        "volume": stmt.excluded.volume,
                        "adjusted_close": stmt.excluded.adjusted_close,
                        "data_quality": stmt.excluded.data_quality,
                    },
                )
                await conn.execute(stmt)
                processed += 1
            except Exception as exc:
                logger.exception("price_pipeline failed for {}", ticker)
                errors.append(f"{ticker}: {exc}")

        from sqlalchemy import update

        await conn.execute(
            update(PipelineRun)
            .where(PipelineRun.id == run_id)
            .values(
                completed_at=datetime.now(timezone.utc),
                status="ok" if not errors else "partial",
                tickers_processed=processed,
                errors="\n".join(errors) if errors else None,
            )
        )

    logger.info("price_pipeline: {} tickers processed, {} errors", processed, len(errors))
