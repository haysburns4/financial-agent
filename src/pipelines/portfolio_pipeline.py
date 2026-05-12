from datetime import datetime, timezone

from loguru import logger
from sqlalchemy import insert, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from src.db import get_connection
from src.etrade.accounts import fetch_positions
from src.models import PipelineRun, Position
from src.pipelines.validators import PositionIn


async def run_portfolio_pipeline() -> None:
    started = datetime.now(timezone.utc)
    processed = 0
    errors: list[str] = []

    async with get_connection() as conn:
        run_id = (
            await conn.execute(
                insert(PipelineRun).values(
                    pipeline="portfolio_pipeline",
                    started_at=started,
                    status="running",
                )
            )
        ).inserted_primary_key[0]

        try:
            raw_positions = await fetch_positions()
            positions = [PositionIn(**p).model_dump() for p in raw_positions]
            if positions:
                stmt = sqlite_insert(Position).values(positions)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["account_id", "ticker"],
                    set_={
                        "quantity": stmt.excluded.quantity,
                        "cost_basis": stmt.excluded.cost_basis,
                        "market_value": stmt.excluded.market_value,
                        "last_updated": stmt.excluded.last_updated,
                    },
                )
                await conn.execute(stmt)
                processed = len(positions)
        except Exception as exc:
            logger.exception("portfolio_pipeline failed")
            errors.append(str(exc))

        await conn.execute(
            update(PipelineRun)
            .where(PipelineRun.id == run_id)
            .values(
                completed_at=datetime.now(timezone.utc),
                status="ok" if not errors else "failed",
                tickers_processed=processed,
                errors="\n".join(errors) if errors else None,
            )
        )

    logger.info("portfolio_pipeline: {} positions processed, {} errors", processed, len(errors))
