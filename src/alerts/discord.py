"""Discord webhook alerts."""
import httpx
from loguru import logger
from sqlalchemy import select, update

from src.config import settings
from src.db import get_connection
from src.models import Signal


def _format(row) -> str:
    direction = f" {row.direction}" if row.direction else ""
    return (
        f"**{row.signal_type.upper()}{direction}** — `{row.ticker}` "
        f"(confidence {row.confidence:.0%})\n"
        f"{row.reasoning}"
    )


async def post(content: str) -> bool:
    if not settings.DISCORD_WEBHOOK_URL:
        logger.debug("Discord webhook not configured — skipping alert")
        return False
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                settings.DISCORD_WEBHOOK_URL,
                json={"content": content},
            )
            resp.raise_for_status()
        return True
    except Exception:
        logger.exception("Discord webhook post failed")
        return False


async def deliver_pending_signals() -> int:
    """Post any undelivered signals to Discord and mark them delivered."""
    sent = 0
    async with get_connection() as conn:
        stmt = select(Signal).where(Signal.delivered.is_(False)).order_by(Signal.created_at)
        rows = (await conn.execute(stmt)).all()
        for row in rows:
            if await post(_format(row)):
                await conn.execute(
                    update(Signal).where(Signal.id == row.id).values(delivered=True)
                )
                sent += 1
    return sent
