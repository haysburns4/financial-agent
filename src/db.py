from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from src import models  # noqa: F401  — ensures all tables are registered on Base.metadata
from src.config import settings
from src.models import Base

engine = create_async_engine(settings.DATABASE_URL, future=True)


@asynccontextmanager
async def get_connection() -> AsyncIterator[AsyncConnection]:
    async with engine.begin() as conn:
        yield conn


async def init_db() -> None:
    """First-run bootstrap. Alembic handles migrations after this."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized at {}", settings.DATABASE_URL)


async def health_check() -> bool:
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("Database health check failed: {}", exc)
        return False
