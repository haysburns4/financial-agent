from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from loguru import logger
from sqlalchemy import event, make_url, text
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine
from sqlalchemy.pool import ConnectionPoolEntry

from src import models  # noqa: F401  — ensures all tables are registered on Base.metadata
from src.config import settings
from src.models import Base

# How long a SQLite write waits for another writer.
SQLITE_BUSY_TIMEOUT_SECONDS = 30.0


def _is_sqlite(url: str) -> bool:
    return make_url(url).get_backend_name() == "sqlite"


def make_engine(url: str) -> AsyncEngine:
    """An engine for `url`; SQLite gets a busy timeout and WAL journaling."""
    if not _is_sqlite(url):
        return create_async_engine(url, future=True)
    created = create_async_engine(url, future=True, connect_args={"timeout": SQLITE_BUSY_TIMEOUT_SECONDS})

    @event.listens_for(created.sync_engine, "connect")
    def _wal(dbapi_connection: DBAPIConnection, _record: ConnectionPoolEntry) -> None:
        # Readers does not wait on the writer, nor it on them.
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    return created


engine = make_engine(settings.DATABASE_URL)


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
