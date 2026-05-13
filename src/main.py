import sys
from contextlib import asynccontextmanager

import uvicorn
from loguru import logger

from src.config import settings
from src.db import engine, init_db
from src.etrade.auth import auth
from src.scheduler import build_scheduler
from src.server import create_app


def _configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level=settings.LOG_LEVEL)


@asynccontextmanager
async def lifespan(app):
    _configure_logging()
    await init_db()

    if await auth.load_persisted():
        logger.info("E-Trade tokens loaded from disk; auth restored")
    else:
        logger.info("No persisted E-Trade tokens; OAuth flow required")

    scheduler = build_scheduler()
    scheduler.start()
    logger.info("Scheduler started with {} jobs", len(scheduler.get_jobs()))
    app.state.scheduler = scheduler

    try:
        yield
    finally:
        scheduler.shutdown(wait=False)
        await engine.dispose()
        logger.info("Shutdown complete")


app = create_app()
app.router.lifespan_context = lifespan


def main() -> None:
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
