# financial-agent

Automated market data collection, technical signal generation, and portfolio monitoring backed by E-Trade.

## Stack

- **Package manager:** uv
- **Database:** SQLite (`data/agent.db`)
- **Schema:** SQLAlchemy 2.0 declarative — Core query pattern (no ORM session)
- **Migrations:** Alembic (autogenerate from `src/models.py`)
- **HTTP server:** FastAPI + uvicorn
- **Data source:** E-Trade API via `pyetrade` (OAuth 1.0a)
- **Scheduler:** APScheduler 3.x
- **Indicators:** pandas-ta

## Setup

1. Install uv:
   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Clone repo and `cd` into it.
3. Install runtime deps (creates `.venv`, writes `uv.lock`):
   ```
   uv sync
   ```
4. Install with test deps:
   ```
   uv sync --group dev
   ```
5. Configure secrets:
   ```
   cp .env.example .env
   ```
   Fill in E-Trade consumer key/secret.
6. Run the agent (auto-creates the SQLite DB on first run):
   ```
   uv run python -m src.main
   ```

## Schema changes

`src/models.py` is the single source of truth for the database schema. To change it:

1. Edit `src/models.py`.
2. Generate a migration:
   ```
   uv run alembic revision --autogenerate -m "describe change"
   ```
3. Apply it:
   ```
   uv run alembic upgrade head
   ```

## Architecture notes

- All table definitions live in `src/models.py`. The application code uses **SQLAlchemy Core** (`select()`, `insert()`, `insert().on_conflict_do_update()`) against raw `AsyncConnection`s — no sessions, no identity map, no lazy loading. Query results are `Row` objects, not model instances.
- `uv.lock` is committed to the repo for reproducible builds.
