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
- **LLM:** provider-neutral layer (`src/llm/`); Anthropic by default, OpenAI optional

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
   uv sync --extra dev
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

## LLM provider

`src/agent/` depends on the `LLMBackend` Protocol in `src/llm/base.py`, so switching providers is configuration rather than code:

```
LLM_PROVIDER=anthropic          # anthropic | openai
LLM_CHAT_MODEL=claude-opus-5
LLM_SYNTHESIS_MODEL=claude-sonnet-5
ANTHROPIC_API_KEY=...
```

For OpenAI, `uv sync --extra openai` and set `LLM_PROVIDER=openai`, the two model
names, and `OPENAI_API_KEY`. Only the selected provider's key is required —
`build_backend()` raises `LLMConfigError` at startup if the provider is unknown,
uninstalled, or missing its key.

The layer is streaming-first: `stream()` yields `TextDelta` / `ToolCallDelta` and
ends with a `MessageComplete` carrying the finished message, parsed tool calls and
token usage; `collect()` drains it for callers that only want the final answer.
Tool calling is in the neutral types (`ToolDef`, `ToolCall`, `ToolResult`) but no
caller passes tools yet.

To add a provider, implement `stream()` in `src/llm/<name>_backend.py`, re-raising
vendor errors as `LLMError` subclasses, and register it in `src/llm/factory.py`
with a lazy import. Keep wire translation in module-level `_to_*` helpers so it
stays testable without a client — see `tests/test_translation.py`.

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

## Backfill & Backtesting

### First-time setup

After starting the app, backfill historical data before relying on any signals — the live price pipeline only collects bars while the server is running, which is not enough history for indicators or backtests.

```
curl -s -X POST http://localhost:8000/pipeline/backfill/run | jq
```

By default this backfills the union of `WATCHLIST` and your currently-held positions (the same "monitored tickers" set the scheduler operates on) with two years of daily bars plus 60 days of 5-minute intraday bars from yfinance. Override per call:

```
curl -s -X POST http://localhost:8000/pipeline/backfill/run \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL", "MSFT"], "period_daily": "5y"}' | jq
```

Verify coverage for a ticker:

```
curl -s http://localhost:8000/prices/AAPL/coverage | jq
```

Backfill is a slow operation (30–120 seconds for a full watchlist) and is **not** scheduled. Run it on first setup, then periodically to refresh history.

### Running a backtest

Replay signal rules bar-by-bar over historical data with strict no-lookahead semantics. Each rule firing is graded against the close N bars forward:

```
curl -s -X POST http://localhost:8000/backtest/run \
  -H "Content-Type: application/json" \
  -d '{"start_date": "2023-01-01", "end_date": "2025-01-01"}' | jq
```

Optional parameters:

- `tickers`: defaults to `WATCHLIST ∪ {held positions}`.
- `forward_window_days`: how many bars ahead to grade the outcome (default 5).
- `outcome_threshold_pct`: the magnitude of move that counts as a "win" (default 0.01 = 1%).

The report is also saved to `data/backtest_latest.json`. Fetch it later:

```
curl -s http://localhost:8000/backtest/latest | jq
```

Review the per-rule stats and pay attention to:

- `hit_rate` — above ~0.55 is generally worth keeping.
- `sample_size_warning` — rules with fewer than 30 evaluable events are inconclusive.
- `suggested_confidence` — the calibrated weight that would be applied.

### Applying calibrated weights

If the report looks reasonable, push the suggested confidences into the live engine:

```
curl -s -X POST http://localhost:8000/backtest/apply | jq
```

This rewrites `src/backtest/confidence_config.py` and updates the in-process dict so the next signal evaluation uses the new weights immediately. Risk/position rules (`stop_loss_warning`, `concentration_risk`, `drawdown_alert`) keep their static weights because they can't be measured in a per-ticker price backtest. Rules that never fired in the backtest window are left unchanged — silence is not evidence.

Re-run the backtest monthly, or after a significant market regime change, to keep weights calibrated.
