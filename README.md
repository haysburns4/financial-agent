# financial-agent <a href="https://github.com/haysburns4/financial-agent/actions/workflows/ci.yml"><img align="right" src="https://github.com/haysburns4/financial-agent/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>

Automated market data collection, technical signal generation, and portfolio monitoring backed by E-Trade, with a chat agent and web dashboard.

## Stack

- **Package manager:** uv
- **Database:** SQLite (`data/agent.db`)
- **Schema:** SQLAlchemy 2.0 declarative — Core query pattern (no ORM session)
- **Migrations:** Alembic (autogenerate from `src/models.py`)
- **HTTP server:** FastAPI + uvicorn
- **Data source:** E-Trade API via `pyetrade` (OAuth 1.0a)
- **Scheduler:** APScheduler 3.x
- **Indicators:** pandas-ta
- **LLM:** provider-neutral layer (`src/llm/`); Anthropic or OpenAI
- **Web UI:** Next.js 16 + CopilotKit v2 (`web/`), talking to the API over AG-UI

## Getting started

You need Node.js 20+ and E-Trade API keys from [developer.etrade.com](https://developer.etrade.com)
(sandbox keys work for trying it out). Then:

```
git clone <this repo> && cd financial-agent
./start
```

`./start` installs [uv](https://docs.astral.sh/uv/) if it is missing. It
asks for any settings that are missing and installs dependencies. It then starts the API
and the web UI, walks you through the E-Trade login and opens the dashboard at
http://127.0.0.1:3000. Ctrl-C stops everything. Later runs skip whatever is already done.

```
./start --dev          # web UI with hot reload (`next dev`) instead of a production build
./start --no-browser   # don't open the dashboard
./start login          # log in to E-Trade again, against the running app
./start setup --all    # change settings (writes .env)
./start doctor         # check keys, Node, ports, and shell variables that override .env
```

E-Trade ends every session at midnight ET, so log in again once a day: press **Log in to
E-Trade** in the dashboard (it appears within a minute of the session ending), or run
`./start login`. Positions refresh every 15 minutes, or immediately via **Refresh** in the
dashboard. Both servers listen on 127.0.0.1 only.

Settings live in `.env`; `.env.example` lists every one with its default and a one-line
description. Variables exported in your shell override `.env`.

### Running without the launcher

1. `uv sync --extra dev` (add `--extra openai` for `LLM_PROVIDER=openai`).
2. `cp .env.example .env && chmod 600 .env`, then fill in at least the E-Trade keys and
   your provider's API key. Generate `TOKEN_ENCRYPTION_KEY` with
   `uv run python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`.
3. `cd web && cp .env.local.example .env.local && npm install`
4. Run `uv run python -m src.main` (API, http://127.0.0.1:8000) and, in a second terminal,
   `cd web && npm run dev` (http://127.0.0.1:3000).
5. Log in to E-Trade: `curl -X POST 127.0.0.1:8000/auth/start`, open the `auth_url`,
   accept, then send the code (it expires in about 5 minutes):
   ```
   curl -X POST 127.0.0.1:8000/auth/complete -H 'Content-Type: application/json' -d '{"verifier":"XXXX"}'
   ```

### Tests and lint

```
uv run pytest
uv run python tools/anti_slop.py src tests tools
cd web && npm run lint:slop
```
