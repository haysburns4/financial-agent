# financial-agent

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

## Setup

1. Install uv:
   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Clone the repo and `cd` into it.
3. Install dependencies (add `--extra openai` if using OpenAI):
   ```
   uv sync --extra dev
   ```
4. Create `.env` in the repo root. The setup wizard will write it for you; to do it by
   hand instead, copy the template and fill in the blanks:
   ```
   cp .env.example .env
   chmod 600 .env
   ```
   `.env.example` lists every setting with its default and a one-line description. You
   need at least `ETRADE_CONSUMER_KEY`, `ETRADE_CONSUMER_SECRET` and the API key for your
   `LLM_PROVIDER` (`ANTHROPIC_API_KEY` by default).
   Generate `TOKEN_ENCRYPTION_KEY` with:
   ```
   uv run python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
   ```
   Variables exported in your shell override `.env`.
5. Install the web UI:
   ```
   cd web
   cp .env.local.example .env.local
   npm install
   ```

## Running

Start the API (creates the SQLite DB on first run):
```
uv run python -m src.main          # http://127.0.0.1:8000
```

Start the web UI in a second terminal:
```
cd web && npm run dev              # http://127.0.0.1:3000
```

### E-Trade login

1. `curl -X POST localhost:8000/auth/start` and open the returned `auth_url`.
2. Log in, accept, and copy the verification code (it expires in ~5 minutes).
3. Send it:
   ```
   curl -X POST localhost:8000/auth/complete -H 'Content-Type: application/json' -d '{"verifier":"XXXX"}'
   ```
4. Check with `curl localhost:8000/auth/status`.

E-Trade ends every session at midnight ET, so repeat this daily. Positions refresh
every 15 minutes, or immediately via **Refresh** in the dashboard.

### Tests and lint

```
uv run pytest
uv run python tools/anti_slop.py src tests tools
cd web && npm run lint:slop
```
