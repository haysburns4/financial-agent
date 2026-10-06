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
4. Create `.env` in the repo root:
   ```
   ETRADE_CONSUMER_KEY=...
   ETRADE_CONSUMER_SECRET=...
   ETRADE_SANDBOX=true             # false for your real account (needs production keys)
   TOKEN_ENCRYPTION_KEY=...        # keeps the E-Trade login across restarts; see below
   WATCHLIST=AAPL,MSFT,GOOGL

   LLM_PROVIDER=anthropic          # anthropic | openai
   LLM_CHAT_MODEL=claude-opus-5
   LLM_SYNTHESIS_MODEL=claude-sonnet-5
   ANTHROPIC_API_KEY=...           # or OPENAI_API_KEY for LLM_PROVIDER=openai
   ```
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
uv run python -m src.main          # http://localhost:8000
```

Start the web UI in a second terminal:
```
cd web && npm run dev              # http://localhost:3000
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
