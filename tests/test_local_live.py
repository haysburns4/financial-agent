"""One real tool-calling chat turn against the local server LOCAL_LLM_*
describe — the check that a newly configured server and model work together.
Skipped unless asked for:

    RUN_LIVE_LOCAL=1 uv run pytest tests/test_local_live.py
"""
import os
from datetime import datetime, timezone

import pytest
from sqlalchemy import insert

from src.agent.chat import AgentChat
from src.config import settings
from src.llm import build_backend
from src.models import Position

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_LOCAL") != "1", reason="set RUN_LIVE_LOCAL=1 to talk to a real server"
)


async def test_a_chat_turn_looks_up_holdings_and_answers(engine):
    async with engine.begin() as conn:
        await conn.execute(insert(Position), [{
            "account_id": "A1", "ticker": "AAPL", "quantity": 10.0, "cost_basis": 1000.0,
            "market_value": 1500.0, "last_updated": datetime.now(timezone.utc),
        }])
    backend = build_backend(settings.LOCAL_LLM_MODEL, "local")

    result = await AgentChat(engine, backend).ask("Which tickers do I hold? Look it up.")

    tools = [c["name"] for c in result["context_used"]["tools_called"]]
    assert "get_positions" in tools, f"the model called {tools}, not get_positions"
    assert "AAPL" in result["answer"]
    assert "<think>" not in result["answer"] and "</think>" not in result["answer"]
