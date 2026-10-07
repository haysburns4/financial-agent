"""Shared fixtures.

The point of the provider-neutral layer is that the agent can be tested with
no vendor SDK, no API key, and no network — `FakeBackend` is the whole seam.
"""
import os

# src.config builds Settings() at import time, and Settings requires the E-Trade
# keys, so without a .env (a fresh clone, CI) the suite could not even be
# collected. Dummy values let it import; nothing in the tests calls E-Trade or an
# LLM. setdefault, so real values from the shell still win. This must run
# before anything imports src.
os.environ.setdefault("ETRADE_CONSUMER_KEY", "test-consumer-key")
os.environ.setdefault("ETRADE_CONSUMER_SECRET", "test-consumer-secret")
os.environ.setdefault("LLM_PROVIDER", "anthropic")
os.environ.setdefault("ANTHROPIC_API_KEY", "test")

from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

import json

from src.llm import (
    Delta,
    Message,
    MessageComplete,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    Usage,
)
from src.models import Base


class FakeBackend:
    """An `LLMBackend` that replays scripted deltas, one script per turn.

    Pass a flat list of deltas for a single-turn reply, or a list of lists to
    script a tool loop — turn 1 asks for tools, turn 2 answers.
    """

    provider = "fake"

    def __init__(self, deltas: Sequence[Delta] | None = None, model: str = "fake-1") -> None:
        script = list(deltas or [])
        self._turns = script if script and isinstance(script[0], list) else [script]
        self._model = model
        self.calls: list[dict[str, Any]] = []

    @property
    def model(self) -> str:
        return self._model

    async def stream(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[Any] = (),
        max_tokens: int = 4096,
        **extra: Any,  # anti-slop: allow no-any-parameters - mirrors the LLMBackend Protocol
    ) -> AsyncIterator[Delta]:
        self.calls.append(
            {
                "system": system,
                "messages": list(messages),
                "tools": list(tools),
                "max_tokens": max_tokens,
                "extra": extra,
            }
        )
        turn = len(self.calls) - 1
        assert turn < len(self._turns), f"FakeBackend has no script for turn {turn + 1}"
        for delta in self._turns[turn]:
            yield delta


def text_reply(text: str) -> list[Delta]:
    """A scripted stream: one text delta plus the terminal MessageComplete."""
    return [
        TextDelta(text),
        MessageComplete(
            message=Message(role="assistant", text=text),
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=5),
        ),
    ]


def tool_reply(name: str, arguments: dict, call_id: str = "c1") -> list[Delta]:
    """A scripted stream in which the model asks to run one tool."""
    call = ToolCall(id=call_id, name=name, arguments=arguments)
    return [
        ToolCallDelta(index=0, id=call_id, name=name),
        ToolCallDelta(index=0, arguments_json=json.dumps(arguments)),
        MessageComplete(
            message=Message(role="assistant", tool_calls=(call,)),
            stop_reason="tool_use",
            usage=Usage(input_tokens=10, output_tokens=5),
        ),
    ]


@pytest_asyncio.fixture
async def engine():
    """Empty in-memory DB with the real schema, shared across connections."""
    eng = create_async_engine(
        "sqlite+aiosqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
def backend():
    return FakeBackend(text_reply("AAPL is your largest position."))
