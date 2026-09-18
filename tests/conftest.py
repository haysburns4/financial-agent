"""Shared fixtures.

The point of the provider-neutral layer is that the agent can be tested with
no vendor SDK, no API key, and no network — `FakeBackend` is the whole seam.
"""
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from src.llm import Delta, Message, MessageComplete, TextDelta, Usage
from src.models import Base


class FakeBackend:
    """An `LLMBackend` that replays a scripted list of deltas."""

    provider = "fake"

    def __init__(self, deltas: Sequence[Delta] | None = None, model: str = "fake-1") -> None:
        self._deltas = list(deltas or [])
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
        **extra: Any,
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
        for delta in self._deltas:
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
