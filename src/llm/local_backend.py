"""Local implementation of `LLMBackend`: an MLX server (`mlx_lm.server`).

mlx_lm.server speaks the OpenAI Chat Completions API. Probed against
mlx-lm 0.32, it streams SSE, honours `max_completion_tokens`, reports usage
on the last chunk when asked, and streams tool calls in OpenAI's format — so
this is the OpenAI backend with a different client. It diverges only where
the server does:

- no authentication: the client sends a placeholder key;
- first-token latency of tens of seconds: a long timeout
  (LOCAL_LLM_TIMEOUT_SECONDS) and no retries, so a dead server fails fast;
- the usual failure is the server not running, so connection errors name the
  base URL and say so, instead of surfacing the SDK's generic message.

Model names are Hugging Face repo paths ("mlx-community/Qwen2.5-7B-Instruct-4bit")
and pass through verbatim. Missing usage is reported as zero tokens.

Needs the `openai` package, like the OpenAI backend — `uv sync --extra openai`.
"""
from collections.abc import AsyncIterator, Sequence
from typing import Any

import openai

from src.llm.base import Delta, LLMConnectionError, Message, ToolDef
from src.llm.openai_backend import OpenAIBackend

# mlx_lm.server ignores credentials, but the SDK requires a non-empty key.
PLACEHOLDER_API_KEY = "not-needed"


def make_client(base_url: str, timeout_seconds: float) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        base_url=base_url,
        api_key=PLACEHOLDER_API_KEY,
        timeout=timeout_seconds,
        # Retrying a local server that isn't running only delays the error.
        max_retries=0,
    )


class LocalBackend(OpenAIBackend):
    provider = "local"

    def __init__(
        self, client: openai.AsyncOpenAI, model: str, base_url: str, timeout_seconds: float
    ) -> None:
        super().__init__(client, model)
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds

    async def stream(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolDef] = (),
        max_tokens: int = 4096,
        **extra: Any,  # anti-slop: allow no-any-parameters - provider passthrough is the documented Protocol contract
    ) -> AsyncIterator[Delta]:
        try:
            async for delta in super().stream(
                system=system, messages=messages, tools=tools, max_tokens=max_tokens, **extra,
            ):
                yield delta
        except LLMConnectionError as exc:
            raise LLMConnectionError(self._explain(exc)) from exc

    def _explain(self, exc: LLMConnectionError) -> str:
        if isinstance(exc.__cause__, openai.APITimeoutError):
            return (
                f"The local MLX server at {self._base_url} did not answer within "
                f"{self._timeout_seconds:.0f}s. A large model can be slow to start; "
                "raise LOCAL_LLM_TIMEOUT_SECONDS if it is still loading."
            )
        return (
            f"Can't reach the local MLX server at {self._base_url}: it appears to be down. "
            f"Start it with `uv run mlx_lm.server --model {self.model} --port 8080`, "
            "or point LOCAL_LLM_BASE_URL at where it runs."
        )
