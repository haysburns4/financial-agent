"""Local implementation of `LLMBackend`: any OpenAI-compatible server.

mlx_lm.server, Ollama, llama.cpp's llama-server, LM Studio and vLLM all speak
the Chat Completions API, so this is the OpenAI backend with a different
client. It diverges only where local servers and open models do, and what
differs between them is data in `local_profiles`, not code here:

- no authentication: the client sends a placeholder key;
- first-token latency of tens of seconds: a long timeout
  (LOCAL_LLM_TIMEOUT_SECONDS) and no retries, so a dead server fails fast;
- the output cap goes in whichever field the server takes;
- thinking models are switched off by default to save costs;
- the context window is the server's, clamped to what it reports;
- tokens are estimated for the model's family, whose tokenizer may split
  every digit.

Model names pass through verbatim (a Hugging Face repo path, an Ollama tag).
Missing usage is reported as zero tokens.

Needs the `openai` package, like the OpenAI backend — `uv sync --extra openai`.
"""
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from typing import Any

import openai
from loguru import logger

from src.llm.base import (
    Delta,
    LLMConnectionError,
    Message,
    MessageComplete,
    TextDelta,
    ToolDef,
    estimate_tokens,
)
from src.llm.health import LocalServerMonitor
from src.llm.local_profiles import (
    DEFAULT_SERVER,
    SERVERS,
    ModelProfile,
    ServerProfile,
    model_profile,
    start_hint,
    thinking_off,
)
from src.llm.openai_backend import OpenAIBackend

# Local servers ignore credentials, but the SDK requires a non-empty key.
PLACEHOLDER_API_KEY = "not-needed"
# Prompt tokens when the factory does not say; a 14B model's window is ~40k.
DEFAULT_CONTEXT_TOKENS = 24_000


def make_client(base_url: str, timeout_seconds: float) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        base_url=base_url,
        api_key=PLACEHOLDER_API_KEY,
        timeout=timeout_seconds,
        # Retrying a local server that isn't running only delays the error.
        max_retries=0,
    )


class ThinkFilter:
    """Removes thinking blocks from answer text that arrives in pieces."""

    def __init__(self, tags: tuple[str, str], opens_implicitly: bool = False) -> None:
        self._open, self._close = tags
        self._thinking = opens_implicitly
        self._implicit = opens_implicitly
        self._pending = ""

    def feed(self, text: str) -> str:
        """The answer text `text` releases."""
        self._pending += text
        out: list[str] = []
        while True:
            tag = self._close if self._thinking else self._open
            at = self._pending.find(tag)
            if at >= 0:
                if not self._thinking:
                    out.append(self._pending[:at])
                self._pending = self._pending[at + len(tag):]
                self._thinking = not self._thinking
                self._implicit = False
                continue
            if self._implicit:
                return "".join(out)  # hold everything until the block closes
            keep = _partial_suffix(self._pending, tag)
            if not self._thinking:
                out.append(self._pending[: len(self._pending) - keep])
            self._pending = self._pending[len(self._pending) - keep:]
            return "".join(out)

    def flush(self) -> str:
        """What is still held at the end of the reply. An implicit block
        that never closed was the answer; an explicit one left open was not."""
        rest, self._pending = self._pending, ""
        if self._implicit or not self._thinking:
            return rest
        return ""


def _partial_suffix(text: str, tag: str) -> int:
    """Length of the longest end of `text` that begins `tag`."""
    for size in range(min(len(tag) - 1, len(text)), 0, -1):
        if tag.startswith(text[-size:]):
            return size
    return 0


class LocalBackend(OpenAIBackend):
    provider = "local"

    def __init__(
        self,
        client: openai.AsyncOpenAI,
        model: str,
        base_url: str,
        timeout_seconds: float,
        *,
        server: ServerProfile = SERVERS[DEFAULT_SERVER],
        family: ModelProfile | None = None,
        context_tokens: int = DEFAULT_CONTEXT_TOKENS,
        monitor: LocalServerMonitor | None = None,
    ) -> None:
        """`family` defaults to the one matched from `model`. `monitor`
        reports the server's context window; None trusts `context_tokens`."""
        super().__init__(client, model, context_tokens=context_tokens)
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds
        self.server = server
        self.family = family or model_profile(model)
        self.max_tokens_param = server.max_tokens_param
        self._thinking_off = thinking_off(server, self.family)
        self._monitor = monitor
        self._clamped_to: int | None = None

    def estimate_tokens(self, text: str) -> int:
        return estimate_tokens(text, self.family.digits_per_token)

    async def prompt_budget(self, max_tokens: int) -> int:
        if self._monitor is None or self.server.window_source is None:
            return self._context_tokens
        window = (await self._monitor.probe()).window_for(self.model)
        if window is None or window - max_tokens >= self._context_tokens:
            return self._context_tokens
        fitted = max(0, window - max_tokens)
        if fitted != self._clamped_to:
            self._clamped_to = fitted
            fix = f" To give it more, {self.server.window_hint}." if self.server.window_hint else ""
            logger.warning(
                "local LLM: {} has a {}-token window; with up to {} reply tokens the prompt "
                "budget is cut from {} to {}.{}",
                self.model, window, max_tokens, self._context_tokens, fitted, fix,
            )
        return fitted

    async def stream(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolDef] = (),
        max_tokens: int = 4096,
        **extra: Any,  # anti-slop: allow no-any-parameters - provider passthrough is the documented Protocol contract
    ) -> AsyncIterator[Delta]:
        off = self._thinking_off
        extra_body = dict(extra.pop("extra_body", None) or {})
        if off.template_kwargs:
            # The caller's own kwargs win, so thinking can be turned back on.
            extra_body["chat_template_kwargs"] = {**off.template_kwargs, **extra_body.get("chat_template_kwargs", {})}
        if extra_body:
            extra["extra_body"] = extra_body
        if off.reasoning_effort:
            extra.setdefault("reasoning_effort", off.reasoning_effort)

        tags = self.family.think_tags
        think = ThinkFilter(tags, self.family.think_opens_implicitly) if tags else None
        released: list[str] = []
        try:
            async for delta in super().stream(
                system=system + off.system_suffix, messages=messages, tools=tools, max_tokens=max_tokens, **extra,
            ):
                if think is None:
                    yield delta
                elif isinstance(delta, TextDelta):
                    if text := think.feed(delta.text):
                        released.append(text)
                        yield TextDelta(text)
                elif isinstance(delta, MessageComplete):
                    if text := think.flush():
                        released.append(text)
                        yield TextDelta(text)
                    # The answer is what the caller was shown, not the raw reply.
                    yield replace(delta, message=replace(delta.message, text="".join(released)))
                else:
                    yield delta
        except LLMConnectionError as exc:
            raise LLMConnectionError(self._explain(exc)) from exc

    def _explain(self, exc: LLMConnectionError) -> str:
        label = self.server.label
        if isinstance(exc.__cause__, openai.APITimeoutError):
            return (
                f"The local LLM server ({label}) at {self._base_url} did not answer within "
                f"{self._timeout_seconds:.0f}s. A large model can be slow to start; "
                "raise LOCAL_LLM_TIMEOUT_SECONDS if it is still loading."
            )
        hint = start_hint(self.server, self._base_url, self.model)
        return (
            f"Can't reach the local LLM server ({label}) at {self._base_url}: it appears to be down. "
            f"{hint[0].upper()}{hint[1:]}."
        )
