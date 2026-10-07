"""Interactive Q&A over portfolio + signal data.

The endpoint is intentionally stateless: callers (the frontend) own the
conversation history and replay it on each request. This keeps the API simple,
lets users clear history at will, and avoids server-side session storage.
History is capped at the last `MAX_HISTORY_TURNS` user+assistant turns.

The agent reaches its data through the tools in [src/agent/tools.py] rather than
receiving a dump of it in the system prompt, so context stays bounded however
large the watchlist and position set grow. The prompt carries only enough
orientation for the model to know what is worth asking for.
"""
from collections.abc import AsyncIterator

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncEngine

from src.agent.tools import TOOLS, data_summary, run_tool
from src.config import settings
from src.etrade.auth import auth as etrade_auth
from src.llm import (
    Delta,
    LLMBackend,
    LLMError,
    Message,
    MessageComplete,
    TextDelta,
    ToolResult,
)

_SYSTEM_PROMPT_TEMPLATE = (
    "You are a financial analysis assistant with access to the user's equity "
    "portfolio and technical indicators across multiple accounts.\n\n"
    "Use the tools to look up data before answering. Do not guess at numbers or "
    "reuse figures from earlier in the conversation — the underlying data moves.\n\n"
    "What exists right now:\n{facts}\n\n"
    "Never give direct buy/sell advice. If asked for something the tools cannot "
    "reach (real-time quotes, news, fundamentals, predictions), say so clearly and "
    "offer what you can answer instead. Be concise, and reference specific tickers, "
    "positions and signals by name. Refer to an account by the last four digits "
    "of its number (e.g. ••9991), as the dashboard does; never write it in full."
)

_STALE_AUTH_NOTE = (
    "\n\nNote: the user is not currently authenticated with E-Trade, so position "
    "and balance data may be stale. Flag this if relevant."
)


_CUT_OFF_NOTE = (
    "\n\n_(The answer was cut off: the model reached its output limit. "
    "Try a narrower question.)_"
)
_TOOL_CEILING_NOTE = (
    "\n\n_(Stopped after {n} rounds of looking things up without an answer. "
    "Try a narrower question.)_"
)


class AgentChat:
    # Output budget per model turn. Reasoning models (gpt-5, Opus 5's adaptive
    # thinking) spend hidden reasoning tokens from this same budget, so at 2048
    # they could think through it entirely and return no text at all.
    MAX_TOKENS = 16_000
    MAX_HISTORY_TURNS = 10
    # Ceiling on model -> tools -> model round trips within a single question.
    MAX_TOOL_ITERATIONS = 5

    def __init__(self, engine: AsyncEngine, backend: LLMBackend) -> None:
        self._engine = engine
        self._backend = backend

    async def ask(
        self,
        question: str,
        conversation_history: list[dict] | None = None,
    ) -> dict:
        system_prompt, summary = await self._prepare()
        messages = self._build_messages(question, conversation_history)

        answer: list[str] = []
        tools_called: list[dict] = []
        async for item in self._run(system_prompt, messages):
            if isinstance(item, TextDelta):
                answer.append(item.text)
            elif isinstance(item, MessageComplete):
                tools_called.extend(
                    {"name": c.name, "arguments": c.arguments}
                    for c in item.message.tool_calls
                )

        text = "".join(answer).strip()
        logger.info(
            "chat: {} -> {} chars, {} tool call(s) ({}/{})",
            question[:60].replace("\n", " "),
            len(text),
            len(tools_called),
            self._backend.provider,
            self._backend.model,
        )
        return {
            "answer": text,
            "context_used": {
                "tools_called": tools_called,
                "data_freshness_minutes": summary["data_freshness_minutes"],
            },
        }

    async def ask_stream(
        self,
        question: str,
        conversation_history: list[dict] | None = None,
    ) -> AsyncIterator[Delta | ToolResult]:
        """`ask` in streaming form, for the AG-UI/CopilotKit endpoint.

        Yields provider deltas plus a `ToolResult` for each tool the agent runs.
        """
        system_prompt, _ = await self._prepare()
        async for item in self._run(
            system_prompt, self._build_messages(question, conversation_history)
        ):
            yield item

    async def _prepare(self) -> tuple[str, dict]:
        """Build the system prompt from a cheap summary of what data exists."""
        async with self._engine.begin() as conn:
            summary = await data_summary(conn)

        prompt = _SYSTEM_PROMPT_TEMPLATE.format(facts=_format_facts(summary))
        if not await etrade_auth.is_authenticated():
            prompt += _STALE_AUTH_NOTE
        return prompt, summary

    async def _run(
        self,
        system_prompt: str,
        messages: list[Message],
    ) -> AsyncIterator[Delta | ToolResult]:
        """Drive model -> tools -> model until the model stops asking for tools."""
        for _ in range(self.MAX_TOOL_ITERATIONS):
            final: MessageComplete | None = None
            try:
                async for delta in self._backend.stream(
                    system=system_prompt,
                    messages=messages,
                    tools=TOOLS,
                    max_tokens=self.MAX_TOKENS,
                ):
                    if isinstance(delta, MessageComplete):
                        final = delta
                    yield delta
            except LLMError:
                logger.exception("AgentChat: {} stream failed", self._backend.provider)
                raise

            if final is None or final.stop_reason != "tool_use":
                if final is not None and final.stop_reason == "refusal":
                    logger.warning("chat: provider refused the request")
                if final is not None and final.stop_reason == "max_tokens":
                    # Otherwise the user sees the tool cards and then nothing.
                    logger.warning("chat: answer cut off at {} output tokens", self.MAX_TOKENS)
                    yield TextDelta(_CUT_OFF_NOTE)
                return

            messages.append(final.message)
            results = []
            for call in final.message.tool_calls:
                result = await run_tool(self._engine, call)
                results.append(result)
                yield result
            messages.append(Message(role="user", tool_results=tuple(results)))

        logger.warning(
            "chat: stopped at the {}-iteration tool ceiling", self.MAX_TOOL_ITERATIONS
        )
        yield TextDelta(_TOOL_CEILING_NOTE.format(n=self.MAX_TOOL_ITERATIONS))

    def _build_messages(
        self,
        question: str,
        history: list[dict] | None,
    ) -> list[Message]:
        messages: list[Message] = []
        if history:
            for turn in history[-(self.MAX_HISTORY_TURNS * 2):]:
                role = turn.get("role")
                content = turn.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append(Message(role=role, text=content))
        messages.append(Message(role="user", text=question))
        return messages


def _format_facts(summary: dict) -> str:
    freshness = summary["data_freshness_minutes"]
    lines = [
        f"- Watchlist: {', '.join(settings.WATCHLIST)}",
        f"- {summary['positions']} positions across {summary['accounts']} account(s)",
        f"- {summary['signals_recent']} signal(s) in the last 2 trading days",
    ]
    lines.append(
        "- No price bars stored yet; a backfill has not been run"
        if freshness < 0
        else f"- Most recent price bar is {freshness} minute(s) old"
    )
    return "\n".join(lines)
