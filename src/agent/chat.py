"""Interactive Q&A over portfolio + signal data.

The endpoint is intentionally stateless: callers (the frontend) own the
conversation history and replay it on each request. This keeps the API simple,
lets users clear history at will, and avoids server-side session storage.
History is capped at the last `MAX_HISTORY_TURNS` user+assistant turns.

The agent reaches its data through the tools in [src/agent/tools.py] rather than
receiving a dump of it in the system prompt, so context stays bounded however
large the watchlist and position set grow. The prompt carries only enough
orientation for the model to know what is worth asking for.

The account highlighted on the dashboard (forwarded by the caller) is the one
analyzed by default per the prompt.

Each turn is archived (src/agent/archive.py) as a user row and an assistant row
sharing a conversation id: the caller's, or a new one returned for threading.
Memory (src/agent/memory.py) adds what the replayed history no longer carries
— older turns of the conversation, past portfolio snapshots, recent briefings.
"""
import hashlib
import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncEngine

from src.agent.archive import record_output
from src.agent.memory import AssembledContext, build_context, context_budget, estimate_tokens
from src.agent.tools import TOOLS, data_summary, known_accounts, mask_account, run_tool
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

_HIGHLIGHT_NOTE = (
    "\n\nThe user has highlighted account {account} on the dashboard. Analyze "
    "only that account; the account tools cover it unless you pass another "
    "account_id. Look at another account, or all of them, only when the user "
    "explicitly asks, and say which account each figure comes from when you do."
)

_STALE_AUTH_NOTE = (
    "\n\nNote: the user is not currently authenticated with E-Trade, so position "
    "and balance data may be stale. Flag this if relevant."
)


_MEMORY_INTRO = (
    "\n\nBackground from memory (earlier in this conversation, past portfolio "
    "snapshots, recent briefings). It may be out of date: use the tools for "
    "current figures.\n\n"
)
# Tool definitions are sent with every request; counted against the budget.
_TOOLS_TOKENS = estimate_tokens(
    json.dumps([{"name": t.name, "description": t.description, "parameters": t.parameters} for t in TOOLS])
)
# agent_outputs.conversation_id is 40 characters; longer ids are hashed to fit.
_CONVERSATION_ID_MAX = 40

_OVER_BUDGET_RESULT = (
    "This result ({tokens} tokens) does not fit in the {room} tokens of context "
    "left for this question, so it was not returned. Answer with what you have, "
    "or ask for less: one ticker, fewer bars, or screen_indicators to compare "
    "tickers."
)

_CUT_OFF_NOTE = (
    "\n\n_(The answer was cut off: the model reached its output limit. "
    "Try a narrower question.)_"
)
_TOOL_CEILING_NOTE = (
    "\n\n_(Stopped after {n} rounds of looking things up without an answer. "
    "Try a narrower question.)_"
)


@dataclass
class _Turn:
    """What a finished turn produced, for `ask` to return."""

    conversation_id: str
    answer: str = ""
    context_used: dict = field(default_factory=dict)


def conversation_key(conversation_id: str | None) -> str:
    """The caller's conversation id, hashed if too long to store; a new uuid4
    when there is none."""
    text = (conversation_id or "").strip()
    if not text:
        return str(uuid.uuid4())
    if len(text) <= _CONVERSATION_ID_MAX:
        return text
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class AgentChat:
    # Output budget per model turn. Reasoning models (gpt-5, Opus 5's adaptive
    # thinking) spend hidden reasoning tokens from this same budget, so at 2048
    # they could think through it entirely and return no text at all.
    MAX_TOKENS = 16_000
    MAX_HISTORY_TURNS = 10
    # Ceiling on model -> tools -> model round trips within a single question.
    MAX_TOOL_ITERATIONS = 5

    def __init__(self, engine: AsyncEngine, backend: LLMBackend, context_tokens: int | None = None) -> None:
        """`context_tokens` is the prompt budget; None takes the chat provider's."""
        self._engine = engine
        self._backend = backend
        self._context_tokens = context_budget("chat") if context_tokens is None else context_tokens

    async def ask(
        self,
        question: str,
        conversation_history: list[dict] | None = None,
        account_id: str | None = None,
        conversation_id: str | None = None,
    ) -> dict:
        """`account_id` is the highlighted account; None analyzes every account.
        The response carries the conversation id to send with the next turn."""
        turn = _Turn(conversation_key(conversation_id))
        async for _ in self._turn(question, conversation_history, account_id, turn):
            pass
        return {
            "answer": turn.answer,
            "conversation_id": turn.conversation_id,
            "context_used": turn.context_used,
        }

    async def ask_stream(
        self,
        question: str,
        conversation_history: list[dict] | None = None,
        account_id: str | None = None,
        conversation_id: str | None = None,
    ) -> AsyncIterator[Delta | ToolResult]:
        """`ask` in streaming form, for the AG-UI/CopilotKit endpoint.

        Yields provider deltas plus a `ToolResult` for each tool the agent runs.
        """
        turn = _Turn(conversation_key(conversation_id))
        async for item in self._turn(question, conversation_history, account_id, turn):
            yield item

    async def _turn(
        self,
        question: str,
        history: list[dict] | None,
        account_id: str | None,
        turn: _Turn,
    ) -> AsyncIterator[Delta | ToolResult]:
        """One question, start to finish: prompt, memory, the tool loop, and
        the archive. Fills `turn` once the answer is complete."""
        started = time.monotonic()
        system_prompt, summary, highlighted = await self._prepare(account_id)
        messages = self._build_messages(question, history)
        memory = await self._background(system_prompt, messages, turn.conversation_id, highlighted)
        if memory is not None and memory.sections:
            system_prompt += _MEMORY_INTRO + memory.render()
        room = self._context_tokens - _TOOLS_TOKENS - estimate_tokens(
            system_prompt + "".join(m.text or "" for m in messages)
        )
        # After memory is built, so the question is not its own background.
        await record_output(
            self._engine, "chat", question, self._backend.provider, self._backend.model,
            role="user", conversation_id=turn.conversation_id,
            context_ref={"highlighted_account": highlighted},
        )

        answer: list[str] = []
        tools_called: list[dict] = []
        async for item in self._run(system_prompt, messages, highlighted, room):
            if isinstance(item, TextDelta):
                answer.append(item.text)
            elif isinstance(item, MessageComplete):
                tools_called.extend(
                    {"name": c.name, "arguments": c.arguments}
                    for c in item.message.tool_calls
                )
            yield item

        turn.answer = "".join(answer).strip()
        turn.context_used = {
            "highlighted_account": highlighted,
            "tools_called": tools_called,
            "data_freshness_minutes": summary["data_freshness_minutes"],
            "memory": memory.summary() if memory is not None else None,
        }
        logger.info(
            "chat: {} -> {} chars, {} tool call(s) ({}/{})",
            question[:60].replace("\n", " "),
            len(turn.answer),
            len(tools_called),
            self._backend.provider,
            self._backend.model,
        )
        if turn.answer:
            await record_output(
                self._engine, "chat", turn.answer, self._backend.provider, self._backend.model,
                role="assistant", conversation_id=turn.conversation_id,
                context_ref=turn.context_used,
                latency_ms=int((time.monotonic() - started) * 1000),
            )

    async def _background(
        self,
        system_prompt: str,
        messages: list[Message],
        conversation_id: str,
        highlighted: str | None,
    ) -> AssembledContext | None:
        """Memory in what the budget leaves after the prompt, messages and tool
        definitions; None if memory is unavailable. Tool results arrive later
        and are not foreseen here."""
        used = estimate_tokens(system_prompt + _MEMORY_INTRO + "".join(m.text or "" for m in messages))
        budget = max(0, self._context_tokens - used - _TOOLS_TOKENS)
        try:
            return await build_context(
                self._engine, "chat", budget,
                conversation_id=conversation_id,
                account_id=highlighted,
                # Those turns are already in `messages`, replayed by the caller.
                skip_recent_turns=len(messages) - 1,
            )
        except Exception:
            logger.exception("chat: memory unavailable; answering without background")
            return None

    async def _prepare(self, account_id: str | None) -> tuple[str, dict, str | None]:
        """Build the system prompt from a cheap summary of what data exists."""
        async with self._engine.begin() as conn:
            highlighted = account_id
            if highlighted is not None and highlighted not in await known_accounts(conn):
                logger.warning("chat: highlighted account {} is not stored; using every account",
                               mask_account(highlighted))
                highlighted = None
            summary = await data_summary(conn, highlighted)

        prompt = _SYSTEM_PROMPT_TEMPLATE.format(facts=_format_facts(summary, highlighted))
        if highlighted is not None:
            prompt += _HIGHLIGHT_NOTE.format(account=mask_account(highlighted))
        if not await etrade_auth.is_authenticated():
            prompt += _STALE_AUTH_NOTE
        return prompt, summary, highlighted

    async def _run(
        self,
        system_prompt: str,
        messages: list[Message],
        highlighted: str | None,
        room: int,
    ) -> AsyncIterator[Delta | ToolResult]:
        """Drive model -> tools -> model until the model stops asking for tools.
        A result that would take the prompt past `room` tokens is withheld and
        the model told why."""
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
                result = await run_tool(self._engine, call, highlighted)
                tokens = estimate_tokens(result.content)
                if tokens > room:
                    logger.warning("chat: withheld {} result of ~{} tokens ({} left)", call.name, tokens, room)
                    result = ToolResult(
                        call.id, _OVER_BUDGET_RESULT.format(tokens=tokens, room=max(room, 0)), is_error=True
                    )
                    tokens = estimate_tokens(result.content)
                room -= tokens
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


def _format_facts(summary: dict, highlighted: str | None) -> str:
    freshness = summary["data_freshness_minutes"]
    holdings = (
        f"- {summary['positions']} positions in the highlighted account "
        f"{mask_account(highlighted)} ({summary['accounts']} account(s) stored)"
        if highlighted is not None
        else f"- {summary['positions']} positions across {summary['accounts']} account(s)"
    )
    lines = [
        f"- Watchlist: {', '.join(settings.WATCHLIST)}",
        holdings,
        f"- {summary['signals_recent']} signal(s) in the last 2 trading days",
    ]
    lines.append(
        "- No price bars stored yet; a backfill has not been run"
        if freshness < 0
        else f"- Most recent price bar is {freshness} minute(s) old"
    )
    return "\n".join(lines)
