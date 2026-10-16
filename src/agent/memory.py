"""Memory assembler: background context for a prompt, fitted to a token budget.

Draws on the agent output archive and position snapshots, never on live data:
chat reaches current positions and signals through its tools, and synthesis is
handed its signals as input. This module only supplies what came before.

Two tasks:
- chat: breadth. Earlier turns of the conversation, portfolio totals at a few
  points in time, and recent briefings on any ticker.
- synthesis: depth. The latest snapshot of the tickers in the batch, and the
  prior briefings about them in full, so the model can say what is new rather
  than restate yesterday's observation.

WHICH archived narratives are eligible is decided by `select_outputs` alone —
the seam a semantic retriever (and reranker) replaces. `build_context` decides
how they are rendered and what fits.

Fitting counts tokens with the caller's estimator (`LLMBackend.estimate_tokens`).
"""
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Literal

from sqlalchemy import Row, func, select
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from src.agent.tools import mask_account
from src.llm import estimate_tokens
from src.models import AgentOutput, AgentOutputRef, PositionSnapshot

MemoryTask = Literal["chat", "synthesis"]
SectionLabel = Literal["portfolio", "conversation", "prior_narratives"]

_TITLES: dict[SectionLabel, str] = {
    "portfolio": "Portfolio history",
    "conversation": "Earlier in this conversation",
    "prior_narratives": "Prior briefings",
}
_CONVERSATION_TURNS = 20
_CHAT_NARRATIVES = 10
_SYNTHESIS_NARRATIVES = 20
# Chat skims many briefings; synthesis reads the relevant ones whole.
_CHAT_NARRATIVE_CHARS = 800
# Chat shows portfolio totals on the latest snapshot date and this many earlier.
_CHAT_HISTORY_POINTS = 4
_TRIMMED = " …[trimmed]"


@dataclass
class ContextSection:
    label: SectionLabel
    content: str
    tokens_estimated: int


@dataclass
class AssembledContext:
    sections: list[ContextSection]
    tokens_estimated: int
    budget_tokens: int
    # Labels of sections cut short or left out for lack of budget.
    truncated: list[str]
    # Archived outputs included (prior briefings, earlier turns), for provenance.
    output_ids: list[int] = field(default_factory=list)

    def render(self) -> str:
        return _render(self.sections)

    def summary(self) -> dict:
        """What was included, for logs and an output's context_ref."""
        return {
            "sections": {s.label: s.tokens_estimated for s in self.sections},
            "tokens_estimated": self.tokens_estimated,
            "budget_tokens": self.budget_tokens,
            "truncated": self.truncated,
            "output_ids": self.output_ids,
        }


@dataclass
class _Item:
    text: str
    output_id: int | None = None


@dataclass
class _Candidate:
    label: SectionLabel
    # Most important first: packing drops from the end.
    items: list[_Item]
    # Shown oldest first (a conversation) rather than in priority order.
    chronological: bool = False


async def select_outputs(
    engine: AsyncEngine,
    task: MemoryTask,
    scope_tickers: list[str] | None,
    lookback_days: int,
    limit: int,
) -> list[Row]:
    """The archived narratives eligible as background for `task`.

    v1 is recency only: syntheses from the last `lookback_days`, newest first,
    limited to those referencing `scope_tickers` when given. Both tasks draw
    the same kind of rows today; `task` is for the retriever that replaces
    this, whose query will differ between them.
    """
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    stmt = select(AgentOutput).where(
        AgentOutput.output_type == "synthesis", AgentOutput.timestamp >= since
    )
    if scope_tickers:
        stmt = stmt.where(
            AgentOutput.id.in_(
                select(AgentOutputRef.output_id).where(
                    AgentOutputRef.ref_type == "ticker",
                    AgentOutputRef.ref_value.in_([t.upper() for t in scope_tickers]),
                )
            )
        )
    stmt = stmt.order_by(AgentOutput.timestamp.desc(), AgentOutput.id.desc()).limit(limit)
    async with engine.connect() as conn:
        return list((await conn.execute(stmt)).all())


async def build_context(
    engine: AsyncEngine,
    task: MemoryTask,
    budget_tokens: int,
    scope_tickers: list[str] | None = None,
    conversation_id: str | None = None,
    lookback_days: int = 30,
    *,
    account_id: str | None = None,
    skip_recent_turns: int = 0,
    count: Callable[[str], int] = estimate_tokens,
) -> AssembledContext:
    """Background for a `task` prompt, at most `budget_tokens` (estimated).

    `account_id` narrows chat's portfolio totals to the highlighted account.
    `skip_recent_turns` leaves out the newest turns of the conversation, which
    the caller already sends as messages. `count` is the model's token
    estimator; the default errs high for an unknown tokenizer.
    """
    if task == "chat":
        narratives = await select_outputs(engine, task, None, lookback_days, _CHAT_NARRATIVES)
        async with engine.connect() as conn:
            candidates = [
                _Candidate("conversation", await _conversation(conn, conversation_id, skip_recent_turns), True),
                _Candidate("portfolio", await _portfolio_totals(conn, account_id, lookback_days)),
                _Candidate("prior_narratives", await _narratives(conn, narratives, _CHAT_NARRATIVE_CHARS)),
            ]
    else:
        narratives = (
            await select_outputs(engine, task, scope_tickers, lookback_days, _SYNTHESIS_NARRATIVES)
            if scope_tickers
            else []
        )
        async with engine.connect() as conn:
            candidates = [
                _Candidate("portfolio", await _portfolio_detail(conn, scope_tickers or [])),
                _Candidate("prior_narratives", await _narratives(conn, narratives, None)),
            ]
    return _pack(candidates, budget_tokens, count)


# ---------- packing ----------


def _pack(candidates: list[_Candidate], budget: int, count: Callable[[str], int]) -> AssembledContext:
    """Fill sections in priority order, item by item, while the whole rendering
    fits. An item too big for what is left is trimmed if it is a section's
    first; otherwise the section stops there."""
    sections: list[ContextSection] = []
    truncated: list[str] = []
    output_ids: list[int] = []
    for candidate in candidates:
        chosen: list[_Item] = []
        for item in candidate.items:
            if _fits(sections, candidate, [*chosen, item], budget, count):
                chosen.append(item)
                continue
            if not chosen and (trimmed := _trim_to_fit(sections, candidate, item, budget, count)):
                chosen.append(trimmed)
            break
        if len(chosen) < len(candidate.items) or any(i.text.endswith(_TRIMMED) for i in chosen):
            truncated.append(candidate.label)
        if chosen:
            sections.append(_section(candidate, chosen, count))
            output_ids.extend(i.output_id for i in chosen if i.output_id is not None)
    return AssembledContext(sections, count(_render(sections)), budget, truncated, output_ids)


def _section(candidate: _Candidate, items: list[_Item], count: Callable[[str], int]) -> ContextSection:
    ordered = list(reversed(items)) if candidate.chronological else items
    content = "\n\n".join(i.text for i in ordered)
    return ContextSection(
        candidate.label, content, count(_render([ContextSection(candidate.label, content, 0)]))
    )


def _fits(
    sections: list[ContextSection], candidate: _Candidate, items: list[_Item], budget: int, count: Callable[[str], int]
) -> bool:
    return count(_render([*sections, _section(candidate, items, count)])) <= budget


def _trim_to_fit(
    sections: list[ContextSection], candidate: _Candidate, item: _Item, budget: int, count: Callable[[str], int]
) -> _Item | None:
    """`item` cut to the longest prefix that fits, or None if not even a line does."""
    low, high, best = 0, len(item.text), None
    while low <= high:
        mid = (low + high) // 2
        trial = _Item(item.text[:mid].rstrip() + _TRIMMED, item.output_id)
        if _fits(sections, candidate, [trial], budget, count):
            best, low = trial, mid + 1
        else:
            high = mid - 1
    return best if best is not None and len(best.text) > len(_TRIMMED) + 20 else None


def _render(sections: list[ContextSection]) -> str:
    return "\n\n".join(f"## {_TITLES[s.label]}\n{s.content}" for s in sections)


# ---------- sources ----------


async def _conversation(conn: AsyncConnection, conversation_id: str | None, skip: int) -> list[_Item]:
    """Earlier turns, newest first (the order they are dropped in)."""
    if conversation_id is None:
        return []
    rows = (
        await conn.execute(
            select(AgentOutput)
            .where(AgentOutput.output_type == "chat", AgentOutput.conversation_id == conversation_id)
            .order_by(AgentOutput.timestamp.desc(), AgentOutput.id.desc())
            .offset(skip)
            .limit(_CONVERSATION_TURNS)
        )
    ).all()
    return [_Item(f"{r.role or 'assistant'} ({_when(r.timestamp)}): {r.content}", r.id) for r in rows]


async def _narratives(conn: AsyncConnection, rows: list[Row], max_chars: int | None) -> list[_Item]:
    if not rows:
        return []
    tickers: dict[int, list[str]] = {}
    refs = await conn.execute(
        select(AgentOutputRef.output_id, AgentOutputRef.ref_value)
        .where(AgentOutputRef.output_id.in_([r.id for r in rows]), AgentOutputRef.ref_type == "ticker")
        .order_by(AgentOutputRef.id)
    )
    for output_id, ticker in refs.all():
        tickers.setdefault(output_id, []).append(ticker)
    items = []
    for r in rows:
        body = r.content if max_chars is None or len(r.content) <= max_chars else r.content[:max_chars].rstrip() + "…"
        about = f" · {', '.join(tickers[r.id])}" if r.id in tickers else ""
        items.append(_Item(f"[{_when(r.timestamp)}{about}]\n{body}", r.id))
    return items


async def _portfolio_totals(conn: AsyncConnection, account_id: str | None, lookback_days: int) -> list[_Item]:
    """Per-account totals on the latest snapshot date and a few earlier ones."""
    since = datetime.now(timezone.utc).date() - timedelta(days=lookback_days)
    stmt = (
        select(
            PositionSnapshot.snapshot_date,
            PositionSnapshot.account_id,
            func.max(PositionSnapshot.market_state),
            func.count(),
            func.sum(PositionSnapshot.market_value),
            func.sum(PositionSnapshot.cost_basis),
        )
        .where(PositionSnapshot.snapshot_date >= since)
        .group_by(PositionSnapshot.snapshot_date, PositionSnapshot.account_id)
        .order_by(PositionSnapshot.snapshot_date.desc(), PositionSnapshot.account_id)
    )
    if account_id is not None:
        stmt = stmt.where(PositionSnapshot.account_id == account_id)
    by_date: dict[tuple[date, str], list[str]] = {}
    for day, acct, state, count, value, cost in (await conn.execute(stmt)).all():
        by_date.setdefault((day, state), []).append(
            f"{mask_account(acct)} {_usd(value)} in {count} position(s), P&L {_pnl(value, cost)}"
        )
    dates = list(by_date)
    return [_Item(f"{day} ({state}): " + "; ".join(by_date[(day, state)])) for day, state in _spread(dates)]


async def _portfolio_detail(conn: AsyncConnection, tickers: list[str]) -> list[_Item]:
    """The latest snapshot's rows for `tickers`, largest first."""
    if not tickers:
        return []
    latest = await conn.scalar(select(func.max(PositionSnapshot.snapshot_date)))
    if latest is None:
        return []
    rows = (
        await conn.execute(
            select(PositionSnapshot)
            .where(
                PositionSnapshot.snapshot_date == latest,
                PositionSnapshot.ticker.in_([t.upper() for t in tickers]),
            )
            .order_by(PositionSnapshot.market_value.desc())
        )
    ).all()
    return [
        _Item(
            f"{r.ticker} in {mask_account(r.account_id)} on {r.snapshot_date} ({r.market_state}): "
            f"{r.quantity:g} shares, {_usd(r.market_value)}, P&L {_pnl(r.market_value, r.cost_basis)}"
        )
        for r in rows
    ]


def _spread(dates: list[tuple[date, str]]) -> list[tuple[date, str]]:
    """The newest date plus up to _CHAT_HISTORY_POINTS earlier ones, evenly spaced."""
    if len(dates) <= _CHAT_HISTORY_POINTS + 1:
        return dates
    earlier = dates[1:]
    step = len(earlier) / _CHAT_HISTORY_POINTS
    return [dates[0], *(earlier[min(len(earlier) - 1, int((i + 1) * step) - 1)] for i in range(_CHAT_HISTORY_POINTS))]


def _when(moment: datetime) -> str:
    utc = moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)
    return utc.strftime("%Y-%m-%d %H:%M UTC")


def _usd(value: float) -> str:
    return f"${value:,.0f}"


def _pnl(value: float, cost: float) -> str:
    pnl = value - cost
    sign = "+" if pnl >= 0 else "-"
    pct = f" ({pnl / cost:+.1%})" if cost else ""
    return f"{sign}${abs(pnl):,.0f}{pct}"
