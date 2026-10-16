"""The agent output archive: every synthesis and chat turn, with provenance.

Rows are written once and never updated: a changed narrative is a new row, so
`content_hash` stays true to `content` for whatever later indexes the archive.
Each row's references (tickers, signal ids, accounts) are pulled out of its
context_ref into agent_output_refs in the same transaction, so filtering by
ticker or account is an indexed join.
"""
import hashlib
from collections.abc import Iterator
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Literal

from loguru import logger
from sqlalchemy import Row, func, insert, select
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from src.models import AgentOutput, AgentOutputRef

OutputType = Literal["synthesis", "chat"]
RefType = Literal["ticker", "signal_id", "account_id"]

# context_ref keys whose values are references, at any depth (so a chat tool
# call's {"ticker": "NVDA"} argument counts), and what each refers to.
_REF_KEYS: dict[str, RefType] = {
    "ticker": "ticker",
    "tickers": "ticker",
    "signal_id": "signal_id",
    "signal_ids": "signal_id",
    "account_id": "account_id",
    "account_ids": "account_id",
    "highlighted_account": "account_id",
}
# Not a reference: the tools' word for every account.
_NOT_REFS = frozenset({"all"})
_REF_VALUE_MAX = 50


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def refs_from_context(context_ref: dict | None) -> list[tuple[RefType, str]]:
    """The (ref_type, ref_value) pairs in a context_ref, deduplicated, in order."""
    seen: dict[tuple[RefType, str], None] = {}
    for ref_type, value in _walk(context_ref):
        text = str(value).strip()
        if ref_type == "ticker":
            text = text.upper()
        if text and text.lower() not in _NOT_REFS:
            seen[(ref_type, text[:_REF_VALUE_MAX])] = None
    return list(seen)


def _walk(node: Any, ref_type: RefType | None = None) -> Iterator[tuple[RefType, Any]]:  # anti-slop: allow no-any-parameters - context_ref is arbitrary JSON
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk(value, _REF_KEYS.get(key))
    elif isinstance(node, list | tuple):
        for item in node:
            yield from _walk(item, ref_type)
    elif ref_type is not None and node is not None and not isinstance(node, bool):
        yield ref_type, node


async def record_output(
    engine: AsyncEngine,
    output_type: OutputType,
    content: str,
    provider: str,
    model: str,
    role: str | None = None,
    conversation_id: str | None = None,
    context_ref: dict | None = None,
    latency_ms: int | None = None,
) -> int | None:
    """Archive one output and its references."""
    try:
        async with engine.begin() as conn:
            output_id = (
                await conn.execute(
                    insert(AgentOutput).values(
                        timestamp=datetime.now(timezone.utc),
                        output_type=output_type,
                        role=role,
                        content=content,
                        content_hash=content_hash(content),
                        provider=provider,
                        model=model,
                        conversation_id=conversation_id,
                        context_ref=context_ref,
                        latency_ms=latency_ms,
                    )
                    .returning(AgentOutput.id)
                )
            ).scalar_one()
            refs = refs_from_context(context_ref)
            if refs:
                await conn.execute(
                    insert(AgentOutputRef),
                    [{"output_id": output_id, "ref_type": t, "ref_value": v} for t, v in refs],
                )
        return output_id
    except Exception:
        logger.exception("archive: could not record {} output ({} chars)", output_type, len(content))
        return None


# ---------- reads ----------


def output_dict(row: Row) -> dict:
    return {
        "id": row.id,
        "timestamp": _utc(row.timestamp).isoformat(),
        "output_type": row.output_type,
        "role": row.role,
        "content": row.content,
        "content_hash": row.content_hash,
        "provider": row.provider,
        "model": row.model,
        "conversation_id": row.conversation_id,
        "context_ref": row.context_ref,
        "latency_ms": row.latency_ms,
    }


async def list_outputs(
    conn: AsyncConnection,
    *,
    output_type: str | None = None,
    conversation_id: str | None = None,
    ticker: str | None = None,
    start: date | None = None,
    end: date | None = None,
    limit: int = 100,
) -> list[dict]:
    """Archived outputs, newest first."""
    stmt = select(AgentOutput)
    if output_type is not None:
        stmt = stmt.where(AgentOutput.output_type == output_type)
    if conversation_id is not None:
        stmt = stmt.where(AgentOutput.conversation_id == conversation_id)
    if ticker is not None:
        stmt = stmt.where(
            AgentOutput.id.in_(
                select(AgentOutputRef.output_id).where(
                    AgentOutputRef.ref_type == "ticker", AgentOutputRef.ref_value == ticker.strip().upper()
                )
            )
        )
    if start is not None:
        stmt = stmt.where(AgentOutput.timestamp >= datetime.combine(start, time.min, timezone.utc))
    if end is not None:
        stmt = stmt.where(AgentOutput.timestamp < datetime.combine(end + timedelta(days=1), time.min, timezone.utc))
    stmt = stmt.order_by(AgentOutput.timestamp.desc(), AgentOutput.id.desc()).limit(limit)
    return [output_dict(r) for r in (await conn.execute(stmt)).all()]


async def get_output(conn: AsyncConnection, output_id: int) -> dict | None:
    """One output with its references, or None."""
    row = (await conn.execute(select(AgentOutput).where(AgentOutput.id == output_id))).first()
    if row is None:
        return None
    refs = await conn.execute(
        select(AgentOutputRef.ref_type, AgentOutputRef.ref_value)
        .where(AgentOutputRef.output_id == output_id)
        .order_by(AgentOutputRef.id)
    )
    return {**output_dict(row), "refs": [{"ref_type": t, "ref_value": v} for t, v in refs.all()]}


async def archive_summary(conn: AsyncConnection) -> dict:
    """The /health "agent_archive" section."""
    total, syntheses, chat_turns, latest = (
        await conn.execute(
            select(
                func.count(),
                func.count().filter(AgentOutput.output_type == "synthesis"),
                func.count().filter(AgentOutput.output_type == "chat"),
                func.max(AgentOutput.timestamp),
            )
        )
    ).one()
    return {
        "total_outputs": total,
        "syntheses": syntheses,
        "chat_turns": chat_turns,
        "latest_output_at": _utc(latest).isoformat() if latest else None,
    }


def _utc(moment: datetime) -> datetime:
    # SQLite drops the offset; the archive writes UTC.
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)
