"""The agent's read-only tool surface over the local database.

Replaces dumping every position, signal and bar into the system prompt: the
model asks for what it needs, so context stays bounded as the watchlist grows.

Arguments come from the model and are untrusted — every handler normalises
tickers and clamps limits rather than passing values straight into a query.

Account data (positions, risk, position alerts) covers the account highlighted
on the dashboard by default; a call reaches other accounts only by naming one,
or "all", in its `account_id` argument. Market data (entry/exit signals, price
history) is not per account and is never narrowed by account. On the
dashboard's Watchlist tab, signals and the indicator screen cover only
watchlist tickers.
"""
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger
from sqlalchemy import Row, and_, distinct, func, select
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from src.llm import ToolCall, ToolDef, ToolResult
from src.models import Indicator, Position, PriceBar, Signal
from src.portfolio import position_dict, risk_by_account
from src.signals.confirmation import CONFIRMATION_RULES, minus_trading_days, required_trend
from src.signals.engine import SIGNAL_CATEGORIES, signal_types_for_category
from src.watchlist.resolver import monitored_on

_CATEGORIES = sorted(set(SIGNAL_CATEGORIES.values()))
# An account overview ranks only this many signals; the rest are counted.
_TOP_SIGNALS = 10
_DEFAULT_SIGNAL_DAYS = 2
# Position alerts (stop-loss, concentration, drawdown) always fire with a
# fixed confidence, so ranking them against calibrated signals would put them
# all on top. They are listed on their own instead.
_ALERT_RULES = frozenset(r for r in CONFIRMATION_RULES if required_trend(r) is None)
# The `account_id` value that widens a call to every account.
ALL_ACCOUNTS = "all"


@dataclass(frozen=True)
class Scope:
    """What the dashboard is showing: the highlighted account (None for every
    account), and whether the Watchlist tab is open."""

    account: str | None = None
    watchlist_only: bool = False


def _account_arg(subject: str) -> dict:
    return {
        "type": "string",
        "description": (
            f"Whose {subject}: omit for the highlighted account (every account if "
            f"none is highlighted). Only when the user asks about another account "
            f"or all of them, pass its number (the last four digits are enough) or "
            f"\"{ALL_ACCOUNTS}\"."
        ),
    }


TOOLS: tuple[ToolDef, ...] = (
    ToolDef(
        name="get_positions",
        description=(
            "Current portfolio positions, with quantity, cost basis, market value "
            "and P&L. Use for holdings, position sizes and per-ticker exposure."
        ),
        parameters={
            "type": "object",
            "properties": {"account_id": _account_arg("positions")},
        },
    ),
    ToolDef(
        name="get_portfolio_risk",
        description=(
            "Total exposure, per-ticker concentration and drawdown, combined and "
            "broken down by account. Use for concentration or risk questions."
        ),
        parameters={
            "type": "object",
            "properties": {"account_id": _account_arg("risk")},
        },
    ),
    ToolDef(
        name="get_signals",
        description=(
            "Signals recorded over the last few trading days, dated by the bar "
            "they fired on (so this works outside market hours too): the 10 "
            "highest-confidence entry/exit signals with their rule, direction, "
            "calibrated confidence and reasoning; a count per rule for the rest; "
            "and the current position alerts (stop-loss, concentration, "
            "drawdown) for tickers held in the account, listed separately "
            "because their confidence is a fixed default, not a calibrated one."
        ),
        parameters={
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Restrict to one ticker."},
                "category": {
                    "type": "string",
                    "enum": _CATEGORIES,
                    "description": "Restrict to entry, exit or risk signals.",
                },
                "days": {
                    "type": "integer",
                    "description": f"Look-back in trading days (default {_DEFAULT_SIGNAL_DAYS}, max 30).",
                },
                "account_id": _account_arg("position alerts"),
            },
        },
    ),
    ToolDef(
        name="screen_indicators",
        description=(
            "The latest close, RSI, MACD histogram and EMAs for every watchlist "
            "ticker and every ticker held in the account, one row each, lowest "
            "RSI first. Use to compare or rank tickers (e.g. which are oversold) "
            "instead of calling get_price_history per ticker."
        ),
        parameters={
            "type": "object",
            "properties": {"account_id": _account_arg("holdings")},
        },
    ),
    ToolDef(
        name="get_price_history",
        description=(
            "Recent price bars for one ticker with their RSI, MACD and EMA values. "
            "Use for trend, momentum or 'what is the indicator doing' questions "
            "about a single ticker; to compare tickers, use screen_indicators."
        ),
        parameters={
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "limit": {
                    "type": "integer",
                    "description": "Number of most recent bars (default 30, max 200).",
                },
            },
            "required": ["ticker"],
        },
    ),
)


async def run_tool(engine: AsyncEngine, call: ToolCall, scope: Scope = Scope()) -> ToolResult:
    """Execute one tool call within `scope`, returning its result as JSON.

    Never raises: a failure is reported back to the model as an error result so
    it can recover or explain, rather than killing the run.
    """
    handler = _HANDLERS.get(call.name)
    if handler is None:
        logger.warning("tool: unknown tool {!r}", call.name)
        return ToolResult(call.id, f"unknown tool {call.name!r}", is_error=True)

    try:
        async with engine.begin() as conn:
            payload = await handler(conn, call.arguments, scope)
    except Exception as exc:
        logger.exception("tool: {} failed", call.name)
        return ToolResult(call.id, f"{type(exc).__name__}: {exc}", is_error=True)

    content = json.dumps(_rounded(payload), default=str)
    logger.info("tool: {}({}) -> {} chars", call.name, call.arguments, len(content))
    return ToolResult(call.id, content)


async def known_accounts(conn: AsyncConnection) -> list[str]:
    return sorted((await conn.scalars(select(distinct(Position.account_id)))).all())


async def data_summary(conn: AsyncConnection, highlighted: str | None = None) -> dict:
    """Counts and freshness for the system prompt, so the model knows what exists."""
    count = select(func.count(Position.id))
    if highlighted is not None:
        count = count.where(Position.account_id == highlighted)
    positions = await conn.scalar(count) or 0
    accounts = await conn.scalar(select(func.count(distinct(Position.account_id)))) or 0
    cutoff = minus_trading_days(datetime.now(timezone.utc), _DEFAULT_SIGNAL_DAYS)
    signals = await conn.scalar(
        select(func.count(Signal.id)).where(Signal.timestamp >= cutoff)
    ) or 0
    latest_bar = await conn.scalar(select(func.max(PriceBar.timestamp)))
    return {
        "positions": positions,
        "accounts": accounts,
        "signals_recent": signals,
        "data_freshness_minutes": _age_minutes(latest_bar),
    }


# ---------- handlers ----------


async def _account_scope(conn: AsyncConnection, args: dict, highlighted: str | None) -> str | None:
    """The account a call covers, or None for every account.

    No `account_id` argument means the highlighted account. Otherwise it is
    "all", an account number, or a unique last-digits suffix ("9991", "••9991"),
    the form the model is told to show accounts in.
    """
    requested = _text(args.get("account_id"))
    if requested is None:
        return highlighted
    if requested.lower() == ALL_ACCOUNTS:
        return None
    known = await known_accounts(conn)
    if requested in known:
        return requested
    digits = requested.lstrip("•*.… ")
    matches = [a for a in known if digits and a.endswith(digits)]
    if len(matches) == 1:
        return matches[0]
    choices = ", ".join(mask_account(a) for a in known) or "none stored"
    problem = "matches more than one account" if matches else "is not a known account"
    raise ValueError(f"account_id {requested!r} {problem} (accounts: {choices})")


async def _get_positions(conn: AsyncConnection, args: dict, scope: Scope) -> list[dict]:
    stmt = select(Position).order_by(Position.market_value.desc())
    if account_id := await _account_scope(conn, args, scope.account):
        stmt = stmt.where(Position.account_id == account_id)
    return [position_dict(r) for r in (await conn.execute(stmt)).all()]


async def _get_portfolio_risk(conn: AsyncConnection, args: dict, scope: Scope) -> dict:
    stmt = select(Position)
    if account_id := await _account_scope(conn, args, scope.account):
        stmt = stmt.where(Position.account_id == account_id)
    return risk_by_account((await conn.execute(stmt)).all())


def _signal_dict(r: Row) -> dict:
    return {
        "id": r.id,
        "ticker": r.ticker,
        "timestamp": r.timestamp,
        "signal_type": r.signal_type,
        "category": SIGNAL_CATEGORIES.get(r.signal_type),
        "direction": r.direction,
        "confidence": r.confidence,
        "reasoning": r.reasoning,
    }


async def _get_signals(conn: AsyncConnection, args: dict, scope: Scope) -> dict:
    """An overview that stays short however many signals fired: the top few by
    confidence, a count per rule, and position alerts on their own.

    Alerts are about positions, so they are narrowed to the tickers the account
    holds; entry/exit signals are about the market and are not. On the
    Watchlist tab, both cover only watchlist tickers."""
    days = _clamp(args.get("days"), default=_DEFAULT_SIGNAL_DAYS, low=1, high=30)
    cutoff = minus_trading_days(datetime.now(timezone.utc), days)
    stmt = select(Signal).where(Signal.timestamp >= cutoff)
    if ticker := _ticker(args.get("ticker")):
        stmt = stmt.where(Signal.ticker == ticker)
    if scope.watchlist_only:
        stmt = stmt.where(Signal.ticker.in_(await _watchlist_tickers(conn)))
    if category := _text(args.get("category")):
        stmt = stmt.where(Signal.signal_type.in_(signal_types_for_category(category)))
    rows = (await conn.execute(stmt)).all()
    held: set[str] | None = None
    if account_id := await _account_scope(conn, args, scope.account):
        held = set(
            (await conn.scalars(select(Position.ticker).where(Position.account_id == account_id))).all()
        )

    ranked = sorted(
        (r for r in rows if r.signal_type not in _ALERT_RULES),
        key=lambda r: (r.confidence, r.timestamp),
        reverse=True,
    )
    # Alerts repeat while a position stays down: keep the newest per ticker.
    alerts: dict[tuple[str, str], Row] = {}
    for r in sorted((r for r in rows if r.signal_type in _ALERT_RULES), key=lambda r: r.timestamp):
        if held is None or r.ticker in held:
            alerts[(r.ticker, r.signal_type)] = r
    return {
        "since": cutoff,
        "trading_days": days,
        "total_signals": len(ranked),
        "by_type": dict(Counter(r.signal_type for r in ranked).most_common()),
        "top": [_signal_dict(r) for r in ranked[:_TOP_SIGNALS]],
        "alerts": [_signal_dict(r) for _, r in sorted(alerts.items())],
    }


async def _get_price_history(conn: AsyncConnection, args: dict, scope: Scope) -> dict:
    ticker = _ticker(args.get("ticker"))
    if not ticker:
        raise ValueError("ticker is required")
    limit = _clamp(args.get("limit"), default=30, low=1, high=200)

    stmt = (
        select(
            PriceBar.timestamp, PriceBar.open, PriceBar.high, PriceBar.low,
            PriceBar.close, PriceBar.volume,
            Indicator.rsi_14, Indicator.macd_line, Indicator.macd_signal,
            Indicator.macd_hist, Indicator.ema_9, Indicator.ema_21,
        )
        .outerjoin(
            Indicator,
            and_(
                PriceBar.ticker == Indicator.ticker,
                PriceBar.timestamp == Indicator.timestamp,
            ),
        )
        .where(PriceBar.ticker == ticker)
        .order_by(PriceBar.timestamp.desc())
        .limit(limit)
    )
    rows = (await conn.execute(stmt)).all()
    return {
        "ticker": ticker,
        "bar_count": len(rows),
        "bars": [dict(r._mapping) for r in reversed(rows)],
    }


async def _screen_indicators(conn: AsyncConnection, args: dict, scope: Scope) -> dict:
    held_stmt = select(distinct(Position.ticker))
    if account_id := await _account_scope(conn, args, scope.account):
        held_stmt = held_stmt.where(Position.account_id == account_id)
    held = set((await conn.scalars(held_stmt)).all())
    # Monitored tickers, narrowed to the account's holdings plus the watchlist,
    # or to the watchlist alone on its tab.
    watched = await _watchlist_tickers(conn)
    tickers = watched if scope.watchlist_only else held | watched

    latest = (
        select(Indicator.ticker, func.max(Indicator.timestamp).label("timestamp"))
        .where(Indicator.ticker.in_(tickers))
        .group_by(Indicator.ticker)
        .subquery()
    )
    stmt = (
        select(
            Indicator.ticker, Indicator.timestamp, PriceBar.close,
            Indicator.rsi_14, Indicator.macd_hist, Indicator.ema_9, Indicator.ema_21,
        )
        .join(latest, and_(Indicator.ticker == latest.c.ticker, Indicator.timestamp == latest.c.timestamp))
        .outerjoin(PriceBar, and_(PriceBar.ticker == Indicator.ticker, PriceBar.timestamp == Indicator.timestamp))
    )
    rows = [{**r._mapping, "held": r.ticker in held} for r in (await conn.execute(stmt)).all()]
    rows.sort(key=lambda r: (r["rsi_14"] is None, r["rsi_14"] or 0.0))
    return {
        "tickers": rows,
        "no_indicators": sorted(tickers - {r["ticker"] for r in rows}),
    }


async def _watchlist_tickers(conn: AsyncConnection) -> set[str]:
    return {m.ticker for m in await monitored_on(conn) if m.source != "held"}


_HANDLERS = {
    "get_positions": _get_positions,
    "get_portfolio_risk": _get_portfolio_risk,
    "get_signals": _get_signals,
    "screen_indicators": _screen_indicators,
    "get_price_history": _get_price_history,
}
assert {t.name for t in TOOLS} == set(_HANDLERS), "TOOLS and _HANDLERS disagree"


# ---------- argument coercion ----------


def mask_account(account_id: str) -> str:
    """The last four digits, as the dashboard shows an account (••9991)."""
    return f"••{account_id[-4:]}"



def _text(value: Any) -> str | None:  # anti-slop: allow no-any-parameters - this IS the parse boundary for model-supplied JSON
    return value.strip() if isinstance(value, str) and value.strip() else None


def _ticker(value: Any) -> str | None:  # anti-slop: allow no-any-parameters - this IS the parse boundary for model-supplied JSON
    text = _text(value)
    return text.upper() if text else None


def _rounded(value: object) -> object:
    """`value` with floats cut to two decimals, or four below 1. Every model
    pays tokens for digits (some tokenizers one per digit), and float noise
    (177.89990234375) was most of a price-history payload."""
    if isinstance(value, float):
        return round(value, 2) if abs(value) >= 1 else round(value, 4)
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_rounded(v) for v in value]
    return value


def _clamp(value: Any, *, default: int, low: int, high: int) -> int:  # anti-slop: allow no-any-parameters - this IS the parse boundary for model-supplied JSON
    if not isinstance(value, int) or isinstance(value, bool):
        return default
    return max(low, min(high, value))


def _age_minutes(timestamp: datetime | None) -> int:
    """Minutes since `timestamp`, or -1 when there is no data at all."""
    if timestamp is None:
        return -1
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return max(0, int((datetime.now(timezone.utc) - timestamp).total_seconds() // 60))
