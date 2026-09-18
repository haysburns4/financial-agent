"""Interactive Q&A over portfolio + signal data, backed by Claude.

The endpoint is intentionally stateless: callers (the frontend) own the
conversation history and replay it on each request. This keeps the API
simple, lets users clear history at will, and avoids server-side session
storage. History is capped at the last `MAX_HISTORY_TURNS` user+assistant
turns to keep context bounded.
"""
import re
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from src.config import settings
from src.etrade.auth import auth as etrade_auth
from src.llm import Delta, LLMBackend, LLMError, Message, collect
from src.models import Indicator, Position, PriceBar, Signal


_SYSTEM_PROMPT_TEMPLATE = (
    "You are a financial analysis assistant with access to the user's equity "
    "portfolio data and technical indicators across multiple accounts. Answer "
    "questions about portfolio composition, market conditions, signal history, "
    "and indicator state. You have the following data available:\n\n"
    "{data_context}\n\n"
    "Never give direct buy/sell advice. If asked about data you don't have "
    "(real-time prices, news, fundamentals, predictions), say so clearly and "
    "offer what you can answer instead. Be concise. Reference specific tickers, "
    "positions, and signals by name when relevant."
)

_TICKER_RE = re.compile(r"\b([A-Z]{1,5}(?:\.[A-Z])?)\b")


class AgentChat:
    MAX_TOKENS = 2048
    MAX_HISTORY_TURNS = 10
    BARS_PER_TICKER = 20
    SIGNAL_LOOKBACK_HOURS = 24

    def __init__(self, engine: AsyncEngine, backend: LLMBackend) -> None:
        self._engine = engine
        self._backend = backend

    async def ask(
        self,
        question: str,
        conversation_history: list[dict] | None = None,
    ) -> dict:
        system_prompt, messages, ctx = await self._prepare(question, conversation_history)
        try:
            response = await collect(
                self._backend.stream(
                    system=system_prompt,
                    messages=messages,
                    max_tokens=self.MAX_TOKENS,
                )
            )
        except LLMError:
            logger.exception("AgentChat: {} call failed", self._backend.provider)
            raise

        if response.stop_reason == "refusal":
            logger.warning("chat: provider refused the request")

        answer = response.text.strip()
        logger.info(
            "chat: {} -> {} chars ({} in / {} out tokens, {}/{})",
            question[:60].replace("\n", " "),
            len(answer),
            response.usage.input_tokens,
            response.usage.output_tokens,
            self._backend.provider,
            self._backend.model,
        )

        return {"answer": answer, "context_used": self._context_used(answer, ctx)}

    async def ask_stream(
        self,
        question: str,
        conversation_history: list[dict] | None = None,
    ) -> AsyncIterator[Delta]:
        """`ask` in streaming form, for the AG-UI/CopilotKit endpoint.

        The terminal `MessageComplete` carries the full answer, so a caller
        wanting `context_used` can build it from there via `_context_used`.
        """
        system_prompt, messages, _ = await self._prepare(question, conversation_history)
        try:
            async for delta in self._backend.stream(
                system=system_prompt,
                messages=messages,
                max_tokens=self.MAX_TOKENS,
            ):
                yield delta
        except LLMError:
            logger.exception("AgentChat: {} stream failed", self._backend.provider)
            raise

    async def _prepare(
        self,
        question: str,
        conversation_history: list[dict] | None,
    ) -> tuple[str, list[Message], dict]:
        """Load context, render the system prompt, and build the turn list."""
        async with self._engine.begin() as conn:
            bars_by_ticker = await self._load_bars(conn)
            positions = await self._load_positions(conn)
            recent_signals = await self._load_recent_signals(conn)

        data_context = self._format_data_context(bars_by_ticker, positions, recent_signals)
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(data_context=data_context)
        if not await etrade_auth.is_authenticated():
            system_prompt += (
                "\n\nNote: the user is not currently authenticated with E-Trade, "
                "so position and balance data may be stale. Flag this if relevant."
            )

        ctx = {
            "bars_by_ticker": bars_by_ticker,
            "positions": positions,
            "recent_signals": recent_signals,
            "data_freshness_minutes": self._compute_freshness(bars_by_ticker),
        }
        return system_prompt, self._build_messages(question, conversation_history), ctx

    def _context_used(self, answer: str, ctx: dict) -> dict:
        return {
            "tickers_referenced": self._extract_tickers(answer, ctx["bars_by_ticker"].keys()),
            "positions_referenced": self._extract_position_tickers(answer, ctx["positions"]),
            "signals_referenced": self._extract_signal_ids(answer, ctx["recent_signals"]),
            "data_freshness_minutes": ctx["data_freshness_minutes"],
        }

    # ---------- context loaders ----------

    async def _load_bars(self, conn: AsyncConnection) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for ticker in settings.WATCHLIST:
            ticker = ticker.strip().upper()
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
                .limit(self.BARS_PER_TICKER)
            )
            rows = (await conn.execute(stmt)).all()
            if not rows:
                continue
            out[ticker] = [
                {
                    "timestamp": _coerce_utc(r.timestamp),
                    "open": r.open, "high": r.high, "low": r.low,
                    "close": r.close, "volume": r.volume,
                    "rsi_14": r.rsi_14, "macd_line": r.macd_line,
                    "macd_signal": r.macd_signal, "macd_hist": r.macd_hist,
                    "ema_9": r.ema_9, "ema_21": r.ema_21,
                }
                for r in reversed(rows)
            ]
        return out

    async def _load_positions(self, conn: AsyncConnection) -> list[dict]:
        stmt = select(Position).order_by(Position.market_value.desc())
        rows = (await conn.execute(stmt)).all()
        return [
            {
                "account_id": r.account_id,
                "ticker": r.ticker,
                "quantity": r.quantity,
                "cost_basis": r.cost_basis,
                "market_value": r.market_value,
            }
            for r in rows
        ]

    async def _load_recent_signals(self, conn: AsyncConnection) -> list[dict]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.SIGNAL_LOOKBACK_HOURS)
        stmt = (
            select(Signal)
            .where(Signal.created_at >= cutoff)
            .order_by(Signal.created_at.desc())
        )
        rows = (await conn.execute(stmt)).all()
        return [
            {
                "id": r.id,
                "ticker": r.ticker,
                "signal_type": r.signal_type,
                "direction": r.direction,
                "confidence": r.confidence,
                "reasoning": r.reasoning,
                "timestamp": _coerce_utc(r.timestamp),
            }
            for r in rows
        ]

    # ---------- formatting ----------

    @staticmethod
    def _compute_freshness(bars_by_ticker: dict[str, list[dict]]) -> int:
        latest = max(
            (bars[-1]["timestamp"] for bars in bars_by_ticker.values() if bars),
            default=None,
        )
        if latest is None:
            return -1
        age = datetime.now(timezone.utc) - latest
        return max(0, int(age.total_seconds() // 60))

    @staticmethod
    def _format_data_context(
        bars_by_ticker: dict[str, list[dict]],
        positions: list[dict],
        signals: list[dict],
    ) -> str:
        lines: list[str] = []

        lines.append(f"### Positions ({len(positions)} total)")
        if positions:
            for p in positions[:50]:
                pnl_pct = (
                    (p["market_value"] - p["cost_basis"]) / p["cost_basis"]
                    if p["cost_basis"]
                    else 0.0
                )
                lines.append(
                    f"- {p['ticker']} ({p['account_id']}): qty={p['quantity']:.4g}, "
                    f"cost=${p['cost_basis']:.0f}, mkt=${p['market_value']:.0f}, "
                    f"pnl={pnl_pct:+.1%}"
                )
        else:
            lines.append("- (no positions)")

        lines.append(f"\n### Recent signals (last 24h, {len(signals)} total)")
        if signals:
            for s in signals[:30]:
                ts = s["timestamp"].isoformat() if s["timestamp"] else "?"
                lines.append(
                    f"- #{s['id']} [{s['signal_type']}] {s['ticker']} "
                    f"dir={s['direction']} conf={s['confidence']:.2f} @ {ts} — "
                    f"{s['reasoning']}"
                )
        else:
            lines.append("- (no signals fired in the last 24h)")

        lines.append("\n### Watchlist bars + indicators (most recent up to 20 each)")
        if bars_by_ticker:
            for ticker, bars in bars_by_ticker.items():
                last = bars[-1]
                ts = last["timestamp"].isoformat() if last["timestamp"] else "?"
                lines.append(
                    f"- {ticker}: last={last['close']:.2f} @ {ts}, "
                    f"rsi={_fmt(last['rsi_14'])}, macd_hist={_fmt(last['macd_hist'])}, "
                    f"ema9={_fmt(last['ema_9'])}, ema21={_fmt(last['ema_21'])}, "
                    f"({len(bars)} bars available)"
                )
        else:
            lines.append("- (no bars loaded; backfill the watchlist first)")

        return "\n".join(lines)

    def _build_messages(
        self,
        question: str,
        history: list[dict] | None,
    ) -> list[Message]:
        messages: list[Message] = []
        if history:
            trimmed = history[-(self.MAX_HISTORY_TURNS * 2):]
            for turn in trimmed:
                role = turn.get("role")
                content = turn.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append(Message(role=role, text=content))
        messages.append(Message(role="user", text=question))
        return messages

    # ---------- context-used extraction ----------

    @staticmethod
    def _extract_tickers(answer: str, candidates) -> list[str]:
        candidate_set = {t.upper() for t in candidates}
        if not candidate_set:
            return []
        found = {m.upper() for m in _TICKER_RE.findall(answer)}
        return sorted(found & candidate_set)

    @staticmethod
    def _extract_position_tickers(answer: str, positions: list[dict]) -> list[str]:
        candidate_set = {p["ticker"] for p in positions}
        if not candidate_set:
            return []
        found = {m.upper() for m in _TICKER_RE.findall(answer)}
        return sorted(found & candidate_set)

    @staticmethod
    def _extract_signal_ids(answer: str, signals: list[dict]) -> list[int]:
        id_to_ticker = {s["id"]: s["ticker"] for s in signals}
        if not id_to_ticker:
            return []
        # Match "#42" or "signal 42" / "signal #42"
        explicit_ids = set()
        for m in re.finditer(r"(?:signal\s*#?|#)(\d+)", answer, re.IGNORECASE):
            try:
                sid = int(m.group(1))
                if sid in id_to_ticker:
                    explicit_ids.add(sid)
            except ValueError:
                continue
        # Plus: any signal whose ticker appears in the answer
        mentioned_tickers = {m.upper() for m in _TICKER_RE.findall(answer)}
        by_ticker = {sid for sid, ticker in id_to_ticker.items() if ticker in mentioned_tickers}
        return sorted(explicit_ids | by_ticker)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def _coerce_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return None
