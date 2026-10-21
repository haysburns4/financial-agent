"""Watchlist mutations: add, remove, restore, edit, and backfill-on-add.

Every write and the watchlist_events row describing it commit in one
transaction. Network work (the E-Trade symbol lookup, the yfinance backfill)
happens outside any transaction. Removal is a soft delete and never touches
price_bars, indicators, signals or snapshots.
"""
import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Protocol

from loguru import logger
from sqlalchemy import Row, func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from src.etrade.market import SymbolStatus
from src.models import Position, WatchlistEvent, WatchlistItem, as_utc
from src.pipelines.backfill_pipeline import BackfillResult

EventSource = Literal["ui", "api"]
EventKind = Literal["added", "removed", "restored", "note_changed", "tags_changed"]
BackfillStatus = Literal["pending", "running", "done", "failed"]
ErrorCode = Literal["invalid_ticker", "invalid_tag", "at_capacity", "unknown_symbol", "not_found", "not_failed"]

TICKER_PATTERN = re.compile(r"^[A-Z.\-]{1,10}$")
_MAX_TAGS = 20
_MAX_TAG_CHARS = 32


class WatchlistError(Exception):
    """A rejected mutation; `code` says which rule it broke."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class Unset(Enum):
    """Marks an update argument that was not passed (as opposed to None)."""

    UNSET = "unset"


UNSET = Unset.UNSET


class SymbolLookup(Protocol):
    async def lookup_symbol(self, ticker: str) -> SymbolStatus: ...


class Backfiller(Protocol):
    async def run(self, tickers: list[str]) -> BackfillResult: ...


@dataclass(frozen=True)
class WatchlistEntry:
    ticker: str
    note: str | None
    tags: list[str]
    added_at: datetime
    validated: bool
    backfill_status: BackfillStatus
    backfill_error: str | None
    backfill_done_at: datetime | None
    updated_at: datetime

    @classmethod
    def from_row(cls, row: Row) -> "WatchlistEntry":
        return cls(
            ticker=row.ticker,
            note=row.note,
            tags=list(row.tags or []),
            added_at=as_utc(row.added_at),
            validated=bool(row.validated),
            backfill_status=row.backfill_status,
            backfill_error=row.backfill_error,
            backfill_done_at=as_utc(row.backfill_done_at) if row.backfill_done_at else None,
            updated_at=as_utc(row.updated_at),
        )

    def as_json(self) -> dict[str, str | bool | list[str] | None]:
        return {
            "ticker": self.ticker,
            "note": self.note,
            "tags": self.tags,
            "added_at": self.added_at.isoformat(),
            "validated": self.validated,
            "backfill_status": self.backfill_status,
            "backfill_error": self.backfill_error,
            "backfill_done_at": self.backfill_done_at.isoformat() if self.backfill_done_at else None,
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(frozen=True)
class RemoveResult:
    entry: WatchlistEntry
    held: bool


def normalize_ticker(ticker: str) -> str:
    symbol = ticker.strip().upper()
    if not TICKER_PATTERN.match(symbol):
        raise WatchlistError(
            "invalid_ticker", f"{ticker.strip()!r} is not a ticker: use 1-10 letters, '.' or '-'"
        )
    return symbol


def clean_note(note: str | None) -> str | None:
    text = (note or "").strip()
    return text or None


def clean_tags(tags: list[str] | None) -> list[str]:
    out: list[str] = []
    for tag in tags or []:
        text = tag.strip()
        if len(text) > _MAX_TAG_CHARS:
            raise WatchlistError("invalid_tag", f"Tag {text[:12]!r}… is longer than {_MAX_TAG_CHARS} characters")
        if text and text not in out:
            out.append(text)
    if len(out) > _MAX_TAGS:
        raise WatchlistError("invalid_tag", f"At most {_MAX_TAGS} tags per ticker")
    return out


def _now() -> datetime:
    return datetime.now(timezone.utc)


class WatchlistService:
    def __init__(
        self,
        engine: AsyncEngine,
        lookup: SymbolLookup,
        backfiller: Backfiller,
        max_size: int,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self._engine = engine
        self._lookup = lookup
        self._backfiller = backfiller
        self._max_size = max_size
        self._clock = clock
        # One backfill per ticker at a time; kept referenced so tasks are not
        # collected mid-run.
        self._backfills: dict[str, asyncio.Task[None]] = {}

    # ---------- reads ----------

    async def list_active(self) -> list[WatchlistEntry]:
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    select(WatchlistItem).where(WatchlistItem.removed_at.is_(None)).order_by(WatchlistItem.ticker)
                )
            ).all()
        return [WatchlistEntry.from_row(r) for r in rows]

    async def get(self, ticker: str) -> WatchlistEntry | None:
        async with self._engine.connect() as conn:
            row = await _row(conn, normalize_ticker(ticker))
        return WatchlistEntry.from_row(row) if row is not None and row.removed_at is None else None

    # ---------- mutations ----------

    async def add(
        self,
        ticker: str,
        note: str | None = None,
        tags: list[str] | None = None,
        source: EventSource = "api",
    ) -> WatchlistEntry:
        """Add `ticker`, or restore it if it was removed. Already active: returned
        unchanged. The backfill is scheduled, never awaited."""
        symbol = normalize_ticker(ticker)
        new_note = clean_note(note) if note is not None else None
        new_tags = clean_tags(tags) if tags is not None else None

        async with self._engine.connect() as conn:
            existing = await _row(conn, symbol)
            if existing is not None and existing.removed_at is None:
                return WatchlistEntry.from_row(existing)
            await self._check_capacity(conn)

        # Outside any transaction: E-Trade can be slow.
        validated = existing is not None and bool(existing.validated)
        if not validated:
            status = await self._lookup.lookup_symbol(symbol)
            if status == "unknown":
                raise WatchlistError("unknown_symbol", f"E-Trade has no quote for {symbol}: unknown symbol")
            if status == "unavailable":
                logger.info("watchlist: {} added unvalidated (E-Trade unavailable); the next price run checks it", symbol)
            validated = status == "valid"

        now = self._clock()
        async with self._engine.begin() as conn:
            row = await _row(conn, symbol)
            if row is not None and row.removed_at is None:
                return WatchlistEntry.from_row(row)  # added concurrently
            await self._check_capacity(conn)
            if row is None:
                values = {
                    "ticker": symbol,
                    "note": new_note,
                    "tags": new_tags or [],
                    "added_at": now,
                    "validated": validated,
                    "backfill_status": "pending",
                }
                await conn.execute(insert(WatchlistItem).values(**values))
                await _event(conn, symbol, "added", {"note": new_note, "tags": new_tags or []}, source)
            else:
                note_kept = row.note if note is None else new_note
                tags_kept = list(row.tags or []) if new_tags is None else new_tags
                await conn.execute(
                    update(WatchlistItem)
                    .where(WatchlistItem.ticker == symbol)
                    .values(
                        removed_at=None,
                        note=note_kept,
                        tags=tags_kept,
                        validated=validated,
                        # Removal stopped live collection; backfill again to close the gap.
                        backfill_status="pending",
                        backfill_error=None,
                    )
                )
                await _event(
                    conn, symbol, "restored",
                    {"note": note_kept, "tags": tags_kept, "removed_at": as_utc(row.removed_at).isoformat()},
                    source,
                )
            entry = WatchlistEntry.from_row(await _row(conn, symbol))
        self.schedule_backfill(symbol)
        return entry

    async def remove(self, ticker: str, source: EventSource = "api") -> RemoveResult:
        """Soft-delete. A held ticker is removed from the watchlist but stays monitored."""
        symbol = normalize_ticker(ticker)
        async with self._engine.begin() as conn:
            row = await _row(conn, symbol)
            if row is None or row.removed_at is not None:
                raise WatchlistError("not_found", f"{symbol} is not on the watchlist")
            held = await _is_held(conn, symbol)
            await conn.execute(
                update(WatchlistItem).where(WatchlistItem.ticker == symbol).values(removed_at=self._clock())
            )
            await _event(conn, symbol, "removed", {"held": held}, source)
            entry = WatchlistEntry.from_row(await _row(conn, symbol))
        return RemoveResult(entry, held)

    async def update(
        self,
        ticker: str,
        *,
        note: str | None | Unset = UNSET,
        tags: list[str] | Unset = UNSET,
        source: EventSource = "api",
    ) -> WatchlistEntry:
        """Change the note and/or tags; an event per field that actually changed."""
        symbol = normalize_ticker(ticker)
        async with self._engine.begin() as conn:
            row = await _row(conn, symbol)
            if row is None or row.removed_at is not None:
                raise WatchlistError("not_found", f"{symbol} is not on the watchlist")
            values: dict[str, str | list[str] | None] = {}
            if not isinstance(note, Unset):  # anti-slop: allow no-defensive-isinstance - sentinel for "not passed"
                new_note = clean_note(note)
                if new_note != row.note:
                    values["note"] = new_note
                    await _event(conn, symbol, "note_changed", {"old_note": row.note, "new_note": new_note}, source)
            if not isinstance(tags, Unset):  # anti-slop: allow no-defensive-isinstance - sentinel for "not passed"
                new_tags = clean_tags(tags)
                old_tags = list(row.tags or [])
                if new_tags != old_tags:
                    values["tags"] = new_tags
                    await _event(conn, symbol, "tags_changed", {"old_tags": old_tags, "new_tags": new_tags}, source)
            if values:
                await conn.execute(update(WatchlistItem).where(WatchlistItem.ticker == symbol).values(**values))
            entry = WatchlistEntry.from_row(await _row(conn, symbol))
        return entry

    async def retry_backfill(self, ticker: str) -> WatchlistEntry:
        """Run a failed backfill again."""
        symbol = normalize_ticker(ticker)
        async with self._engine.begin() as conn:
            row = await _row(conn, symbol)
            if row is None or row.removed_at is not None:
                raise WatchlistError("not_found", f"{symbol} is not on the watchlist")
            if row.backfill_status != "failed":
                raise WatchlistError(
                    "not_failed", f"{symbol}'s backfill is {row.backfill_status}; only a failed one can be retried"
                )
            await conn.execute(
                update(WatchlistItem)
                .where(WatchlistItem.ticker == symbol)
                .values(backfill_status="pending", backfill_error=None)
            )
            entry = WatchlistEntry.from_row(await _row(conn, symbol))
        self.schedule_backfill(symbol)
        return entry

    # ---------- backfill ----------

    def schedule_backfill(self, ticker: str) -> None:
        """Start `ticker`'s backfill in the background unless one is running."""
        running = self._backfills.get(ticker)
        if running is not None and not running.done():
            return
        task = asyncio.create_task(self._backfill(ticker), name=f"watchlist-backfill-{ticker}")
        self._backfills[ticker] = task
        task.add_done_callback(lambda finished: self._forget(ticker, finished))

    def _forget(self, ticker: str, finished: asyncio.Task[None]) -> None:
        if self._backfills.get(ticker) is finished:
            del self._backfills[ticker]

    async def resume_backfills(self) -> list[str]:
        """Schedule every active row still pending or running (a restart
        interrupted it)."""
        async with self._engine.connect() as conn:
            unfinished = list(
                (
                    await conn.scalars(
                        select(WatchlistItem.ticker).where(
                            WatchlistItem.removed_at.is_(None),
                            WatchlistItem.backfill_status.in_(("pending", "running")),
                        )
                    )
                ).all()
            )
        for ticker in unfinished:
            self.schedule_backfill(ticker)
        return unfinished

    async def drain(self) -> None:
        """Wait for the backfills in flight (tests, shutdown)."""
        while self._backfills:
            await asyncio.gather(*self._backfills.values(), return_exceptions=True)

    async def _backfill(self, ticker: str) -> None:
        await self._set_backfill(ticker, "running", None)
        try:
            result = await self._backfiller.run([ticker])
        except Exception as exc:
            logger.exception("watchlist: backfill for {} raised", ticker)
            error: str | None = f"{type(exc).__name__}: {exc}"
        else:
            if result.errors:
                error = "; ".join(result.errors)
            elif result.daily_bars_added + result.intraday_bars_added == 0:
                error = "no price history came back for this symbol"
            else:
                error = None
        await self._set_backfill(ticker, "failed" if error else "done", error)
        logger.info("watchlist: backfill for {} {}", ticker, f"failed: {error}" if error else "done")

    async def _set_backfill(self, ticker: str, status: BackfillStatus, error: str | None) -> None:
        values: dict[str, str | datetime | None] = {"backfill_status": status, "backfill_error": error}
        if status == "done":
            values["backfill_done_at"] = self._clock()
        async with self._engine.begin() as conn:
            await conn.execute(update(WatchlistItem).where(WatchlistItem.ticker == ticker).values(**values))

    # ---------- helpers ----------

    async def _check_capacity(self, conn: AsyncConnection) -> None:
        active = await active_count(conn)
        if active >= self._max_size:
            raise WatchlistError(
                "at_capacity",
                f"The watchlist is full ({active} of {self._max_size}). Remove a ticker, "
                f"or raise WATCHLIST_MAX_SIZE.",
            )


async def active_count(conn: AsyncConnection) -> int:
    return (
        await conn.scalar(select(func.count(WatchlistItem.id)).where(WatchlistItem.removed_at.is_(None)))
    ) or 0


async def mark_validated(conn: AsyncConnection, tickers: list[str]) -> None:
    """E-Trade quoted these, so any watchlist row added while it was
    unavailable is now confirmed."""
    if tickers:
        await conn.execute(
            update(WatchlistItem)
            .where(WatchlistItem.ticker.in_(tickers), WatchlistItem.validated.is_(False))
            .values(validated=True)
        )


async def list_events(conn: AsyncConnection, ticker: str | None, limit: int) -> list[dict[str, str | int | dict | None]]:
    stmt = select(WatchlistEvent).order_by(WatchlistEvent.created_at.desc(), WatchlistEvent.id.desc()).limit(limit)
    if ticker:
        stmt = stmt.where(WatchlistEvent.ticker == normalize_ticker(ticker))
    return [
        {
            "id": r.id,
            "ticker": r.ticker,
            "event": r.event,
            "detail": r.detail,
            "source": r.source,
            "created_at": as_utc(r.created_at).isoformat(),
        }
        for r in (await conn.execute(stmt)).all()
    ]


async def _row(conn: AsyncConnection, ticker: str) -> Row | None:
    return (await conn.execute(select(WatchlistItem).where(WatchlistItem.ticker == ticker))).first()


async def _is_held(conn: AsyncConnection, ticker: str) -> bool:
    held = await conn.scalar(
        select(Position.id).where(Position.ticker == ticker, Position.quantity != 0).limit(1)
    )
    return held is not None


async def _event(
    conn: AsyncConnection, ticker: str, event: EventKind, detail: dict, source: EventSource
) -> None:
    await conn.execute(insert(WatchlistEvent).values(ticker=ticker, event=event, detail=detail, source=source))
