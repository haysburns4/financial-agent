"""Snapshot pipeline: copy every current position into the append-only history."""
import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Protocol

from loguru import logger
from sqlalchemy import delete, func, select, tuple_
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from src.etrade.auth import auth as etrade_auth
from src.market_calendar import MarketState, market_moment
from src.models import PipelineRun, Position, PositionSnapshot, as_utc
from src.pipelines.portfolio_pipeline import PortfolioPipeline
from src.portfolio import position_dict

# Positions older than this are refreshed from E-Trade before a snapshot.
STALE_AFTER_MINUTES = 30
PIPELINE_NAME = "portfolio_history_pipeline"


class Authenticator(Protocol):
    async def is_authenticated(self) -> bool: ...


@dataclass
class SnapshotRunResult:
    snapshot_date: date
    captured_at: datetime
    capture_source: str
    market_state: MarketState
    accounts: list[dict] = field(default_factory=list)
    rows_written: int = 0
    positions_refreshed: bool = False
    # Age of the snapshotted values at capture; -1 when there were none.
    positions_age_minutes: int = -1
    # True if this trading day already had rows, which this run replaced.
    was_overwrite: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def fresh(self) -> bool:
        """Whether the day's record reflects current positions."""
        return self.rows_written > 0 and 0 <= self.positions_age_minutes <= STALE_AFTER_MINUTES


class PortfolioHistoryPipeline:
    def __init__(
        self,
        engine: AsyncEngine,
        *,
        portfolio: PortfolioPipeline | None = None,
        auth: Authenticator = etrade_auth,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        """`portfolio` refreshes stale positions first."""
        self._engine = engine
        self._portfolio = portfolio
        self._auth = auth
        self._clock = clock
        # Startup and post-login captures can overlap; they take turns.
        self._lock = asyncio.Lock()
        self.last_result: SnapshotRunResult | None = None

    async def run(self, source: str = "manual", refresh_first: bool = True) -> SnapshotRunResult:
        """Capture today's snapshot. Never raises: failures land in `errors`."""
        async with self._lock:
            captured_at = self._clock()
            snapshot_date, market_state = market_moment(captured_at)
            result = SnapshotRunResult(snapshot_date, captured_at, source, market_state)
            try:
                await self._run(result, refresh_first)
            except Exception as exc:
                logger.exception("snapshot failed")
                result.errors.append(f"{type(exc).__name__}: {exc}")
                await self._record_run(result, "failed")
            self.last_result = result
            return result

    async def needs_capture(self) -> bool:
        """Whether today's trading day lacks a snapshot of current positions."""
        snapshot_date, _ = market_moment(self._clock())
        last = self.last_result
        if last is not None and last.snapshot_date == snapshot_date:
            return not last.fresh
        async with self._engine.connect() as conn:
            existing = await conn.scalar(
                select(func.count()).where(PositionSnapshot.snapshot_date == snapshot_date)
            )
        return not existing

    async def _run(self, result: SnapshotRunResult, refresh_first: bool) -> None:
        authenticated = await self._auth.is_authenticated()
        age = await self._positions_age(result.captured_at)

        if refresh_first and (age is None or age > STALE_AFTER_MINUTES):
            result.positions_refreshed = await self._refresh(result, authenticated)
            if result.positions_refreshed:
                age = await self._positions_age(self._clock())

        async with self._engine.begin() as conn:
            rows = (await conn.execute(select(Position))).all()
            if not rows:
                if not authenticated:
                    result.errors.append(
                        "No positions stored and E-Trade is not authenticated: log in, "
                        "then capture again (POST /pipeline/portfolio_history/run)."
                    )
                    await self._record_run(result, "auth_required", conn)
                else:
                    logger.info("Snapshot {}: no positions to record", result.snapshot_date)
                    await self._record_run(result, "empty", conn)
                return

            result.positions_age_minutes = age if age is not None else -1
            result.was_overwrite = bool(await conn.scalar(
                select(func.count()).where(PositionSnapshot.snapshot_date == result.snapshot_date)
            ))
            snapshot = [self._snapshot_row(r, result) for r in rows]
            # Last capture of the day wins: replace the day's rows wholesale, so
            # a position closed since an earlier capture leaves no row behind.
            await conn.execute(
                delete(PositionSnapshot)
                .where(PositionSnapshot.snapshot_date == result.snapshot_date)
                .where(
                    tuple_(PositionSnapshot.account_id, PositionSnapshot.ticker).not_in(
                        [(r.account_id, r.ticker) for r in rows]
                    )
                )
            )
            stmt = sqlite_insert(PositionSnapshot).values(snapshot)
            replaced = {c: stmt.excluded[c] for c in snapshot[0] if c not in ("snapshot_date", "account_id", "ticker")}
            await conn.execute(
                stmt.on_conflict_do_update(
                    index_elements=["snapshot_date", "account_id", "ticker"], set_=replaced
                )
            )
            result.rows_written = len(snapshot)
            result.accounts = _account_totals(snapshot)
            await self._record_run(result, "partial" if result.errors else "ok", conn)

        total = sum(a["total_market_value"] for a in result.accounts)
        logger.info(
            "Snapshot {} ({}, {}): {} rows across {} account(s), ${:,.0f} total, positions {}m old{}",
            result.snapshot_date, result.market_state, result.capture_source,
            result.rows_written, len(result.accounts), total, result.positions_age_minutes,
            "" if result.positions_refreshed or result.fresh else " (not refreshed)",
        )

    async def _refresh(self, result: SnapshotRunResult, authenticated: bool) -> bool:
        """Pull current positions from E-Trade."""
        if self._portfolio is None:
            return False
        if not authenticated:
            result.errors.append("Positions not refreshed: E-Trade is not authenticated.")
            return False
        refresh = await self._portfolio.run()
        result.errors.extend(f"refresh: {e}" for e in refresh.errors)
        if refresh.status not in ("ok", "partial"):
            result.errors.append(f"Positions not refreshed: portfolio pipeline {refresh.status}.")
            return False
        return True

    async def _positions_age(self, now: datetime) -> int | None:
        async with self._engine.connect() as conn:
            latest = await conn.scalar(select(func.max(Position.last_updated)))
        if latest is None:
            return None
        return max(0, int((now - as_utc(latest)).total_seconds() // 60))

    @staticmethod
    def _snapshot_row(row, result: SnapshotRunResult) -> dict:
        # position_dict is what the portfolio endpoints serve, so the numbers agree.
        p = position_dict(row)
        return {
            "snapshot_date": result.snapshot_date,
            "captured_at": result.captured_at,
            "capture_source": result.capture_source,
            "market_state": result.market_state,
            "account_id": p["account_id"],
            "ticker": p["ticker"],
            "quantity": p["quantity"],
            "cost_basis": p["cost_basis"],
            "market_value": p["market_value"],
            "pnl": p["pnl"],
            "pnl_pct": p["pnl_pct"],
            "positions_as_of": as_utc(row.last_updated),
        }

    async def _record_run(
        self, result: SnapshotRunResult, status: str, conn: AsyncConnection | None = None
    ) -> None:
        """A pipeline_runs row, in `conn`'s transaction if given, else its own."""
        values = {
            "pipeline": PIPELINE_NAME,
            "started_at": result.captured_at,
            "completed_at": datetime.now(timezone.utc),
            "status": status,
            "tickers_processed": result.rows_written,
            "errors": "\n".join(result.errors) or None,
        }
        if conn is not None:
            await conn.execute(sqlite_insert(PipelineRun).values(values))
            return
        async with self._engine.begin() as own:
            await own.execute(sqlite_insert(PipelineRun).values(values))


def _account_totals(rows: list[dict]) -> list[dict]:
    totals: dict[str, dict] = {}
    for r in rows:
        acct = totals.setdefault(
            r["account_id"], {"account_id": r["account_id"], "position_count": 0, "total_market_value": 0.0}
        )
        acct["position_count"] += 1
        acct["total_market_value"] += r["market_value"]
    return [totals[a] for a in sorted(totals)]
