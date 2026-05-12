"""Portfolio pipeline: pull E-Trade positions, upsert into the local DB.

Each E-Trade account is treated as a distinct portfolio. Positions are keyed by
(account_id, ticker), and per-account summaries are surfaced on every run.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone

from loguru import logger
from sqlalchemy import update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncEngine

from src.etrade.accounts import ETradeAccountClient
from src.etrade.auth import auth as etrade_auth
from src.models import PipelineRun, Position


@dataclass
class PortfolioRunResult:
    started_at: datetime
    completed_at: datetime
    status: str  # "ok" / "partial" / "failed" / "skipped"
    accounts_processed: int
    positions_stored: int
    accounts: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class PortfolioPipeline:
    def __init__(self, accounts: ETradeAccountClient, engine: AsyncEngine) -> None:
        self._accounts = accounts
        self._engine = engine

    async def run(self) -> PortfolioRunResult:
        started = datetime.now(timezone.utc)

        if not etrade_auth.is_authenticated():
            logger.warning("portfolio_pipeline skipped: E-Trade not authenticated")
            return PortfolioRunResult(
                started_at=started,
                completed_at=datetime.now(timezone.utc),
                status="skipped",
                accounts_processed=0,
                positions_stored=0,
            )

        logger.info("portfolio_pipeline starting")
        positions_stored = 0
        accounts_processed = 0
        account_summaries: list[dict] = []
        errors: list[str] = []

        async with self._engine.begin() as conn:
            run_id = (
                await conn.execute(
                    sqlite_insert(PipelineRun).values(
                        pipeline="portfolio_pipeline",
                        started_at=started,
                        status="running",
                    )
                )
            ).inserted_primary_key[0]

            accounts = await self._accounts.list_accounts()
            logger.info("portfolio_pipeline: {} account(s) returned", len(accounts))

            for account in accounts:
                account_id = account.get("account_id") or ""
                account_id_key = account.get("account_id_key")
                if not account_id_key:
                    logger.warning("Skipping account {} (no accountIdKey)", account_id)
                    continue

                try:
                    async with conn.begin_nested():
                        raw = await self._accounts.get_positions(account_id_key)
                        normalized = self._normalize(account_id, raw)
                        if normalized:
                            await self._upsert(conn, normalized)
                        positions_stored += len(normalized)
                        accounts_processed += 1

                        total_mv = sum(p["market_value"] for p in normalized)
                        tickers = [p["ticker"] for p in normalized]
                        account_summaries.append({
                            "account_id": account_id,
                            "tickers": tickers,
                            "position_count": len(normalized),
                            "total_market_value": total_mv,
                        })
                        logger.info(
                            "Account {}: {} positions, ${:,.0f} market value",
                            account_id,
                            len(normalized),
                            total_mv,
                        )
                except Exception as exc:
                    logger.exception("portfolio_pipeline failed for account {}", account_id)
                    errors.append(f"{account_id}: {exc}")

            completed = datetime.now(timezone.utc)
            if errors and accounts_processed == 0:
                status = "failed"
            elif errors:
                status = "partial"
            else:
                status = "ok"

            await conn.execute(
                update(PipelineRun)
                .where(PipelineRun.id == run_id)
                .values(
                    completed_at=completed,
                    status=status,
                    tickers_processed=positions_stored,
                    errors="\n".join(errors) if errors else None,
                )
            )

        result = PortfolioRunResult(
            started_at=started,
            completed_at=completed,
            status=status,
            accounts_processed=accounts_processed,
            positions_stored=positions_stored,
            accounts=account_summaries,
            errors=errors,
        )
        logger.info(
            "portfolio_pipeline done: status={} accounts={} positions={} errors={}",
            result.status,
            result.accounts_processed,
            result.positions_stored,
            len(result.errors),
        )
        return result

    def _normalize(self, account_id: str, positions: list[dict]) -> list[dict]:
        now = datetime.now(timezone.utc)
        rows: list[dict] = []
        for p in positions:
            ticker = p.get("ticker")
            if not ticker:
                logger.warning("position missing ticker for account {}", account_id)
                continue
            rows.append(
                {
                    "account_id": account_id,
                    "ticker": ticker.strip().upper(),
                    "quantity": float(p.get("quantity", 0.0)),
                    "cost_basis": float(p.get("costBasis", 0.0)),
                    "market_value": float(p.get("marketValue", 0.0)),
                    "last_updated": now,
                }
            )
        return rows

    async def _upsert(self, conn, rows: list[dict]) -> None:
        stmt = sqlite_insert(Position).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["account_id", "ticker"],
            set_={
                "quantity": stmt.excluded.quantity,
                "cost_basis": stmt.excluded.cost_basis,
                "market_value": stmt.excluded.market_value,
                "last_updated": stmt.excluded.last_updated,
            },
        )
        await conn.execute(stmt)
