"""Portfolio shaping shared by the HTTP layer and the agent's tools.

Kept out of `src/server.py` so `src/agent/tools.py` can reuse the same P&L and
risk math instead of reimplementing it.
"""


def position_dict(row) -> dict:
    pnl = row.market_value - row.cost_basis
    return {
        "account_id": row.account_id,
        "ticker": row.ticker,
        "quantity": row.quantity,
        "cost_basis": row.cost_basis,
        "market_value": row.market_value,
        "pnl": pnl,
        "pnl_pct": (pnl / row.cost_basis) if row.cost_basis else None,
        "last_updated": row.last_updated.isoformat(),
    }


def risk_summary(positions: list) -> dict:
    """Risk metrics for an arbitrary set of positions (one account or combined)."""
    total_exposure = sum(p.market_value for p in positions)
    total_cost = sum(p.cost_basis for p in positions)
    concentration = sorted(
        (
            {
                "ticker": p.ticker,
                "market_value": p.market_value,
                "pct_of_portfolio": (p.market_value / total_exposure) if total_exposure else 0.0,
            }
            for p in positions
        ),
        key=lambda x: x["pct_of_portfolio"],
        reverse=True,
    )
    # Proxy for true peak-to-trough drawdown until we persist portfolio snapshots.
    drawdown = (total_cost - total_exposure) / total_cost if total_cost > 0 else None
    return {
        "total_exposure": total_exposure,
        "concentration": concentration,
        "drawdown": drawdown,
        "position_count": len(positions),
    }


def risk_by_account(rows: list) -> dict:
    """Combined risk plus a per-account breakdown."""
    by_account: dict[str, list] = {}
    for row in rows:
        by_account.setdefault(row.account_id, []).append(row)
    return {
        "combined": risk_summary(rows),
        "by_account": {acct: risk_summary(ps) for acct, ps in sorted(by_account.items())},
    }


def _totals(positions: list[dict]) -> dict:
    market_value = sum(p["market_value"] for p in positions)
    cost_basis = sum(p["cost_basis"] for p in positions)
    pnl = market_value - cost_basis
    return {
        "market_value": market_value,
        "cost_basis": cost_basis,
        "pnl": pnl,
        "pnl_pct": (pnl / cost_basis) if cost_basis else None,
        "position_count": len(positions),
    }


def dashboard(rows: list) -> dict:
    """Everything the web dashboard's portfolio panel shows, in one payload.

    Accounts are a list rather than a dict keyed by id so the frontend gets a
    fixed contract; positions within each are largest first.
    """
    positions = [position_dict(r) for r in rows]
    by_account: dict[str, list[dict]] = {}
    for p in sorted(positions, key=lambda p: p["market_value"], reverse=True):
        by_account.setdefault(p["account_id"], []).append(p)

    accounts = [
        {"account_id": account_id, **_totals(held), "positions": held}
        for account_id, held in sorted(by_account.items())
    ]

    return {
        "totals": {**_totals(positions), "account_count": len(accounts)},
        "accounts": accounts,
        "last_updated": max((r.last_updated for r in rows), default=None),
    }
