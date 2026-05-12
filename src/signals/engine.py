"""Signal aggregation: turn indicator state into Signal rows."""
from datetime import datetime, timezone

from loguru import logger
from sqlalchemy import insert

from src.config import settings
from src.db import get_connection
from src.models import Signal
from src.signals.technical import compute_indicators, ema_cross, load_price_frame


async def evaluate_watchlist() -> int:
    """Run signal rules across the watchlist. Returns count of signals persisted."""
    created = 0
    async with get_connection() as conn:
        for ticker in settings.WATCHLIST:
            df = await load_price_frame(conn, ticker)
            if df.empty or len(df) < 30:
                continue

            df = compute_indicators(df)
            last = df.iloc[-1]
            cross = ema_cross(df)

            rules: list[tuple[str, str | None, float, str]] = []

            if cross == "bullish":
                rules.append(("entry", "long", 0.6, "9/21 EMA bullish cross"))
            if cross == "bearish":
                rules.append(("exit", None, 0.6, "9/21 EMA bearish cross"))
            if "rsi_14" in df.columns and not df["rsi_14"].isna().all():
                rsi = float(last.rsi_14)
                if rsi < 30:
                    rules.append(("entry", "long", 0.55, f"RSI oversold ({rsi:.1f})"))
                if rsi > 70:
                    rules.append(("risk", None, 0.55, f"RSI overbought ({rsi:.1f})"))

            if not rules:
                continue

            await conn.execute(
                insert(Signal),
                [
                    {
                        "ticker": ticker,
                        "timestamp": last.name if isinstance(last.name, datetime) else datetime.now(timezone.utc),
                        "signal_type": signal_type,
                        "direction": direction,
                        "confidence": confidence,
                        "reasoning": reasoning,
                        "delivered": False,
                    }
                    for signal_type, direction, confidence, reasoning in rules
                ],
            )
            created += len(rules)

    logger.info("signal engine: {} signals generated", created)
    return created
