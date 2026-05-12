"""Technical indicators via pandas-ta."""
import pandas as pd
import pandas_ta as ta
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncConnection

from src.models import PriceBar


async def load_price_frame(conn: AsyncConnection, ticker: str, limit: int = 200) -> pd.DataFrame:
    stmt = (
        select(PriceBar)
        .where(PriceBar.ticker == ticker)
        .order_by(PriceBar.timestamp.desc())
        .limit(limit)
    )
    rows = (await conn.execute(stmt)).all()
    rows = list(reversed(rows))
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "timestamp": [r.timestamp for r in rows],
            "open": [r.open for r in rows],
            "high": [r.high for r in rows],
            "low": [r.low for r in rows],
            "close": [r.close for r in rows],
            "volume": [r.volume for r in rows],
        }
    ).set_index("timestamp")
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with rsi_14, macd_*, ema_9, ema_21 columns."""
    if df.empty:
        return df

    out = df.copy()
    out["rsi_14"] = ta.rsi(out["close"], length=14)

    macd = ta.macd(out["close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        out["macd_line"] = macd.iloc[:, 0]
        out["macd_signal"] = macd.iloc[:, 2]
        out["macd_hist"] = macd.iloc[:, 1]

    out["ema_9"] = ta.ema(out["close"], length=9)
    out["ema_21"] = ta.ema(out["close"], length=21)
    return out


def ema_cross(df: pd.DataFrame) -> str | None:
    """Detect a 9/21 EMA cross on the last two bars. Returns 'bullish', 'bearish', or None."""
    if len(df) < 2 or "ema_9" not in df.columns or "ema_21" not in df.columns:
        return None
    prev = df.iloc[-2]
    last = df.iloc[-1]
    if pd.isna(prev.ema_9) or pd.isna(prev.ema_21):
        return None
    if prev.ema_9 <= prev.ema_21 and last.ema_9 > last.ema_21:
        return "bullish"
    if prev.ema_9 >= prev.ema_21 and last.ema_9 < last.ema_21:
        return "bearish"
    return None
