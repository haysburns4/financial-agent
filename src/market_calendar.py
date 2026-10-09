"""The NYSE trading calendar: which days trade, and what the market is doing now."""
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Literal
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal

MarketState = Literal["pre_open", "intraday", "after_close", "weekend", "holiday"]

NYSE_TZ = ZoneInfo("America/New_York")
_LOOKBACK = timedelta(days=14)


@lru_cache(maxsize=1)
def _nyse() -> mcal.MarketCalendar:
    return mcal.get_calendar("XNYS")


def trading_days(start: date, end: date) -> list[date]:
    """NYSE trading days from `start` to `end`, inclusive."""
    if end < start:
        return []
    return [d.date() for d in _nyse().valid_days(start_date=start, end_date=end)]


def market_moment(moment: datetime) -> tuple[date, MarketState]:
    """The trading day `moment` belongs to, and what the market is doing."""
    local = moment.astimezone(NYSE_TZ)
    today = local.date()
    schedule = _nyse().schedule(start_date=today - _LOOKBACK, end_date=today)
    days = [d.date() for d in schedule.index]

    if days and days[-1] == today:
        session = schedule.iloc[-1]
        if local < session["market_open"]:
            return today, "pre_open"
        if local < session["market_close"]:
            return today, "intraday"
        return today, "after_close"

    previous = days[-1] if days else today - timedelta(days=1)
    return previous, "weekend" if today.weekday() >= 5 else "holiday"
