# Watchlist

The tickers you follow are stored in the database (the `watchlist` table), and you
change them from the dashboard while the app runs. A fresh install starts with an
empty watchlist.

**Monitored** tickers are your watchlist plus every ticker you hold with a nonzero
quantity. `get_monitored` in `src/watchlist/resolver.py` computes this list each
time it's called and never stores it. The price pipeline, the signal engine, the
backfill and backtest defaults, and the agent all use it, so a ticker you add is
picked up on the next price run.

## API

```
GET    /watchlist                      active rows
GET    /watchlist/monitored            resolver output: source, accounts, note, tags, indicators_ready
GET    /watchlist/events?ticker=&limit=
POST   /watchlist                      {"ticker": "NVDA", "note": "...", "tags": ["semis"]}
PATCH  /watchlist/{ticker}             {"note": ..., "tags": [...]}  (omitted fields are left alone)
DELETE /watchlist/{ticker}             soft delete; the response says whether it is still held
POST   /watchlist/{ticker}/backfill    retry a failed history backfill
```

- **Validation:** a ticker must be 1–10 characters (A–Z, `.`, `-`). E-Trade must
  also return a quote for it, through a read-only lookup. If E-Trade is
  unavailable, the add is accepted unvalidated and the next price run confirms it.
- **Size cap:** `WATCHLIST_MAX_SIZE` (default 50).
- **Removal is a soft delete.** Price bars, indicators and signals are kept. Adding
  the ticker again restores its note and tags.
- **History:** every change is recorded in `watchlist_events`.
- **Health:** `GET /health` reports the watchlist size, the monitored count, and
  any backfills that are pending, running or failed.

## Backfill on add

Adding a ticker starts a background yfinance backfill for that ticker only. The
add request doesn't wait for it. Until the backfill is done, the signal engine
skips entry/exit rules for that ticker, though risk rules on held positions still
run. If the backfill fails, the row shows the error and a Retry button. Backfills
that were still pending when the app stopped resume at startup.

## The agent

The agent reads the watchlist but can't change it: no tool or CopilotKit action
adds or removes tickers, and `tests/test_watchlist.py` checks this. Notes and tags
go into the chat and synthesis context, so a note like "waiting on earnings"
shapes how the agent frames a signal on that ticker.

### `Scope(account, watchlist_only)`

Adding this note because it introduced a bit of complexity that didn't sit well with me, 
and there's probably a more elegant scheme for delineating a watchlist from an account.
The dashboard sends two context values with every chat run: `highlighted_account`
(the selected account, or `all`) and `dashboard_tab` (`watchlist` or `accounts`).
`src/agui.py` reads both. The chat bundles them into a `Scope(account, watchlist_only)` 
from `src/agent/tools.py` and passes it to every tool call, and
each handler applies the part that concerns it:

- Positions and risk are filtered by `scope.account`.
- On the Watchlist tab (`scope.watchlist_only`), `get_signals` and
  `screen_indicators` cover only watchlist tickers, and the system prompt says so.

The model never sets the scope. It comes only from the dashboard. An `account_id`
argument from the model can still widen account tools to another account or `all`,
but nothing the model sends can turn off the watchlist-only filter.
