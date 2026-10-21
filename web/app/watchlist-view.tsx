"use client";

import { useCallback, useEffect, useId, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import type { z } from "zod";

import { maskAccount, proxyErrorSchema } from "./portfolio";
import {
  removedSchema,
  watchlistEntrySchema,
  watchlistViewSchema,
  type AddRequest,
  type BackfillStatus,
  type MonitoredTicker,
  type PatchRequest,
  type WatchlistEntry,
} from "./watchlist";

// While a backfill is loading, poll fast enough to see it finish; otherwise
// slowly, to notice positions opened or closed at E-Trade.
const LOADING_POLL_MS = 5_000;
const IDLE_POLL_MS = 60_000;
const HELD_ONLY_TOOLTIP = "Monitored because you hold it";

/** One line of the panel: a monitored ticker, with its watchlist row if it has one. */
type Row = MonitoredTicker & {
  entry: WatchlistEntry | null;
  /** Shown before the server has confirmed it. */
  provisional: boolean;
};

type Editing = { ticker: string; field: "note" | "tags"; draft: string } | null;

async function call<T>(url: string, method: string, schema: z.ZodType<T>, body?: AddRequest | PatchRequest): Promise<T> {
  const init: RequestInit = { method, cache: "no-store" };
  if (body !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(body);
  }
  const response = await fetch(url, init);
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const failure = proxyErrorSchema.safeParse(payload);
    throw new Error(failure.success ? failure.data.error : `HTTP ${response.status}`);
  }
  return schema.parse(payload);
}

function rowsFrom(monitored: MonitoredTicker[], entries: WatchlistEntry[]): Row[] {
  const byTicker = new Map(entries.map((e) => [e.ticker, e]));
  return monitored.map((m) => ({ ...m, entry: byTicker.get(m.ticker) ?? null, provisional: false }));
}

function parseTags(text: string): string[] {
  return [...new Set(text.split(",").map((t) => t.trim()).filter(Boolean))];
}

const SUGGESTIONS = 8;

/** The tag being typed: whatever follows the last comma. */
function partialTag(draft: string): string {
  return draft.slice(draft.lastIndexOf(",") + 1).trim();
}

/** `draft` with the tag being typed replaced by `tag`, ready for the next one. */
function completeTag(draft: string, tag: string): string {
  const cut = draft.lastIndexOf(",");
  const head = cut < 0 ? "" : `${draft.slice(0, cut + 1)} `;
  return `${head}${tag}, `;
}

/** Tags already in use that fit what is being typed and are not on this row yet. */
function suggestTags(known: string[], draft: string): string[] {
  const chosen = new Set(parseTags(draft).map((t) => t.toLowerCase()));
  const typing = partialTag(draft).toLowerCase();
  // The tag being typed is not chosen yet: an exact match should still be offered.
  if (typing) chosen.delete(typing);
  return known
    .filter((t) => !chosen.has(t.toLowerCase()) && t.toLowerCase().includes(typing))
    .sort((a, b) => Number(!a.toLowerCase().startsWith(typing)) - Number(!b.toLowerCase().startsWith(typing)))
    .slice(0, SUGGESTIONS);
}

function loading(status: BackfillStatus | undefined): boolean {
  return status === "pending" || status === "running";
}

function message(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

/** The Watchlist tab of the portfolio panel: every monitored ticker, editable. */
export function WatchlistView() {
  const [rows, setRows] = useState<Row[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [addError, setAddError] = useState<string | null>(null);
  const [ticker, setTicker] = useState("");
  const [editing, setEditing] = useState<Editing>(null);
  // Held-only tickers (monitored only because you hold them) are hidden until
  // asked for. A held ticker you also put on the watchlist always shows.
  const [showHoldings, setShowHoldings] = useState(false);
  const inputId = useId();
  const errorId = useId();
  // Bumped by every optimistic change, so a poll that started before it
  // cannot overwrite it with stale rows.
  const generation = useRef(0);

  const load = useCallback(async () => {
    const started = generation.current;
    try {
      const view = await call("/api/watchlist", "GET", watchlistViewSchema);
      if (generation.current === started) setRows(rowsFrom(view.monitored, view.entries));
      setError(null);
    } catch (exc) {
      setError(message(exc));
    }
  }, []);

  const anyLoading = rows?.some((r) => loading(r.entry?.backfill_status)) ?? false;
  useEffect(() => {
    void load();
    const timer = setInterval(() => void load(), anyLoading ? LOADING_POLL_MS : IDLE_POLL_MS);
    return () => clearInterval(timer);
  }, [load, anyLoading]);

  const heldOnly = rows?.filter((r) => r.source === "held").length ?? 0;
  const knownTags = [...new Set(rows?.flatMap((r) => r.entry?.tags ?? []) ?? [])].sort((a, b) => a.localeCompare(b));
  const visible = rows?.filter((r) => showHoldings || r.source !== "held") ?? null;

  /** Replace one ticker's row; null removes it. Other rows are left alone, so
   * rolling back one change never undoes another in flight. */
  function swap(symbol: string, next: Row | null) {
    generation.current += 1;
    setRows((current) => {
      if (!current) return current;
      const others = current.filter((r) => r.ticker !== symbol);
      return next ? [...others, next].sort((a, b) => a.ticker.localeCompare(b.ticker)) : others;
    });
  }

  /** Apply `optimistic` now, run `request`, and put the old row back if it fails. */
  async function mutate(symbol: string, optimistic: Row | null, request: () => Promise<void>): Promise<boolean> {
    const before = rows?.find((r) => r.ticker === symbol) ?? null;
    swap(symbol, optimistic);
    try {
      await request();
      return true;
    } catch (exc) {
      swap(symbol, before);
      setError(message(exc));
      return false;
    }
  }

  async function add(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const symbol = ticker.trim().toUpperCase();
    if (!symbol) return;
    setAddError(null);
    const existing = rows?.find((r) => r.ticker === symbol) ?? null;
    if (existing?.entry) {
      setAddError(`${symbol} is already on the watchlist`);
      return;
    }
    const provisional: Row = {
      ticker: symbol,
      source: existing ? "both" : "watchlist",
      note: null,
      tags: [],
      accounts: existing?.accounts ?? [],
      indicators_ready: existing?.indicators_ready ?? false,
      entry: null,
      provisional: true,
    };
    swap(symbol, provisional);
    setTicker("");
    try {
      const entry = await call("/api/watchlist", "POST", watchlistEntrySchema, { ticker: symbol });
      swap(symbol, { ...provisional, note: entry.note, tags: entry.tags, entry, provisional: false });
      void load();
    } catch (exc) {
      // Inline, next to the input: unknown symbol, watchlist full, bad format.
      swap(symbol, existing);
      setTicker(symbol);
      setAddError(message(exc));
    }
  }

  async function remove(row: Row) {
    const stays = row.source === "both";
    const optimistic: Row | null = stays
      ? { ...row, source: "held", note: null, tags: [], entry: null }
      : null;
    await mutate(row.ticker, optimistic, async () => {
      await call(`/api/watchlist/${encodeURIComponent(row.ticker)}`, "DELETE", removedSchema);
    });
  }

  async function save(row: Row, field: "note" | "tags", draft: string) {
    setEditing(null);
    const entry = row.entry;
    if (!entry) return;
    const patch: PatchRequest = {};
    const next: Row = { ...row, entry: { ...entry } };
    if (field === "note") {
      const note = draft.trim() || null;
      if (note === entry.note) return;
      patch.note = note;
      next.note = note;
      next.entry = { ...entry, note };
    } else {
      // Reuse an existing tag's spelling rather than start a near-duplicate.
      const tags = [
        ...new Set(parseTags(draft).map((t) => knownTags.find((k) => k.toLowerCase() === t.toLowerCase()) ?? t)),
      ];
      if (tags.join(",") === entry.tags.join(",")) return;
      patch.tags = tags;
      next.tags = tags;
      next.entry = { ...entry, tags };
    }
    await mutate(row.ticker, next, async () => {
      const saved = await call(`/api/watchlist/${encodeURIComponent(row.ticker)}`, "PATCH", watchlistEntrySchema, patch);
      swap(row.ticker, { ...next, note: saved.note, tags: saved.tags, entry: saved });
    });
  }

  async function retry(row: Row) {
    const entry = row.entry;
    if (!entry) return;
    const next: Row = { ...row, entry: { ...entry, backfill_status: "pending", backfill_error: null } };
    await mutate(row.ticker, next, async () => {
      await call(`/api/watchlist/${encodeURIComponent(row.ticker)}/backfill`, "POST", watchlistEntrySchema);
    });
  }

  function editKeys(event: KeyboardEvent<HTMLInputElement>, row: Row) {
    if (event.key === "Escape") {
      event.preventDefault();
      setEditing(null);
    } else if (event.key === "Enter" && editing) {
      event.preventDefault();
      void save(row, editing.field, editing.draft);
    } else if (event.key === "Tab" && editing?.field === "tags" && partialTag(editing.draft)) {
      // Tab completes the best match for the tag being typed.
      const [best] = suggestTags(knownTags, editing.draft);
      if (best) {
        event.preventDefault();
        setEditing({ ...editing, draft: completeTag(editing.draft, best) });
      }
    }
  }

  function editable(row: Row, field: "note" | "tags") {
    const entry = row.entry;
    if (!entry) return <span className="muted">—</span>;
    if (editing?.ticker === row.ticker && editing.field === field) {
      const input = (
        <input
          className="inline-edit"
          value={editing.draft}
          onChange={(event) => setEditing({ ...editing, draft: event.target.value })}
          onKeyDown={(event) => editKeys(event, row)}
          onBlur={() => void save(row, field, editing.draft)}
          placeholder={field === "tags" ? "comma, separated" : undefined}
          aria-label={`${field === "note" ? "Note" : "Tags"} for ${row.ticker}`}
          autoFocus
        />
      );
      if (field === "note") return input;
      const suggestions = suggestTags(knownTags, editing.draft);
      return (
        <div className="tag-editor">
          {input}
          {suggestions.length > 0 && (
            <div className="tag-suggestions" role="group" aria-label="Existing tags">
              {suggestions.map((t) => (
                <button
                  key={t}
                  type="button"
                  className="tag suggestion"
                  // Keep focus in the input: a blur there would save and close the editor.
                  onMouseDown={(event) => event.preventDefault()}
                  onClick={() => setEditing({ ...editing, draft: completeTag(editing.draft, t) })}
                  aria-label={`Add tag ${t}`}
                >
                  {t}
                </button>
              ))}
            </div>
          )}
        </div>
      );
    }
    const draft = field === "note" ? entry.note ?? "" : entry.tags.join(", ");
    return (
      <button
        type="button"
        className="inline-value"
        onClick={() => setEditing({ ticker: row.ticker, field, draft })}
        title={`Edit ${field}`}
      >
        {field === "note" ? (
          entry.note ?? <span className="muted">+</span>
        ) : entry.tags.length ? (
          entry.tags.map((t) => (
            <span key={t} className="tag">
              {t}
            </span>
          ))
        ) : (
          <span className="muted">+</span>
        )}
      </button>
    );
  }

  function history(row: Row) {
    const entry = row.entry;
    if (row.provisional || loading(entry?.backfill_status)) {
      return <span className="pill loading">history loading</span>;
    }
    if (entry?.backfill_status === "failed") {
      return (
        <span className="backfill-failed">
          <span className="pill bad" title={entry.backfill_error ?? undefined}>
            history failed
          </span>
          <button type="button" className="link" onClick={() => void retry(row)}>
            Retry
          </button>
        </span>
      );
    }
    if (!row.indicators_ready) return <span className="muted">no history yet</span>;
    return <span className="muted">ready</span>;
  }

  return (
    <div className="watchlist" role="tabpanel" aria-label="Watchlist">
      <form className="watchlist-add" onSubmit={(event) => void add(event)}>
        <label htmlFor={inputId}>Add a ticker</label>
        <div className="row">
          <input
            id={inputId}
            value={ticker}
            onChange={(event) => {
              setTicker(event.target.value);
              setAddError(null);
            }}
            placeholder="e.g. NVDA"
            maxLength={10}
            spellCheck={false}
            autoComplete="off"
            aria-invalid={addError !== null}
            aria-describedby={addError ? errorId : undefined}
          />
          <button type="submit" className="refresh" disabled={!ticker.trim()}>
            Add
          </button>
          <button
            type="button"
            className="refresh holdings-toggle"
            aria-pressed={showHoldings}
            onClick={() => setShowHoldings(!showHoldings)}
            disabled={heldOnly === 0 && !showHoldings}
            title={
              heldOnly === 0
                ? "Everything you hold is already on the watchlist"
                : showHoldings
                  ? "Hide tickers that are here only because you hold them"
                  : "Show the tickers you hold that aren't on the watchlist"
            }
          >
            {showHoldings ? "Hide Holdings" : "Show Holdings"}
            {heldOnly > 0 && <span className="tab-value">{heldOnly}</span>}
          </button>
        </div>
        {addError && (
          <p id={errorId} className="field-error" role="alert">
            {addError}
          </p>
        )}
      </form>

      {error && (
        <div className="panel-error" role="alert">
          {error}
        </div>
      )}

      {rows === null && !error && <p className="placeholder">Loading watchlist…</p>}
      {visible?.length === 0 && (
        <p className="placeholder">
          {heldOnly > 0
            ? "Nothing on your watchlist yet. Add a ticker above, or press Add Holdings to see what you hold."
            : "Nothing monitored yet. Add a ticker above."}
        </p>
      )}

      {visible && visible.length > 0 && (
        <div className="table-wrap">
          <table className="positions watchlist-table">
            <thead>
              <tr>
                <th>Ticker</th>
                <th>Source</th>
                <th>Note</th>
                <th>Tags</th>
                <th>History</th>
                <th aria-label="Actions" />
              </tr>
            </thead>
            <tbody>
              {visible.map((row) => (
                <tr key={row.ticker} className={row.provisional ? "provisional" : ""}>
                  <td className="ticker">
                    {row.ticker}
                    {row.entry && !row.entry.validated && (
                      <span className="unverified" title="Not yet confirmed by an E-Trade quote">
                        ?
                      </span>
                    )}
                  </td>
                  <td>
                    <span className="sources">
                      {row.source !== "held" && <span className="pill source-watchlist">watchlist</span>}
                      {row.accounts.map((id) => (
                        <span key={id} className="pill source-account" title={`Held in account ${id}`}>
                          {maskAccount(id)}
                        </span>
                      ))}
                    </span>
                  </td>
                  <td className="note-cell">{editable(row, "note")}</td>
                  <td>{editable(row, "tags")}</td>
                  <td>{history(row)}</td>
                  <td className="num">
                    {row.source === "held" ? (
                      // A disabled button gets no hover events, so the tooltip sits on its wrapper.
                      <span title={HELD_ONLY_TOOLTIP}>
                        <button type="button" className="link" disabled aria-label={`${row.ticker}: ${HELD_ONLY_TOOLTIP}`}>
                          Remove
                        </button>
                      </span>
                    ) : (
                      <button
                        type="button"
                        className="link"
                        disabled={row.provisional}
                        onClick={() => void remove(row)}
                        title={row.source === "both" ? "Removes it from the watchlist; it stays monitored while you hold it" : undefined}
                      >
                        Remove
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
