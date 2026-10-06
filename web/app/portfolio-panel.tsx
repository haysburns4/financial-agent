"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { dashboardSchema, proxyErrorSchema, type Dashboard, type Position } from "./portfolio";

// Stored positions change at most every PORTFOLIO_POLL_MINUTES (15) on the
// server; polling faster than that only re-reads the same rows.
const POLL_MS = 60_000;

type SortKey = "ticker" | "market_value" | "pnl" | "pnl_pct" | "weight";
type Row = Position & { price: number | null; weight: number };

const usd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });
const usdWhole = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});
const pct = new Intl.NumberFormat("en-US", {
  style: "percent",
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});
const qty = new Intl.NumberFormat("en-US", { maximumFractionDigits: 4 });

function signed(value: number, format: Intl.NumberFormat): string {
  return value > 0 ? `+${format.format(value)}` : format.format(value);
}

function tone(value: number | null): string {
  if (value === null || value === 0) return "";
  return value > 0 ? "gain" : "loss";
}

/** Show only the last four digits of an account number on screen. */
function maskAccount(id: string): string {
  return `••${id.slice(-4)}`;
}

function ago(iso: string, now: number): string {
  const minutes = Math.max(0, Math.round((now - Date.parse(iso)) / 60_000));
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  return hours < 24 ? `${hours}h ago` : `${Math.floor(hours / 24)}d ago`;
}

async function request(method: "GET" | "POST"): Promise<Dashboard> {
  const response = await fetch("/api/portfolio", { method, cache: "no-store" });
  const body = await response.json();
  if (!response.ok) {
    const failure = proxyErrorSchema.safeParse(body);
    throw new Error(failure.success ? failure.data.error : `HTTP ${response.status}`);
  }
  return dashboardSchema.parse(body);
}

export function PortfolioPanel() {
  const [data, setData] = useState<Dashboard | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [account, setAccount] = useState<string | null>(null);
  const [sort, setSort] = useState<{ key: SortKey; descending: boolean }>({
    key: "market_value",
    descending: true,
  });
  const [now, setNow] = useState(() => Date.now());

  const load = useCallback(async (method: "GET" | "POST") => {
    try {
      setData(await request(method));
      setError(null);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setNow(Date.now());
    }
  }, []);

  useEffect(() => {
    void load("GET");
    const timer = setInterval(() => void load("GET"), POLL_MS);
    return () => clearInterval(timer);
  }, [load]);

  async function refresh() {
    setRefreshing(true);
    await load("POST");
    setRefreshing(false);
  }

  const selected = data?.accounts.find((a) => a.account_id === account) ?? null;
  const view = selected ?? data?.totals ?? null;

  const rows = useMemo<Row[]>(() => {
    if (!data) return [];
    const positions = selected ? selected.positions : data.accounts.flatMap((a) => a.positions);
    const total = positions.reduce((sum, p) => sum + p.market_value, 0);
    const withDerived = positions.map((p) => ({
      ...p,
      price: p.quantity !== 0 ? p.market_value / p.quantity : null,
      weight: total ? p.market_value / total : 0,
    }));
    const direction = sort.descending ? -1 : 1;
    return withDerived.sort((a, b) => {
      if (sort.key === "ticker") return direction * a.ticker.localeCompare(b.ticker);
      return direction * ((a[sort.key] ?? -Infinity) - (b[sort.key] ?? -Infinity));
    });
  }, [data, selected, sort]);

  function header(key: SortKey, label: string, numeric = true) {
    const active = sort.key === key;
    return (
      <th className={numeric ? "num" : ""} aria-sort={active ? (sort.descending ? "descending" : "ascending") : "none"}>
        <button
          type="button"
          onClick={() =>
            setSort({ key, descending: active ? !sort.descending : key !== "ticker" })
          }
        >
          {label}
          <span className="caret">{active ? (sort.descending ? "▾" : "▴") : ""}</span>
        </button>
      </th>
    );
  }

  return (
    <section className="portfolio" aria-label="Portfolio">
      <div className="portfolio-head">
        <div>
          <h2>Portfolio</h2>
          <p className="meta">
            {data?.last_updated ? `Positions as of ${ago(data.last_updated, now)}` : "No positions stored yet"}
            {data && (
              <span className={data.authenticated ? "pill ok" : "pill warn"}>
                {data.authenticated ? "E-Trade connected" : "E-Trade logged out"}
              </span>
            )}
          </p>
        </div>
        <button
          type="button"
          className="refresh"
          onClick={() => void refresh()}
          disabled={refreshing || data?.authenticated === false}
          title={data?.authenticated === false ? "Log in to E-Trade to refresh" : "Pull fresh positions from E-Trade"}
        >
          {refreshing ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      {error && (
        <div className="panel-error" role="alert">
          {error}
        </div>
      )}

      {!data && !error && <p className="placeholder">Loading portfolio…</p>}

      {data && view && (
        <>
          <div className="tiles">
            <div className="tile">
              <span className="label">Market value</span>
              <span className="value">{usd.format(view.market_value)}</span>
            </div>
            <div className="tile">
              <span className="label">Unrealized P&amp;L</span>
              <span className={`value ${tone(view.pnl)}`}>{signed(view.pnl, usd)}</span>
              {view.pnl_pct !== null && (
                <span className={`sub ${tone(view.pnl_pct)}`}>{signed(view.pnl_pct, pct)}</span>
              )}
            </div>
            <div className="tile">
              <span className="label">Cost basis</span>
              <span className="value">{usd.format(view.cost_basis)}</span>
            </div>
            <div className="tile">
              <span className="label">Positions</span>
              <span className="value">{view.position_count}</span>
              {!selected && (
                <span className="sub">
                  across {data.totals.account_count} account{data.totals.account_count === 1 ? "" : "s"}
                </span>
              )}
            </div>
          </div>

          {data.accounts.length > 1 && (
            <div className="tabs" role="tablist" aria-label="Accounts">
              <button
                type="button"
                role="tab"
                aria-selected={selected === null}
                onClick={() => setAccount(null)}
              >
                All accounts
                <span className="tab-value">{usdWhole.format(data.totals.market_value)}</span>
              </button>
              {data.accounts.map((a) => (
                <button
                  key={a.account_id}
                  type="button"
                  role="tab"
                  aria-selected={selected?.account_id === a.account_id}
                  onClick={() => setAccount(a.account_id)}
                  title={`Account ${a.account_id}`}
                >
                  {maskAccount(a.account_id)}
                  <span className="tab-value">{usdWhole.format(a.market_value)}</span>
                </button>
              ))}
            </div>
          )}

          {rows.length === 0 ? (
            <p className="placeholder">
              No positions yet. Log in to E-Trade, then press Refresh.
            </p>
          ) : (
            <div className="table-wrap">
              <table className="positions">
                <thead>
                  <tr>
                    {header("ticker", "Ticker", false)}
                    {!selected && <th>Acct</th>}
                    <th className="num">Qty</th>
                    <th className="num">Price</th>
                    {header("market_value", "Value")}
                    {header("pnl", "P&L")}
                    {header("pnl_pct", "P&L %")}
                    {header("weight", "Weight")}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((p) => (
                    <tr key={`${p.account_id}:${p.ticker}`}>
                      <td className="ticker">{p.ticker}</td>
                      {!selected && <td className="muted">{maskAccount(p.account_id)}</td>}
                      <td className="num">{qty.format(p.quantity)}</td>
                      <td className="num">{p.price === null ? "—" : usd.format(p.price)}</td>
                      <td className="num">{usd.format(p.market_value)}</td>
                      <td className={`num ${tone(p.pnl)}`}>{signed(p.pnl, usd)}</td>
                      <td className={`num ${tone(p.pnl_pct)}`}>
                        {p.pnl_pct === null ? "—" : signed(p.pnl_pct, pct)}
                      </td>
                      <td className="num weight">
                        <span className="bar" style={{ width: `${Math.min(100, p.weight * 100)}%` }} />
                        <span>{pct.format(p.weight)}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </section>
  );
}
