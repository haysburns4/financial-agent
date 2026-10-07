"use client";

import { defineToolCallRenderer } from "@copilotkit/react-core/v2";
import type { ReactNode } from "react";
import { z } from "zod";

/**
 * Renders the agent's backend tool calls in the transcript as one quiet line —
 * what was looked up and the gist of it — that expands into a small table.
 *
 * Results arrive as the JSON strings src/agent/tools.py produces, so each one
 * is parsed with a schema mirroring its handler. A tool error arrives as plain
 * text instead, and anything unparseable is shown as-is rather than hidden.
 */

const usd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
const pct = new Intl.NumberFormat("en-US", { style: "percent", minimumFractionDigits: 1, maximumFractionDigits: 1 });
const num = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });

const ROWS = 8;

function parseJson<T>(raw: string, schema: z.ZodType<T>): T | null {
  try {
    const parsed = schema.safeParse(JSON.parse(raw));
    return parsed.success ? parsed.data : null;
  } catch {
    return null;
  }
}

function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

type CardProps = {
  label: string;
  /** Undefined while the tool is still running. */
  result: string | undefined;
  summary: string | null;
  children?: ReactNode;
};

function ToolCard({ label, result, summary, children }: CardProps) {
  const running = result === undefined;
  const unreadable = !running && summary === null;
  const state = running ? "running" : unreadable ? "error" : "done";

  const head = (
    <>
      <span className={`tool-state ${state}`} aria-hidden="true" />
      <span className="tool-label">{running ? `${label}…` : label}</span>
      {summary && <span className="tool-summary">{summary}</span>}
    </>
  );

  if (running) {
    return (
      <div className="tool" role="status">
        <div className="tool-head">{head}</div>
      </div>
    );
  }
  return (
    <details className="tool">
      <summary className="tool-head">{head}</summary>
      <div className="tool-body">{unreadable ? <pre className="tool-raw">{result}</pre> : children}</div>
    </details>
  );
}

// ---------- get_positions ----------

const positionsResult = z.array(
  z.object({
    account_id: z.string(),
    ticker: z.string(),
    quantity: z.number(),
    market_value: z.number(),
    pnl_pct: z.number().nullable(),
  }),
);

const positions = defineToolCallRenderer({
  name: "get_positions",
  args: z.object({ account_id: z.string().optional() }),
  render: function Positions({ result }) {
    const rows = result === undefined ? null : parseJson(result, positionsResult);
    const total = rows?.reduce((sum, p) => sum + p.market_value, 0) ?? 0;
    const accounts = new Set(rows?.map((p) => p.account_id)).size;
    const summary = rows && `${plural(rows.length, "position")} · ${plural(accounts, "account")} · ${usd.format(total)}`;
    return (
      <ToolCard label="Positions" result={result} summary={summary}>
        {rows && (
          <table>
            <tbody>
              {rows.slice(0, ROWS).map((p) => (
                <tr key={`${p.account_id}:${p.ticker}`}>
                  <td className="t">{p.ticker}</td>
                  <td>{usd.format(p.market_value)}</td>
                  <td>{total ? pct.format(p.market_value / total) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {rows && rows.length > ROWS && <p className="more">+{rows.length - ROWS} more</p>}
      </ToolCard>
    );
  },
});

// ---------- get_portfolio_risk ----------

const riskSummary = z.object({
  total_exposure: z.number(),
  concentration: z.array(z.object({ ticker: z.string(), pct_of_portfolio: z.number() })),
  drawdown: z.number().nullable(),
  position_count: z.number(),
});
const riskResult = z.object({ combined: riskSummary });

const risk = defineToolCallRenderer({
  name: "get_portfolio_risk",
  args: z.object({}),
  render: function Risk({ result }) {
    const data = result === undefined ? null : parseJson(result, riskResult);
    // The tool lists one entry per account position; a ticker held in two
    // accounts appears twice. Combine them so the card matches the dashboard.
    const byTicker = new Map<string, number>();
    for (const c of data?.combined.concentration ?? []) {
      byTicker.set(c.ticker, (byTicker.get(c.ticker) ?? 0) + c.pct_of_portfolio);
    }
    const top = [...byTicker]
      .map(([ticker, pct_of_portfolio]) => ({ ticker, pct_of_portfolio }))
      .sort((a, b) => b.pct_of_portfolio - a.pct_of_portfolio)
      .slice(0, 5);
    const topShare = top.reduce((sum, c) => sum + c.pct_of_portfolio, 0);
    const summary =
      data &&
      [
        `top 5 = ${pct.format(topShare)}`,
        data.combined.drawdown === null ? null : `drawdown ${pct.format(data.combined.drawdown)}`,
      ]
        .filter(Boolean)
        .join(" · ");
    return (
      <ToolCard label="Concentration & risk" result={result} summary={summary}>
        <table>
          <tbody>
            {top.map((c) => (
              <tr key={c.ticker}>
                <td className="t">{c.ticker}</td>
                <td className="bar-cell">
                  <span className="bar" style={{ width: `${Math.min(100, (c.pct_of_portfolio / (top[0]?.pct_of_portfolio || 1)) * 100)}%` }} />
                </td>
                <td>{pct.format(c.pct_of_portfolio)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </ToolCard>
    );
  },
});

// ---------- get_signals ----------

const signalsResult = z.array(
  z.object({
    id: z.number(),
    ticker: z.string(),
    signal_type: z.string(),
    direction: z.string().nullable(),
    confidence: z.number().nullable(),
  }),
);

const signals = defineToolCallRenderer({
  name: "get_signals",
  args: z.object({ ticker: z.string().optional(), category: z.string().optional(), hours: z.number().optional() }),
  render: function Signals({ args, result }) {
    const rows = result === undefined ? null : parseJson(result, signalsResult);
    const scope = [args.ticker, args.category, `last ${args.hours ?? 24}h`].filter(Boolean).join(" · ");
    const summary = rows && `${plural(rows.length, "signal")} · ${scope}`;
    return (
      <ToolCard label="Signals" result={result} summary={summary}>
        {rows && rows.length === 0 && <p className="more">None in this window.</p>}
        {rows && rows.length > 0 && (
          <table>
            <tbody>
              {rows.slice(0, ROWS).map((s) => (
                <tr key={s.id}>
                  <td className="t">{s.ticker}</td>
                  <td className="muted">{s.signal_type.replaceAll("_", " ")}</td>
                  <td className={s.direction === "long" ? "gain" : "muted"}>
                    {s.direction ?? "—"}
                  </td>
                  <td>{s.confidence === null ? "—" : pct.format(s.confidence)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {rows && rows.length > ROWS && <p className="more">+{rows.length - ROWS} more</p>}
      </ToolCard>
    );
  },
});

// ---------- get_price_history ----------

const priceResult = z.object({
  ticker: z.string(),
  bar_count: z.number(),
  bars: z.array(
    z.object({
      timestamp: z.string(),
      close: z.number(),
      rsi_14: z.number().nullable(),
      macd_hist: z.number().nullable(),
    }),
  ),
});

const prices = defineToolCallRenderer({
  name: "get_price_history",
  args: z.object({ ticker: z.string(), limit: z.number().optional() }),
  render: function Prices({ args, result }) {
    const data = result === undefined ? null : parseJson(result, priceResult);
    const first = data?.bars.at(0);
    const last = data?.bars.at(-1);
    const change = first && last && first.close ? last.close / first.close - 1 : null;
    const summary =
      data &&
      (last
        ? [`${data.ticker} ${num.format(last.close)}`, change === null ? null : `${change >= 0 ? "+" : ""}${pct.format(change)} over ${plural(data.bar_count, "bar")}`, last.rsi_14 === null ? null : `RSI ${num.format(last.rsi_14)}`]
            .filter(Boolean)
            .join(" · ")
        : `${data.ticker} · no bars stored`);
    return (
      <ToolCard label={args.ticker ? `${args.ticker.toUpperCase()} price history` : "Price history"} result={result} summary={summary}>
        {data && data.bars.length > 0 && (
          <table>
            <tbody>
              {data.bars.slice(-ROWS).reverse().map((b) => (
                <tr key={b.timestamp}>
                  <td className="muted">{b.timestamp.slice(0, 16).replace("T", " ")}</td>
                  <td>{num.format(b.close)}</td>
                  <td className="muted">{b.rsi_14 === null ? "" : `RSI ${num.format(b.rsi_14)}`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </ToolCard>
    );
  },
});

// Left to inference: each renderer is typed by its own args schema.
export const toolRenderers = [positions, risk, signals, prices];
