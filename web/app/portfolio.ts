/**
 * The portfolio panel's data contract — mirrors `dashboard()` in src/portfolio.py.
 *
 * Parsed with zod on both sides of the proxy: the route handler parses what
 * FastAPI sends, and the panel parses what the route handler sends, so neither
 * trusts a payload it did not check.
 */
import { z } from "zod";

const totalsFields = {
  market_value: z.number(),
  cost_basis: z.number(),
  pnl: z.number(),
  pnl_pct: z.number().nullable(),
  position_count: z.number().int(),
};

export const positionSchema = z.object({
  account_id: z.string(),
  ticker: z.string(),
  quantity: z.number(),
  cost_basis: z.number(),
  market_value: z.number(),
  pnl: z.number(),
  pnl_pct: z.number().nullable(),
  last_updated: z.string(),
});

export const accountSchema = z.object({
  account_id: z.string(),
  ...totalsFields,
  positions: z.array(positionSchema),
});

export const dashboardSchema = z.object({
  totals: z.object({ ...totalsFields, account_count: z.number().int() }),
  accounts: z.array(accountSchema),
  last_updated: z.string().nullable(),
  authenticated: z.boolean(),
});

/** What the route handler returns instead of a dashboard when something fails. */
export const proxyErrorSchema = z.object({ error: z.string() });

/** FastAPI's HTTPException body. */
export const fastApiErrorSchema = z.object({ detail: z.string() });

export type Position = z.infer<typeof positionSchema>;
export type Account = z.infer<typeof accountSchema>;
export type Dashboard = z.infer<typeof dashboardSchema>;
