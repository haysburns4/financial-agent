/**
 * The watchlist panel's data contract — mirrors WatchlistEntry.as_json() and
 * MonitoredTicker in src/watchlist/. Parsed with zod on both sides of the
 * proxy, like the portfolio schemas.
 */
import { z } from "zod";

export const backfillStatusSchema = z.enum(["pending", "running", "done", "failed"]);

export const watchlistEntrySchema = z.object({
  ticker: z.string(),
  note: z.string().nullable(),
  tags: z.array(z.string()),
  added_at: z.string(),
  validated: z.boolean(),
  backfill_status: backfillStatusSchema,
  backfill_error: z.string().nullable(),
  backfill_done_at: z.string().nullable(),
  updated_at: z.string(),
});

export const monitoredTickerSchema = z.object({
  ticker: z.string(),
  source: z.enum(["watchlist", "held", "both"]),
  note: z.string().nullable(),
  tags: z.array(z.string()),
  /** Accounts holding it; empty when it is only on the watchlist. */
  accounts: z.array(z.string()),
  indicators_ready: z.boolean(),
});

/** GET /api/watchlist: the resolver's output plus the watchlist rows behind it. */
export const watchlistViewSchema = z.object({
  monitored: z.array(monitoredTickerSchema),
  entries: z.array(watchlistEntrySchema),
});

/** DELETE /watchlist/{ticker}: the removed row, and whether it stays monitored. */
export const removedSchema = watchlistEntrySchema.extend({ held: z.boolean() });

/** What the panel sends the proxy to add a ticker. */
export const addRequestSchema = z.object({
  ticker: z.string().trim().min(1),
  note: z.string().nullable().optional(),
  tags: z.array(z.string()).optional(),
});

/** What the panel sends the proxy to edit one; omitted fields are left alone. */
export const patchRequestSchema = z.object({
  note: z.string().nullable().optional(),
  tags: z.array(z.string()).optional(),
});

export type BackfillStatus = z.infer<typeof backfillStatusSchema>;
export type WatchlistEntry = z.infer<typeof watchlistEntrySchema>;
export type MonitoredTicker = z.infer<typeof monitoredTickerSchema>;
export type WatchlistView = z.infer<typeof watchlistViewSchema>;
export type AddRequest = z.infer<typeof addRequestSchema>;
export type PatchRequest = z.infer<typeof patchRequestSchema>;
