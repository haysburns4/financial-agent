/**
 * Watchlist panel endpoint. Proxies FastAPI server-to-server, like the
 * portfolio route.
 *
 *   GET  -> GET /watchlist/monitored + GET /watchlist -> { monitored, entries }
 *   POST -> POST /watchlist { ticker, note?, tags?, source: "ui" }
 *
 * A rejected add (unknown symbol, watchlist full, bad ticker) comes back as
 * `{ error }` with FastAPI's status and message, shown inline by the panel.
 */
import { z } from "zod";

import { addRequestSchema, monitoredTickerSchema, watchlistEntrySchema, watchlistViewSchema } from "../../watchlist";
import { API_ORIGIN, failure, unexpected, unreachable } from "../upstream";

async function view(): Promise<Response> {
  const [monitored, entries] = await Promise.all([
    fetch(`${API_ORIGIN}/watchlist/monitored`, { cache: "no-store" }),
    fetch(`${API_ORIGIN}/watchlist`, { cache: "no-store" }),
  ]);
  if (!monitored.ok) return failure(monitored, "Loading the monitored tickers");
  if (!entries.ok) return failure(entries, "Loading the watchlist");

  const parsed = watchlistViewSchema.safeParse({
    monitored: z.array(monitoredTickerSchema).parse(await monitored.json()),
    entries: z.array(watchlistEntrySchema).parse(await entries.json()),
  });
  if (!parsed.success) return unexpected("/watchlist", parsed.error);
  return Response.json(parsed.data);
}

export async function GET(): Promise<Response> {
  try {
    return await view();
  } catch (exc) {
    if (exc instanceof z.ZodError) return unexpected("/watchlist", exc);
    return unreachable();
  }
}

export async function POST(request: Request): Promise<Response> {
  const body = addRequestSchema.safeParse(await request.json().catch(() => null));
  if (!body.success) return Response.json({ error: "Expected {\"ticker\":\"…\"}" }, { status: 400 });
  try {
    const response = await fetch(`${API_ORIGIN}/watchlist`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...body.data, source: "ui" }),
      cache: "no-store",
    });
    if (!response.ok) return failure(response, `Adding ${body.data.ticker.toUpperCase()}`);
    const parsed = watchlistEntrySchema.safeParse(await response.json());
    if (!parsed.success) return unexpected("/watchlist", parsed.error);
    return Response.json(parsed.data, { status: 201 });
  } catch {
    return unreachable();
  }
}

export const dynamic = "force-dynamic";
