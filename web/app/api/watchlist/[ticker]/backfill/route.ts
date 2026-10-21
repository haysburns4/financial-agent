/**
 * Retry a failed history backfill.
 *
 *   POST -> POST /watchlist/{ticker}/backfill -> the row, now "pending"
 */
import { watchlistEntrySchema } from "../../../../watchlist";
import { API_ORIGIN, failure, unexpected, unreachable } from "../../../upstream";

export async function POST(_request: Request, ctx: RouteContext<"/api/watchlist/[ticker]/backfill">): Promise<Response> {
  const { ticker } = await ctx.params;
  try {
    const response = await fetch(`${API_ORIGIN}/watchlist/${encodeURIComponent(ticker)}/backfill`, {
      method: "POST",
      cache: "no-store",
    });
    if (!response.ok) return failure(response, `Retrying ${ticker}'s history`);
    const parsed = watchlistEntrySchema.safeParse(await response.json());
    if (!parsed.success) return unexpected(`/watchlist/${ticker}/backfill`, parsed.error);
    return Response.json(parsed.data);
  } catch {
    return unreachable();
  }
}

export const dynamic = "force-dynamic";
