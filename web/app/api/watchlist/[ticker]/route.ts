/**
 * One watchlist row.
 *
 *   PATCH  -> PATCH  /watchlist/{ticker} { note?, tags?, source: "ui" }
 *   DELETE -> DELETE /watchlist/{ticker}?source=ui -> the row plus `held`
 */
import { patchRequestSchema, removedSchema, watchlistEntrySchema } from "../../../watchlist";
import { API_ORIGIN, failure, unexpected, unreachable } from "../../upstream";

export async function PATCH(request: Request, ctx: RouteContext<"/api/watchlist/[ticker]">): Promise<Response> {
  const { ticker } = await ctx.params;
  const body = patchRequestSchema.safeParse(await request.json().catch(() => null));
  if (!body.success) return Response.json({ error: "Expected {\"note\"?: …, \"tags\"?: […]}" }, { status: 400 });
  try {
    const response = await fetch(`${API_ORIGIN}/watchlist/${encodeURIComponent(ticker)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...body.data, source: "ui" }),
      cache: "no-store",
    });
    if (!response.ok) return failure(response, `Saving ${ticker}`);
    const parsed = watchlistEntrySchema.safeParse(await response.json());
    if (!parsed.success) return unexpected(`/watchlist/${ticker}`, parsed.error);
    return Response.json(parsed.data);
  } catch {
    return unreachable();
  }
}

export async function DELETE(_request: Request, ctx: RouteContext<"/api/watchlist/[ticker]">): Promise<Response> {
  const { ticker } = await ctx.params;
  try {
    const response = await fetch(`${API_ORIGIN}/watchlist/${encodeURIComponent(ticker)}?source=ui`, {
      method: "DELETE",
      cache: "no-store",
    });
    if (!response.ok) return failure(response, `Removing ${ticker}`);
    const parsed = removedSchema.safeParse(await response.json());
    if (!parsed.success) return unexpected(`/watchlist/${ticker}`, parsed.error);
    return Response.json(parsed.data);
  } catch {
    return unreachable();
  }
}

export const dynamic = "force-dynamic";
