/**
 * Portfolio panel endpoint.
 *
 * Like the CopilotKit route, this proxies to FastAPI server-to-server so the
 * browser never talks to the Python service directly.
 *
 *   GET  -> GET  /dashboard
 *   POST -> POST /pipeline/portfolio/run (pull fresh positions from E-Trade),
 *           then GET /dashboard
 */
import { dashboardSchema } from "../../portfolio";
import { API_ORIGIN, failure, unexpected, unreachable } from "../upstream";

async function dashboard(): Promise<Response> {
  const response = await fetch(`${API_ORIGIN}/dashboard`, { cache: "no-store" });
  if (!response.ok) return failure(response, "Loading the portfolio");

  const parsed = dashboardSchema.safeParse(await response.json());
  if (!parsed.success) return unexpected("/dashboard", parsed.error);
  return Response.json(parsed.data);
}

export async function GET(): Promise<Response> {
  try {
    return await dashboard();
  } catch {
    return unreachable();
  }
}

export async function POST(): Promise<Response> {
  try {
    const run = await fetch(`${API_ORIGIN}/pipeline/portfolio/run`, {
      method: "POST",
      cache: "no-store",
    });
    if (!run.ok) return failure(run, "Refreshing from E-Trade");
    return await dashboard();
  } catch {
    return unreachable();
  }
}

export const dynamic = "force-dynamic";
