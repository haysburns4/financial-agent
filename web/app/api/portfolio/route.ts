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
import { dashboardSchema, fastApiErrorSchema } from "../../portfolio";

// FastAPI serves /agui and /dashboard from the same origin.
const API_ORIGIN = new URL(process.env.AGUI_URL ?? "http://127.0.0.1:8000/agui").origin;

async function failure(response: Response, action: string): Promise<Response> {
  const body = fastApiErrorSchema.safeParse(await response.json().catch(() => null));
  const reason = body.success ? body.data.detail : response.statusText;
  return Response.json({ error: `${action} failed: ${reason}` }, { status: response.status });
}

async function dashboard(): Promise<Response> {
  const response = await fetch(`${API_ORIGIN}/dashboard`, { cache: "no-store" });
  if (!response.ok) return failure(response, "Loading the portfolio");

  const parsed = dashboardSchema.safeParse(await response.json());
  if (!parsed.success) {
    return Response.json(
      { error: `Unexpected /dashboard payload: ${parsed.error.message}` },
      { status: 502 },
    );
  }
  return Response.json(parsed.data);
}

function unreachable(): Response {
  return Response.json(
    { error: `Can't reach the API at ${API_ORIGIN}. Is \`uv run python -m src.main\` running?` },
    { status: 502 },
  );
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
