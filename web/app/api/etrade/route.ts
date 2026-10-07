/**
 * E-Trade login from the dashboard. Proxies FastAPI's OAuth endpoints so the
 * daily re-auth (E-Trade ends sessions at midnight ET) needs no terminal.
 *
 *   POST { action: "start" }                -> POST /auth/start    -> { auth_url }
 *   POST { action: "complete", verifier }   -> POST /auth/complete -> { authenticated: true }
 */
import { authCompleteSchema, authStartSchema, etradeActionSchema } from "../../etrade";
import { API_ORIGIN, failure, unexpected, unreachable } from "../upstream";

async function start(): Promise<Response> {
  const response = await fetch(`${API_ORIGIN}/auth/start`, { method: "POST", cache: "no-store" });
  if (!response.ok) return failure(response, "Starting the E-Trade login");

  const parsed = authStartSchema.safeParse(await response.json());
  if (!parsed.success) return unexpected("/auth/start", parsed.error);
  return Response.json(parsed.data);
}

async function complete(verifier: string): Promise<Response> {
  const response = await fetch(`${API_ORIGIN}/auth/complete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ verifier }),
    cache: "no-store",
  });
  if (!response.ok) return failure(response, "Logging in to E-Trade");

  const parsed = authCompleteSchema.safeParse(await response.json());
  if (!parsed.success) return unexpected("/auth/complete", parsed.error);
  return Response.json(parsed.data);
}

export async function POST(request: Request): Promise<Response> {
  const action = etradeActionSchema.safeParse(await request.json().catch(() => null));
  if (!action.success) {
    return Response.json(
      { error: 'Expected {"action":"start"} or {"action":"complete","verifier":"…"}' },
      { status: 400 },
    );
  }
  try {
    return action.data.action === "start" ? await start() : await complete(action.data.verifier);
  } catch {
    return unreachable();
  }
}
