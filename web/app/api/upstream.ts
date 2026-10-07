/**
 * Shared by the route handlers that proxy to FastAPI server-to-server, so the
 * browser never talks to the Python service directly. Every failure becomes
 * `{ error }` (see proxyErrorSchema) with the upstream status where there is one.
 */
import type { z } from "zod";

import { fastApiErrorSchema } from "../portfolio";

// FastAPI serves /agui, /dashboard and /auth from the same origin.
export const API_ORIGIN = new URL(process.env.AGUI_URL ?? "http://127.0.0.1:8000/agui").origin;

/** FastAPI answered with an error: pass its `detail` on, prefixed with what we were doing. */
export async function failure(response: Response, action: string): Promise<Response> {
  const body = fastApiErrorSchema.safeParse(await response.json().catch(() => null));
  const reason = body.success ? body.data.detail : response.statusText;
  return Response.json({ error: `${action} failed: ${reason}` }, { status: response.status });
}

/** FastAPI answered 2xx with a body that does not match its contract. */
export function unexpected(path: string, error: z.ZodError): Response {
  return Response.json({ error: `Unexpected ${path} payload: ${error.message}` }, { status: 502 });
}

/** FastAPI did not answer at all. */
export function unreachable(): Response {
  return Response.json(
    { error: `Can't reach the API at ${API_ORIGIN}. Is \`./start\` (or \`uv run python -m src.main\`) running?` },
    { status: 502 },
  );
}
