/**
 * The E-Trade login contract: what the panel sends /api/etrade, and what
 * FastAPI's /auth endpoints send back. Parsed on both sides of the proxy, like
 * the portfolio schemas.
 */
import { z } from "zod";

export const etradeActionSchema = z.discriminatedUnion("action", [
  z.object({ action: z.literal("start") }),
  z.object({ action: z.literal("complete"), verifier: z.string().trim().min(1) }),
]);

/** POST /auth/start. The URL is opened in a new tab, so only ever an https E-Trade page. */
export const authStartSchema = z.object({
  auth_url: z.url({ protocol: /^https$/, hostname: /(^|\.)etrade\.com$/ }),
});

/** POST /auth/complete. Rejections are a 502 with FastAPI's `detail` instead. */
export const authCompleteSchema = z.object({ authenticated: z.literal(true) });

export type EtradeAction = z.input<typeof etradeActionSchema>;
