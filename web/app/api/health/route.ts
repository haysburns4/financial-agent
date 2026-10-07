/**
 * Liveness probe for the launcher (`./start`): it tells this app apart from
 * whatever else might be holding port 3000.
 */
export function GET(): Response {
  return Response.json({ service: "financial-agent-web" });
}
