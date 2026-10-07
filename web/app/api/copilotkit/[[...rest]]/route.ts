/**
 * CopilotKit runtime endpoint.
 *
 * The browser never talks to FastAPI directly — it posts here, and the runtime
 * proxies to the Python AG-UI endpoint server-to-server. That is why the Python
 * service needs no CORS middleware and never has to be exposed to the browser.
 */
import { HttpAgent } from "@ag-ui/client";
import { CopilotRuntime, createCopilotRuntimeHandler } from "@copilotkit/runtime/v2";

const AGUI_URL = process.env.AGUI_URL ?? "http://127.0.0.1:8000/agui";

const runtime = new CopilotRuntime({
  agents: {
    // Must match the `agentId` passed to CopilotKitProvider.
    financial: new HttpAgent({ url: AGUI_URL }),
  },
});

const handler = createCopilotRuntimeHandler({
  runtime,
  basePath: "/api/copilotkit",
});

export const POST = handler;
export const GET = handler;
export const OPTIONS = handler;

// The agent streams; never let Next try to cache or prerender this route.
export const dynamic = "force-dynamic";
