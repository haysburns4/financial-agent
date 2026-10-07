"use client";

import { CopilotKitProvider } from "@copilotkit/react-core/v2";
import { useState, type ReactNode } from "react";

import { toolRenderers } from "./tool-renderers";

export function Providers({ children }: { children: ReactNode }) {
  const [error, setError] = useState<string | null>(null);

  return (
    <CopilotKitProvider
      runtimeUrl="/api/copilotkit"
      // Must match the key registered in app/api/copilotkit/[[...rest]]/route.ts.
      agentId="financial"
      renderToolCalls={toolRenderers}
      // The dev-only Inspector floats over the chat with its own promos.
      enableInspector={false}
      // The backend reports failures as an AG-UI RUN_ERROR on a 200 stream, so
      // without this they would surface as a chat that silently does nothing.
      onError={(event) => setError(event.error.message)}
    >
      {error && (
        <div className="banner" role="alert">
          Agent error: {error}
        </div>
      )}
      {children}
    </CopilotKitProvider>
  );
}
