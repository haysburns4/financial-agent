"use client";

import { CopilotChat, useConfigureSuggestions } from "@copilotkit/react-core/v2";

// Questions the agent's tools can actually answer (see src/agent/tools.py).
const STARTERS = [
  { title: "Biggest concentration risks", message: "Where is my portfolio most concentrated, and how much do the top holdings drive my risk?" },
  { title: "What's dragging returns", message: "Which positions are losing the most, in dollars and in percent?" },
  { title: "Recent signals", message: "Summarise the signals that fired in the last 24 hours." },
  { title: "Compare my accounts", message: "How do my accounts differ in allocation and drawdown?" },
];

// Must match the agentId in providers.tsx. Outside <CopilotChat> the hook
// cannot infer it and would target an agent called "default" instead.
const AGENT_ID = "financial";

export function ChatPanel() {
  useConfigureSuggestions(
    { suggestions: STARTERS, available: "before-first-message", consumerAgentId: AGENT_ID },
    [],
  );

  return (
    // `dark` switches on CopilotKit's dark-mode utilities; globals.css maps its
    // colour tokens onto the dashboard palette.
    <aside className="chat dark" aria-label="Chat with the agent">
      <CopilotChat
        labels={{
          welcomeMessageText: "Ask about your portfolio",
          chatInputPlaceholder: "Ask about holdings, risk or signals…",
          chatDisclaimerText: "Answers come from your stored positions and signals. Not investment advice.",
        }}
      />
    </aside>
  );
}
