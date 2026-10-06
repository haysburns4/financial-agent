"use client";

import { CopilotChat } from "@copilotkit/react-core/v2";

import { PortfolioPanel } from "./portfolio-panel";

export default function Page() {
  return (
    <div className="shell">
      <header className="masthead">
        <h1>financial-agent</h1>
        <p>portfolio &amp; signals · ask about holdings, concentration or recent signals</p>
      </header>
      <main className="dashboard">
        <PortfolioPanel />
        <aside className="chat" aria-label="Chat with the agent">
          <CopilotChat />
        </aside>
      </main>
    </div>
  );
}
