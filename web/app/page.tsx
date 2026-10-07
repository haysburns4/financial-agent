"use client";

import { ChatPanel } from "./chat-panel";
import { PortfolioPanel } from "./portfolio-panel";

export default function Page() {
  return (
    <div className="shell">
      <header className="masthead">
        <h1>financial-agent</h1>
        <p>portfolio &amp; signals</p>
      </header>
      <main className="dashboard">
        <PortfolioPanel />
        <ChatPanel />
      </main>
    </div>
  );
}
