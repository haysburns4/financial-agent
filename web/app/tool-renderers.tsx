"use client";

import type { ReactToolCallRenderer } from "@copilotkit/react-core/v2";

/**
 * Renders the agent's backend tool calls in the transcript.
 *
 * Status is inferred from whether a result has arrived rather than compared
 * against the ToolCallStatus enum, so these keep working if its members change.
 * `result` is typed `string` on the completed branch and `undefined` otherwise,
 * and the backend always sends it as JSON, so it needs no runtime narrowing.
 */
function toolCard(label: string): ReactToolCallRenderer<any>["render"] {
  return function ToolCard({ args, result }) {
    const argText = args && Object.keys(args).length > 0 ? JSON.stringify(args) : "";

    return (
      <div className="toolcard">
        <div className="row">
          <span className={result === undefined ? "dot" : "dot done"} />
          <span>{result === undefined ? `${label}…` : label}</span>
          {argText && <span className="args">{argText}</span>}
        </div>
        {result !== undefined && (
          <details>
            <summary>result</summary>
            <pre>{result}</pre>
          </details>
        )}
      </div>
    );
  };
}

export const toolRenderers: ReactToolCallRenderer<any>[] = [
  { name: "get_positions", render: toolCard("Reading positions") },
  { name: "get_portfolio_risk", render: toolCard("Checking concentration & risk") },
  { name: "get_signals", render: toolCard("Reading recent signals") },
  { name: "get_price_history", render: toolCard("Reading price history") },
];
