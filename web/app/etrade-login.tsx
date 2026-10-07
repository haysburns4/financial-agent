"use client";

import { useId, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import type { z } from "zod";

import { authCompleteSchema, authStartSchema, type EtradeAction } from "./etrade";
import { proxyErrorSchema } from "./portfolio";

const EXPIRED_HINT = "Verification codes expire after a few minutes and work only once.";

type Step =
  | { kind: "closed" }
  | { kind: "starting" }
  | { kind: "code"; authUrl: string; blocked: boolean }
  | { kind: "submitting"; authUrl: string }
  | { kind: "failed"; message: string; hint: string | null }
  | { kind: "done" };

type EtradeLoginProps = {
  /** Called after E-Trade accepts the code; should pull fresh positions. */
  onLoggedIn: () => Promise<void>;
};

async function send<T>(action: EtradeAction, schema: z.ZodType<T>): Promise<T> {
  const response = await fetch("/api/etrade", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(action),
    cache: "no-store",
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const failure = proxyErrorSchema.safeParse(body);
    throw new Error(failure.success ? failure.data.error : `HTTP ${response.status}`);
  }
  return schema.parse(body);
}

/**
 * Replaces the "logged out" pill: a button that opens a small inline flow —
 * E-Trade's authorize page in a new tab, then a field for the code it shows.
 */
export function EtradeLogin({ onLoggedIn }: EtradeLoginProps) {
  const [step, setStep] = useState<Step>({ kind: "closed" });
  const [code, setCode] = useState("");
  const button = useRef<HTMLButtonElement>(null);
  const flowId = useId();
  const inputId = useId();

  const open = step.kind !== "closed";

  function cancel() {
    setStep({ kind: "closed" });
    setCode("");
    button.current?.focus();
  }

  function begin() {
    // Open the tab inside the click: after an await, browsers block it as an
    // unrequested popup. It is pointed at E-Trade once the URL arrives.
    const tab = window.open("about:blank", "_blank");
    if (tab) tab.opener = null;
    setCode("");
    setStep({ kind: "starting" });
    void start(tab);
  }

  async function start(tab: Window | null) {
    try {
      const { auth_url } = await send({ action: "start" }, authStartSchema);
      if (tab) tab.location.href = auth_url;
      setStep({ kind: "code", authUrl: auth_url, blocked: tab === null });
    } catch (exc) {
      tab?.close();
      setStep({ kind: "failed", message: exc instanceof Error ? exc.message : String(exc), hint: null });
    }
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const verifier = code.trim();
    if (step.kind !== "code" || !verifier) return;
    setStep({ kind: "submitting", authUrl: step.authUrl });
    try {
      await send({ action: "complete", verifier }, authCompleteSchema);
    } catch (exc) {
      setStep({ kind: "failed", message: exc instanceof Error ? exc.message : String(exc), hint: EXPIRED_HINT });
      return;
    }
    setStep({ kind: "done" });
    // Once positions load the panel shows "connected" and unmounts this.
    await onLoggedIn();
    setStep({ kind: "closed" });
  }

  function onKeyDown(event: KeyboardEvent<HTMLElement>) {
    if (event.key === "Escape") {
      event.preventDefault();
      cancel();
    }
  }

  return (
    <>
      <button
        ref={button}
        type="button"
        className="pill warn login"
        aria-expanded={open}
        aria-controls={flowId}
        onClick={() => (open ? cancel() : begin())}
      >
        Log in to E-Trade
      </button>

      {open && (
        <div id={flowId} className="etrade-login" role="group" aria-label="E-Trade login" onKeyDown={onKeyDown}>
          {step.kind === "starting" && <p>Opening E-Trade…</p>}

          {(step.kind === "code" || step.kind === "submitting") && (
            <form onSubmit={(event) => void submit(event)}>
              <p>
                {step.kind === "code" && step.blocked ? "Your browser blocked the new tab. " : ""}
                Log in on the{" "}
                <a href={step.authUrl} target="_blank" rel="noopener noreferrer">
                  E-Trade page
                </a>
                , click Accept, then enter the code it shows.
              </p>
              <label htmlFor={inputId}>Verification code</label>
              <div className="row">
                <input
                  id={inputId}
                  value={code}
                  onChange={(event) => setCode(event.target.value)}
                  autoFocus
                  autoComplete="one-time-code"
                  spellCheck={false}
                  disabled={step.kind === "submitting"}
                  required
                />
                <button type="submit" className="refresh" disabled={step.kind === "submitting" || !code.trim()}>
                  {step.kind === "submitting" ? "Logging in…" : "Log in"}
                </button>
                <button type="button" className="link" onClick={cancel}>
                  Cancel
                </button>
              </div>
            </form>
          )}

          {step.kind === "failed" && (
            <>
              <p role="alert" className="login-error">
                {step.message}
                {step.hint && <span className="hint">{step.hint}</span>}
              </p>
              <div className="row">
                <button type="button" className="refresh" onClick={begin} autoFocus>
                  Get a new code
                </button>
                <button type="button" className="link" onClick={cancel}>
                  Cancel
                </button>
              </div>
            </>
          )}

          {step.kind === "done" && <p role="status">Logged in. Loading your positions…</p>}
        </div>
      )}
    </>
  );
}
