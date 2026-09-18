"""Signal synthesizer — turn a batch of raw signals into a concise briefing.

Goes through the provider-neutral LLM layer. Given a list of signal dicts plus context
(positions and recent bars per ticker), asks Claude to produce a ≤200-word
actionable briefing. Callers (Discord alerter, /signals/synthesize endpoint)
substitute this narrative for raw per-signal posts.
"""
from typing import Any

from loguru import logger

from src.llm import LLMBackend, Message, collect


_SYSTEM_PROMPT = (
    "You are a financial analysis assistant for a personal equity portfolio. "
    "You receive technical signals, portfolio positions, and market data. "
    "Produce a concise, actionable briefing. State what signals fired, why "
    "they matter given the portfolio context, and what the user should "
    "consider. Never give direct buy/sell advice — frame as 'consider', "
    "'worth watching', 'may warrant review'. Keep it under 200 words."
)


class SignalSynthesizer:
    MAX_TOKENS = 1024

    def __init__(self, backend: LLMBackend) -> None:
        self._backend = backend

    async def synthesize(self, signals: list[dict], context: dict) -> str:
        """Produce a narrative summary for the given signals.

        `signals` is the raw signal dicts (id, ticker, signal_type, direction,
        confidence, reasoning, timestamp). `context` carries supplementary
        data — see `_render_user_message` for the expected shape.

        Raises on API failure; callers decide whether to fall back.
        """
        if not signals:
            return "No new signals."
        user_message = self._render_user_message(signals, context)
        response = await collect(
            self._backend.stream(
                system=_SYSTEM_PROMPT,
                messages=[Message(role="user", text=user_message)],
                max_tokens=self.MAX_TOKENS,
            )
        )
        text = response.text.strip()
        logger.info(
            "synthesizer: {} signal(s) -> {} char briefing "
            "({} in / {} out tokens, {}/{})",
            len(signals),
            len(text),
            response.usage.input_tokens,
            response.usage.output_tokens,
            self._backend.provider,
            self._backend.model,
        )
        return text

    @staticmethod
    def _render_user_message(signals: list[dict], context: dict) -> str:
        positions: list[dict] = context.get("positions") or []
        bars_by_ticker: dict[str, list[dict]] = context.get("bars_by_ticker") or {}

        lines: list[str] = ["## Signals that fired"]
        for s in signals:
            ts = s.get("timestamp")
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            lines.append(
                f"- [{s.get('signal_type')}] {s.get('ticker')} "
                f"dir={s.get('direction')} conf={float(s.get('confidence', 0.0)):.2f} "
                f"@ {ts_str} — {s.get('reasoning')}"
            )

        if positions:
            lines.append("\n## Current positions in affected tickers")
            for p in positions:
                pnl_pct = (
                    (p["market_value"] - p["cost_basis"]) / p["cost_basis"]
                    if p.get("cost_basis")
                    else 0.0
                )
                lines.append(
                    f"- {p['ticker']} ({p.get('account_id', '?')}): "
                    f"qty={p['quantity']:.4g}, "
                    f"cost=${p['cost_basis']:.0f}, mkt=${p['market_value']:.0f}, "
                    f"pnl={pnl_pct:+.1%}"
                )
        else:
            lines.append("\n## Current positions in affected tickers\n- (none)")

        if bars_by_ticker:
            lines.append("\n## Last 5 bars + indicators per ticker")
            for ticker, bars in bars_by_ticker.items():
                lines.append(f"\n### {ticker}")
                for b in bars:
                    ts = b.get("timestamp")
                    ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                    lines.append(
                        f"- {ts_str}  O={b.get('open'):.2f}  H={b.get('high'):.2f}  "
                        f"L={b.get('low'):.2f}  C={b.get('close'):.2f}  "
                        f"V={b.get('volume'):.0f}  "
                        f"rsi={_fmt(b.get('rsi_14'))}  macd_hist={_fmt(b.get('macd_hist'))}  "
                        f"ema9={_fmt(b.get('ema_9'))}  ema21={_fmt(b.get('ema_21'))}"
                    )

        return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"
