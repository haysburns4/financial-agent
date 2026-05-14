"""Calibrated confidence weights per signal rule.

These values are written to fired signals by the live signal engine. Edit
manually or via `POST /backtest/apply` (which rewrites this file with the
suggested confidences from the latest backtest run).

Mutating CONFIDENCE_WEIGHTS in-process takes effect immediately because the
signal engine accesses it by attribute lookup, not by import binding.
"""

CONFIDENCE_WEIGHTS: dict[str, float] = {
    "oversold_reversal": 0.7500,
    "golden_cross": 0.7500,
    "breakout": 0.7500,
    "overbought_reversal": 0.7500,
    "death_cross": 0.7500,
    "stop_loss_warning": 0.7500,
    "concentration_risk": 0.8000,
    "drawdown_alert": 0.8000,
}
