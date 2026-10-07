"""Default confidence weights per signal rule, in two tiers.

`CONFIDENCE_WEIGHTS["_default"]` holds one weight per rule. Every other key is
a ticker, mapping rules to an override for that ticker only; the live signal
engine uses the override when there is one and `_default` otherwise.

These are the committed *defaults*. Calibration (POST /backtest/apply, and the
automatic one after every backfill) writes the live weights to
data/confidence_weights.json instead, and the API layers that file over these
values at startup — so calibrating never touches this source file. Delete the
JSON file to go back to these defaults.

`WEIGHTS_METADATA` records when the live weights were last calibrated, from
which backtest type ("walkforward" or "single_window"), and over which report
window (`{"start": ..., "end": ...}`); all None here, before any calibration.

The signal engine reads these dicts by attribute lookup, so mutating them
in-process (as loading and calibrating do) takes effect immediately.
"""
from typing import Literal, TypedDict


class ReportWindow(TypedDict):
    start: str
    end: str


class WeightsMetadata(TypedDict):
    calibrated_at: str | None
    source: Literal["walkforward", "single_window"] | None
    report_window: ReportWindow | None

CONFIDENCE_WEIGHTS: dict[str, dict[str, float]] = {
    "_default": {
        "oversold_reversal": 0.7500,
        "golden_cross": 0.7500,
        "breakout": 0.7500,
        "overbought_reversal": 0.7500,
        "death_cross": 0.7500,
        "stop_loss_warning": 0.7500,
        "concentration_risk": 0.8000,
        "drawdown_alert": 0.8000,
    },
}

WEIGHTS_METADATA: WeightsMetadata = {
    "calibrated_at": None,
    "source": None,
    "report_window": None,
}
