"""Save / load BacktestReport JSON + apply suggested weights to live config.

The "live" config is [src/backtest/confidence_config.py](src/backtest/confidence_config.py)'s
`CONFIDENCE_WEIGHTS` dict. We mutate it in-place so already-imported modules
(notably src/signals/engine.py) see the new values immediately, AND we
rewrite the source file so the change survives process restarts.
"""
import json
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.backtest import confidence_config
from src.backtest.runner import BacktestReport


_REPORT_PATH = Path("data/backtest_latest.json")
_CONFIG_PATH = Path(__file__).with_name("confidence_config.py")
_PORTFOLIO_RULES = {"concentration_risk", "drawdown_alert", "stop_loss_warning"}


def report_path() -> Path:
    return _REPORT_PATH


def _json_default(obj: Any):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def save_report(report: BacktestReport) -> Path:
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text(json.dumps(asdict(report), default=_json_default, indent=2))
    return _REPORT_PATH


def load_report_json() -> dict | None:
    if not _REPORT_PATH.exists():
        return None
    try:
        return json.loads(_REPORT_PATH.read_text())
    except json.JSONDecodeError:
        return None


def _write_confidence_file(weights: dict[str, float]) -> None:
    lines = [
        '"""Calibrated confidence weights per signal rule.',
        "",
        "These values are written to fired signals by the live signal engine. Edit",
        "manually or via `POST /backtest/apply` (which rewrites this file with the",
        "suggested confidences from the latest backtest run).",
        "",
        "Mutating CONFIDENCE_WEIGHTS in-process takes effect immediately because the",
        "signal engine accesses it by attribute lookup, not by import binding.",
        '"""',
        "",
        "CONFIDENCE_WEIGHTS: dict[str, float] = {",
    ]
    for rule, weight in weights.items():
        lines.append(f'    "{rule}": {float(weight):.4f},')
    lines.append("}")
    _CONFIG_PATH.write_text("\n".join(lines) + "\n")


def apply_latest_report() -> dict:
    """Read the persisted report and push suggested weights into the live
    config + the source file. Returns a summary of what changed.

    Policy: portfolio/position-conditional rules keep their static weights
    (they aren't measurable in a per-ticker price backtest). Rules with no
    events in the backtest window are also left untouched — silence ≠ evidence.
    """
    payload = load_report_json()
    if payload is None:
        raise FileNotFoundError("no backtest report on disk — run POST /backtest/run first")

    updated: dict[str, dict] = {}
    skipped: list[str] = []
    warnings: list[str] = []

    for r in payload.get("rules", []):
        name = r["rule_name"]
        if name in _PORTFOLIO_RULES:
            skipped.append(f"{name}: portfolio-level rule, keeping static weight")
            continue
        if r.get("hit_rate") is None:
            skipped.append(f"{name}: no events in backtest window, weight unchanged")
            continue
        old = confidence_config.CONFIDENCE_WEIGHTS.get(name)
        new = float(r["suggested_confidence"])
        updated[name] = {"old": old, "new": new}
        confidence_config.CONFIDENCE_WEIGHTS[name] = new
        if r.get("sample_size_warning"):
            warnings.append(f"{name}: sample_size < 30, treat with caution")

    _write_confidence_file(confidence_config.CONFIDENCE_WEIGHTS)

    return {
        "updated": updated,
        "skipped": skipped,
        "warnings": warnings,
        "report_generated_at": payload.get("generated_at"),
    }
