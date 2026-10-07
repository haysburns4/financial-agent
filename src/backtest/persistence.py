"""Save / load backtest reports + apply suggested weights to the live config.

The "live" config is [src/backtest/confidence_config.py](src/backtest/confidence_config.py)'s
two-tier `CONFIDENCE_WEIGHTS` (`_default` per rule, plus per-ticker overrides)
and its `WEIGHTS_METADATA`. Applying mutates those dicts in place so
already-imported modules (notably src/signals/engine.py) see the new values
immediately, AND rewrites the source file so the change survives restarts.

Everything goes through a `CalibrationStore` whose paths and dicts are
injected, so tests calibrate against temp files instead of the real ones.
"""
import json
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from src.backtest import confidence_config
from src.backtest.confidence_config import WeightsMetadata
from src.backtest.runner import BacktestReport, WalkforwardReport


_PORTFOLIO_RULES = {"concentration_risk", "drawdown_alert", "stop_loss_warning"}
# A ticker override needs this many graded signals for that ticker and rule...
_MIN_OVERRIDE_EVALUATED = 30
# ...and, from a walk-forward report, evidence from this many separate test
# windows, so one lucky stretch (one regime) cannot set it.
_MIN_OVERRIDE_WINDOWS = 6

_CONFIG_HEADER = '''"""Calibrated confidence weights per signal rule, in two tiers.

`CONFIDENCE_WEIGHTS["_default"]` holds one weight per rule. Every other key is
a ticker, mapping rules to an override for that ticker only; the live signal
engine uses the override when there is one and `_default` otherwise.

`WEIGHTS_METADATA` records when the weights were last calibrated, from which
backtest type ("walkforward" or "single_window"), and over which report window
(`{"start": ..., "end": ...}`); all None until the first calibration.

Edit manually or via `POST /backtest/apply`, which rewrites this file (header
included, from src/backtest/persistence.py) with the latest report's suggested
confidences. Mutating these dicts in-process takes effect immediately because
the signal engine reads them by attribute lookup, not by import binding.
"""
from typing import Literal, TypedDict


class ReportWindow(TypedDict):
    start: str
    end: str


class WeightsMetadata(TypedDict):
    calibrated_at: str | None
    source: Literal["walkforward", "single_window"] | None
    report_window: ReportWindow | None


'''


def _json_default(obj: Any):  # anti-slop: allow no-any-parameters - signature is dictated by json.dumps(default=...)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def render_config(weights: dict[str, dict[str, float]], metadata: WeightsMetadata) -> str:
    """The full text of confidence_config.py for these weights and metadata."""
    lines = [_CONFIG_HEADER.rstrip("\n"), "", "CONFIDENCE_WEIGHTS: dict[str, dict[str, float]] = {"]
    # _default first, then tickers alphabetically, so diffs stay readable.
    for tier in sorted(weights, key=lambda k: (k != "_default", k)):
        lines.append(f'    "{tier}": {{')
        for rule, weight in weights[tier].items():
            lines.append(f'        "{rule}": {float(weight):.4f},')
        lines.append("    },")
    lines.append("}")
    lines.append("")
    lines.append("WEIGHTS_METADATA: WeightsMetadata = {")
    for key in ("calibrated_at", "source", "report_window"):
        lines.append(f'    "{key}": {metadata[key]!r},')
    lines.append("}")
    return "\n".join(lines) + "\n"


def _generated_at(payload: dict | None) -> datetime | None:
    if payload is None:
        return None
    try:
        return datetime.fromisoformat(payload["generated_at"])
    except (KeyError, TypeError, ValueError):
        return None


class CalibrationStore:
    def __init__(
        self,
        data_dir: Path,
        config_path: Path,
        weights: dict[str, dict[str, float]],
        metadata: WeightsMetadata,
    ) -> None:
        self.report_path = data_dir / "backtest_latest.json"
        self.walkforward_path = data_dir / "walkforward_latest.json"
        self.config_path = config_path
        self.weights = weights
        self.metadata = metadata

    # ---------- reports ----------

    def _save(self, path: Path, report: BacktestReport | WalkforwardReport) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(report), default=_json_default, indent=2))
        return path

    def _load(self, path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None

    def save_report(self, report: BacktestReport) -> Path:
        return self._save(self.report_path, report)

    def load_report_json(self) -> dict | None:
        return self._load(self.report_path)

    def save_walkforward_report(self, report: WalkforwardReport) -> Path:
        return self._save(self.walkforward_path, report)

    def load_walkforward_json(self) -> dict | None:
        return self._load(self.walkforward_path)

    # ---------- applying ----------

    def apply_latest_report(self) -> dict:
        """Push the latest report's suggested weights into the live config + file.

        Prefers the walk-forward report when it is newer than the single-window
        one: its suggested confidences already carry the stability penalty.

        Policy: portfolio/position-conditional rules keep their static weights
        (they aren't measurable in a per-ticker price backtest), and rules with
        no graded events keep theirs too — silence ≠ evidence. Ticker overrides
        are written only where that ticker has enough evidence of its own; they
        are added or updated, never removed.
        """
        single = self.load_report_json()
        walkforward = self.load_walkforward_json()
        if single is None and walkforward is None:
            raise FileNotFoundError(
                "no backtest report on disk — run POST /backtest/run or "
                "POST /backtest/walkforward/run first"
            )
        single_at, walkforward_at = _generated_at(single), _generated_at(walkforward)
        use_walkforward = walkforward is not None and (
            single is None or single_at is None
            or (walkforward_at is not None and walkforward_at > single_at)
        )
        source = "walkforward" if use_walkforward else "single_window"
        payload = walkforward if use_walkforward else single
        assert payload is not None  # one of them exists, checked above

        defaults = self.weights.setdefault("_default", {})
        updated: dict[str, dict] = {}
        overrides: dict[str, dict[str, dict]] = {}
        skipped: list[str] = []
        warnings: list[str] = []

        for r in payload.get("rules", []):
            name = r["rule_name"]
            if name in _PORTFOLIO_RULES:
                skipped.append(f"{name}: portfolio-level rule, keeping static weight")
                continue
            hit_rate = r.get("aggregate_hit_rate") if use_walkforward else r.get("hit_rate")
            if hit_rate is None:
                skipped.append(f"{name}: no graded events in the report, weight unchanged")
                continue

            new = float(r["suggested_confidence"])
            updated[name] = {"old": defaults.get(name), "new": new}
            defaults[name] = new
            if r.get("sample_size_warning"):
                warnings.append(
                    f"{name}: a test window had < 20 occurrences, treat with caution"
                    if use_walkforward
                    else f"{name}: sample_size < {_MIN_OVERRIDE_EVALUATED}, treat with caution"
                )

            for ticker, t in (r.get("by_ticker") or {}).items():
                if use_walkforward:
                    qualifies = (
                        t["total_evaluated"] >= _MIN_OVERRIDE_EVALUATED
                        and t["windows_evaluated"] >= _MIN_OVERRIDE_WINDOWS
                    )
                else:
                    qualifies = t["evaluated"] >= _MIN_OVERRIDE_EVALUATED
                if not qualifies:
                    continue
                tier = self.weights.setdefault(ticker, {})
                ticker_new = float(t["suggested_confidence"])
                overrides.setdefault(ticker, {})[name] = {"old": tier.get(name), "new": ticker_new}
                tier[name] = ticker_new

        calibrated_at = datetime.now(timezone.utc).isoformat()
        self.metadata["calibrated_at"] = calibrated_at
        self.metadata["source"] = source
        self.metadata["report_window"] = {
            "start": str(payload.get("start_date")),
            "end": str(payload.get("end_date")),
        }
        self.config_path.write_text(render_config(self.weights, self.metadata))

        return {
            "source": source,
            "updated": updated,
            "overrides": overrides,
            "skipped": skipped,
            "warnings": warnings,
            "report_generated_at": payload.get("generated_at"),
            "calibrated_at": calibrated_at,
        }


# The live store: data/ next to the DB, and the real config module's dicts.
STORE = CalibrationStore(
    data_dir=Path("data"),
    config_path=Path(confidence_config.__file__),
    weights=confidence_config.CONFIDENCE_WEIGHTS,
    metadata=confidence_config.WEIGHTS_METADATA,
)
