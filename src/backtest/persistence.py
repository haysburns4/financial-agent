"""Save / load backtest reports + apply suggested weights to the live weights.

The live weights are the in-memory dicts of
[src/backtest/confidence_config.py](src/backtest/confidence_config.py): two-tier
`CONFIDENCE_WEIGHTS` (`_default` per rule, plus per-ticker overrides) and
`WEIGHTS_METADATA`. That source file holds only the committed defaults.
Applying mutates the dicts in place, so already-imported modules (notably
src/signals/engine.py) see the new values immediately, and writes them to
data/confidence_weights.json, which the API loads over the defaults at startup.

Everything goes through a `CalibrationStore` whose paths and dicts are
injected, so tests calibrate against temp files instead of the real ones.
"""
import copy
import json
import os
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import TypeAdapter, ValidationError

from src.backtest import confidence_config
from src.backtest.confidence_config import WeightsMetadata
from src.backtest.runner import BacktestReport, WalkforwardReport


_PORTFOLIO_RULES = {"concentration_risk", "drawdown_alert", "stop_loss_warning"}
# A ticker override needs this many graded signals for that ticker and rule...
_MIN_OVERRIDE_EVALUATED = 30
# ...and, from a walk-forward report, evidence from this many separate test
# windows, so one lucky stretch (one regime) cannot set it.
_MIN_OVERRIDE_WINDOWS = 6
# Daily-trend confirmation thins rules out; below this many confirmed
# occurrences across all walk-forward windows a rule keeps its _default.
_MIN_WALKFORWARD_OCCURRENCES = 20

_WEIGHTS = TypeAdapter(dict[str, dict[str, float]])
_METADATA = TypeAdapter(WeightsMetadata)


def _json_default(obj: Any):  # anti-slop: allow no-any-parameters - signature is dictated by json.dumps(default=...)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


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
        weights: dict[str, dict[str, float]],
        metadata: WeightsMetadata,
    ) -> None:
        self.report_path = data_dir / "backtest_latest.json"
        self.walkforward_path = data_dir / "walkforward_latest.json"
        self.weights_path = data_dir / "confidence_weights.json"
        self.weights = weights
        self.metadata = metadata
        # The committed defaults, for rules a saved file predates.
        self._default_rules = dict(weights.get("_default", {}))

    # ---------- live weights ----------

    def load_live_weights(self) -> bool:
        """Replace the in-memory weights with the last calibration's, if saved.

        The file is authoritative for every tier it has; a rule added to the
        committed defaults since it was written keeps its default. A missing or
        malformed file leaves the defaults in place.
        """
        payload = self._load(self.weights_path)
        if payload is None:
            return False
        try:
            weights = _WEIGHTS.validate_python(payload.get("weights"))
            metadata = _METADATA.validate_python(payload.get("metadata"))
        except ValidationError as exc:
            logger.warning("ignoring malformed {}: {}", self.weights_path, exc)
            return False
        defaults = weights.setdefault("_default", {})
        for rule, weight in self._default_rules.items():
            defaults.setdefault(rule, weight)
        self.weights.clear()
        self.weights.update(weights)
        self.metadata.update(metadata)
        return True

    def _write_live_weights(self) -> None:
        """Atomically, so a crash mid-write can't leave a half-written file."""
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"weights": copy.deepcopy(self.weights), "metadata": dict(self.metadata)}
        tmp = self.weights_path.with_name(self.weights_path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(tmp, self.weights_path)

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
        """Apply the newer of the two persisted reports (see `apply_report`).

        Prefers the walk-forward report when it is newer than the single-window
        one: its suggested confidences already carry the stability penalty.
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
        payload = walkforward if use_walkforward else single
        assert payload is not None  # one of them exists, checked above
        return self.apply_report(payload, walkforward=use_walkforward)

    def apply_report(self, payload: dict, walkforward: bool) -> dict:
        """Push one report's suggested weights into the live weights + their file.

        Policy: portfolio/position-conditional rules keep their static weights
        (they aren't measurable in a per-ticker price backtest), and rules with
        no graded events — or, from walk-forward, too few confirmed ones — keep
        theirs too: silence ≠ evidence. For each rule it does recalibrate, the
        report replaces that rule's ticker overrides: a ticker keeps one only
        while it has enough evidence of its own, so stale overrides don't pile
        up as calibration repeats.
        """
        use_walkforward = walkforward
        source = "walkforward" if use_walkforward else "single_window"
        defaults = self.weights.setdefault("_default", {})
        updated: dict[str, dict] = {}
        overrides: dict[str, dict[str, dict]] = {}
        removed: dict[str, list[str]] = {}
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
            if use_walkforward and r.get("total_occurrences", 0) < _MIN_WALKFORWARD_OCCURRENCES:
                skipped.append(
                    f"{name}: only {r.get('total_occurrences')} confirmed occurrences across all "
                    f"windows (< {_MIN_WALKFORWARD_OCCURRENCES}), default unchanged"
                )
                warnings.append(f"{name}: too few confirmed signals to recalibrate")
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

            qualifying: dict[str, float] = {}
            for ticker, t in (r.get("by_ticker") or {}).items():
                if use_walkforward:
                    qualifies = (
                        t["total_evaluated"] >= _MIN_OVERRIDE_EVALUATED
                        and t["windows_evaluated"] >= _MIN_OVERRIDE_WINDOWS
                    )
                else:
                    qualifies = t["evaluated"] >= _MIN_OVERRIDE_EVALUATED
                if qualifies:
                    qualifying[ticker] = float(t["suggested_confidence"])

            for ticker in [k for k in self.weights if k != "_default"]:
                tier = self.weights[ticker]
                if name in tier and ticker not in qualifying:
                    del tier[name]
                    removed.setdefault(ticker, []).append(name)
                if not tier:
                    del self.weights[ticker]
            for ticker, ticker_new in qualifying.items():
                tier = self.weights.setdefault(ticker, {})
                overrides.setdefault(ticker, {})[name] = {"old": tier.get(name), "new": ticker_new}
                tier[name] = ticker_new

        calibrated_at = datetime.now(timezone.utc).isoformat()
        self.metadata["calibrated_at"] = calibrated_at
        self.metadata["source"] = source
        self.metadata["report_window"] = {
            "start": str(payload.get("start_date")),
            "end": str(payload.get("end_date")),
        }
        self._write_live_weights()

        return {
            "source": source,
            "updated": updated,
            "overrides": overrides,
            "removed_overrides": removed,
            "skipped": skipped,
            "warnings": warnings,
            "report_generated_at": payload.get("generated_at"),
            "calibrated_at": calibrated_at,
        }


# The live store: data/ next to the DB, and the real config module's dicts.
# The API calls STORE.load_live_weights() at startup (src/main.py).
STORE = CalibrationStore(
    data_dir=Path("data"),
    weights=confidence_config.CONFIDENCE_WEIGHTS,
    metadata=confidence_config.WEIGHTS_METADATA,
)
