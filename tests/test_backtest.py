"""Walk-forward validation, the two-tier confidence config and calibration."""
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from sqlalchemy import insert

from src.backtest import confidence_config
from src.backtest.persistence import CalibrationStore
from src.backtest.runner import BacktestRunner, _Tally, summarize_windows, walkforward_windows
from src.models import Indicator, PriceBar
from src.signals import engine as signal_engine

# ---------- synthetic history ----------
#
# Business days 2024-01-01 .. 2025-12-31 in a steady daily uptrend (EMA-9 3 >
# EMA-21 2, close 100 above both), so confirmation lets long entries through.
# On chosen "event days" the close pokes above the 20-day high (100) on double
# volume: a breakout. The close five bars later decides the outcome: 103 is a
# win for an entry rule, 97 a loss. Highs stay at 100 so later bars never
# break out on their own.

START, END = date(2024, 1, 1), date(2026, 1, 1)


def _business_days(first: date, last: date) -> list[date]:
    days, d = [], first
    while d <= last:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _event_days(days: list[date]) -> list[date]:
    """Two per month, ~2 weeks apart, so forward windows never overlap."""
    out = []
    for year, month in {(d.year, d.month) for d in days}:
        in_month = [d for d in days if (d.year, d.month) == (year, month)]
        out += [in_month[2], in_month[12]]
    return sorted(out)


def _midnight(d: date) -> datetime:
    return datetime.combine(d, time(0), tzinfo=timezone.utc)


async def _seed(engine, ticker: str, days: list[date], wins: dict[date, bool], ema_9: float = 3.0) -> None:
    """One ticker's daily bars + indicators; `wins` maps event days to outcomes."""
    index = {d: i for i, d in enumerate(days)}
    close = {d: 100.0 for d in days}
    volume = {d: 1000.0 for d in days}
    for event, win in wins.items():
        close[event], volume[event] = 101.0, 2000.0
        close[days[index[event] + 5]] = 103.0 if win else 97.0
    bars, indicators = [], []
    for d in days:
        bars.append({
            "ticker": ticker, "timestamp": _midnight(d), "open": 100.0, "high": 100.0,
            "low": 97.0, "close": close[d], "volume": volume[d], "adjusted_close": close[d],
        })
        indicators.append({"ticker": ticker, "timestamp": _midnight(d), "ema_9": ema_9, "ema_21": 2.0})
    async with engine.begin() as conn:
        await conn.execute(insert(PriceBar), bars)
        await conn.execute(insert(Indicator), indicators)


@pytest.fixture
async def history(engine):
    days = _business_days(START, END - timedelta(days=1))
    events = _event_days(days)

    # STBL: in every 2025 month, one breakout wins and one loses -> 50% each window.
    await _seed(engine, "STBL", days, {d: i % 2 == 0 for i, d in enumerate(events)})
    # SWNG: whole months right, then whole months wrong -> 100%, 0%, 100%, ...
    await _seed(engine, "SWNG", days, {d: d.month % 2 == 1 for d in events})
    # FLAT: same breakouts, but EMA-9 below EMA-21: the daily trend is flat, so
    # confirmation (which wants an uptrend for a long entry) filters every one.
    await _seed(engine, "FLAT", days, {d: True for d in events}, ema_9=1.0)

    # LATE: history from July 2024, always right; only the windows whose train
    # period it covers (train_start >= Jul 2024: tests Jul-Dec 2025) count.
    late_days = [d for d in days if d >= date(2024, 7, 1)]
    await _seed(engine, "LATE", late_days, {d: True for d in events if d >= date(2024, 7, 1)})

    # A 5-minute bar that would be a breakout: the daily replay must not see it.
    stray = datetime(2025, 3, 20, 14, 30, tzinfo=timezone.utc)
    async with engine.begin() as conn:
        await conn.execute(insert(PriceBar), [{
            "ticker": "STBL", "timestamp": stray, "open": 100.0, "high": 100.0, "low": 100.0,
            "close": 200.0, "volume": 9000.0, "adjusted_close": 200.0,
        }])
        await conn.execute(insert(Indicator), [{"ticker": "STBL", "timestamp": stray, "ema_9": 3.0, "ema_21": 2.0}])


# ---------- windows ----------


def test_windows_roll_by_step_and_stop_before_end():
    windows = walkforward_windows(START, END, 12, 1, 1)

    assert len(windows) == 12
    first, last = windows[0], windows[-1]
    assert (first.train_start, first.train_end, first.test_start, first.test_end) == (
        date(2024, 1, 1), date(2025, 1, 1), date(2025, 1, 1), date(2025, 2, 1),
    )
    assert last.test_end == END


def test_windows_tile_without_gaps_across_month_ends():
    windows = walkforward_windows(date(2024, 1, 31), date(2026, 1, 31), 12, 1, 1)
    assert [w.test_end for w in windows[:3]] == [date(2025, 2, 28), date(2025, 3, 31), date(2025, 4, 30)]
    assert all(a.test_end == b.test_start for a, b in zip(windows, windows[1:]))


def test_window_parameters_are_respected():
    windows = walkforward_windows(START, END, 6, 3, 3)
    assert [(w.test_start, w.test_end) for w in windows[:2]] == [
        (date(2024, 7, 1), date(2024, 10, 1)),
        (date(2024, 10, 1), date(2025, 1, 1)),
    ]


def test_too_short_a_range_has_no_windows():
    assert walkforward_windows(START, date(2024, 12, 31), 12, 1, 1) == []


# ---------- stats ----------


def _tallies(*rates_and_counts: tuple[int, int]) -> list[_Tally]:
    return [_Tally(occurrences=n, evaluated=n, wins=w) for w, n in rates_and_counts]


def test_stable_rule_keeps_its_hit_rate():
    stats = summarize_windows(_tallies(*[(13, 20)] * 6))
    assert stats.aggregate_hit_rate == pytest.approx(0.65)
    assert stats.hit_rate_std == 0
    assert stats.stability_score == 1
    assert stats.suggested_confidence == pytest.approx(0.65)


def test_wildly_unstable_rule_is_discounted_below_a_stable_one():
    # 75% on average, but 100/0/100/100: the kind of regime-specific swing
    # that should not be trusted over a steady 65%.
    stats = summarize_windows(_tallies((20, 20), (0, 20), (20, 20), (20, 20)))
    assert stats.aggregate_hit_rate == pytest.approx(0.75)
    assert stats.stability_score == pytest.approx(1 - 0.4330127 / 0.75)
    expected = 0.75 * (1 - 0.3 * (1 - stats.stability_score))
    assert stats.suggested_confidence == pytest.approx(expected)
    assert stats.suggested_confidence < 0.65


def test_pooled_rate_weights_windows_by_sample_size():
    stats = summarize_windows([_Tally(10, 10, 10), _Tally(30, 30, 0)])
    assert stats.aggregate_hit_rate == pytest.approx(0.25)
    assert stats.hit_rate_mean == pytest.approx(0.5)


def test_suggestion_is_clamped():
    assert summarize_windows(_tallies(*[(20, 20)] * 3)).suggested_confidence == 0.95
    assert summarize_windows(_tallies(*[(2, 20)] * 3)).suggested_confidence == 0.3


def test_never_right_scores_zero_stability():
    stats = summarize_windows(_tallies((0, 20), (0, 20)))
    assert stats.stability_score == 0
    assert stats.suggested_confidence == 0.3


def test_no_graded_events_means_no_rates():
    stats = summarize_windows([_Tally(), _Tally()])
    assert stats.windows_evaluated == 0
    assert stats.aggregate_hit_rate is None
    assert stats.stability_score is None
    assert stats.window_hit_rates == []
    assert stats.sample_size_warning


def test_sample_size_warning_needs_20_occurrences_in_every_window():
    assert not summarize_windows(_tallies((10, 20), (10, 25))).sample_size_warning
    assert summarize_windows(_tallies((10, 20), (10, 19))).sample_size_warning


# ---------- end to end ----------


def _rule(report, name):
    return next(r for r in report.rules if r.rule_name == name)


async def test_walkforward_measures_only_out_of_sample_windows(engine, history):
    report = await BacktestRunner(engine).run_walkforward(["stbl", "SWNG", "LATE"], START, END)

    assert len(report.windows) == 12
    assert report.tickers == ["STBL", "SWNG", "LATE"]
    assert report.timeframe == "daily"
    breakout = _rule(report, "breakout")

    stable = breakout.by_ticker["STBL"]
    # 2025 events only (2024 is train-only), and not the stray 5-minute bar.
    assert stable.total_occurrences == 24
    assert stable.window_hit_rates == [0.5] * 12
    assert stable.stability_score == 1
    assert stable.suggested_confidence == pytest.approx(0.5)

    swing = breakout.by_ticker["SWNG"]
    assert swing.window_hit_rates == [1.0, 0.0] * 6
    assert swing.aggregate_hit_rate == pytest.approx(0.5)
    assert swing.stability_score == 0
    assert swing.suggested_confidence == pytest.approx(0.35)

    late = breakout.by_ticker["LATE"]
    assert late.windows_evaluated == 6  # Jul-Dec 2025; its Jan-Jun events lack a full train window
    assert late.total_evaluated == 12
    assert late.suggested_confidence == 0.95

    assert breakout.total_evaluated == 24 + 24 + 12
    assert breakout.windows_evaluated == 12
    assert _rule(report, "concentration_risk").total_evaluated == 0


async def test_walkforward_applies_daily_trend_confirmation(engine, history):
    report = await BacktestRunner(engine).run_walkforward(["FLAT"], START, END)
    flat = _rule(report, "breakout")

    assert flat.total_occurrences == 0
    assert flat.total_filtered == 24
    assert flat.aggregate_hit_rate is None
    assert flat.by_ticker["FLAT"].total_filtered == 24


async def test_single_window_backtest_reports_by_ticker_and_ignores_intraday(engine, history):
    report = await BacktestRunner(engine).run(["STBL", "SWNG", "FLAT"], START, END)
    breakout = _rule(report, "breakout")

    assert set(breakout.by_ticker) == {"STBL", "SWNG", "FLAT"}
    # 2 a month for 24 months, minus January 2024's two (fewer than 20 bars of
    # lookback yet), and no stray 5-minute breakout.
    assert breakout.by_ticker["STBL"].occurrences == 46
    assert breakout.by_ticker["STBL"].hit_rate == pytest.approx(0.5)
    assert breakout.by_ticker["FLAT"].occurrences == 0
    assert breakout.by_ticker["FLAT"].filtered == 46
    assert breakout.filtered == 46


async def test_intraday_timeframe_reads_only_the_5_minute_bars(engine, history):
    report = await BacktestRunner(engine).run(["STBL"], START, END, timeframe="intraday")
    # The only intraday bar is the stray one: a single bar can't fire anything.
    assert all(r.occurrences == 0 for r in report.rules)
    assert report.timeframe == "intraday"


async def test_walkforward_rejects_a_range_shorter_than_one_window(engine):
    with pytest.raises(ValueError, match="too short"):
        await BacktestRunner(engine).run_walkforward(["AAPL"], START, date(2024, 6, 1))


async def test_walkforward_endpoint_turns_bad_windows_into_400():
    from src.server import create_app

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/backtest/walkforward/run", json={
            "tickers": ["AAPL"], "start_date": "2024-01-01", "end_date": "2024-06-01",
        })
    assert response.status_code == 400
    assert "too short" in response.json()["detail"]


# ---------- calibration ----------


def _store(tmp_path: Path) -> CalibrationStore:
    weights = {"_default": {"golden_cross": 0.75, "death_cross": 0.75, "breakout": 0.75, "concentration_risk": 0.8}}
    metadata = {"calibrated_at": None, "source": None, "report_window": None}
    return CalibrationStore(tmp_path / "data", weights, metadata)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _wf_rule(name, suggested, aggregate=0.6, by_ticker=None, occurrences=120):
    return {"rule_name": name, "aggregate_hit_rate": aggregate, "suggested_confidence": suggested,
            "total_occurrences": occurrences, "sample_size_warning": False, "by_ticker": by_ticker or {}}


def _wf_ticker(total_evaluated, windows_evaluated, suggested):
    return {"total_evaluated": total_evaluated, "windows_evaluated": windows_evaluated,
            "suggested_confidence": suggested}


WALKFORWARD = {
    "start_date": "2024-01-01", "end_date": "2026-01-01", "generated_at": "2026-10-07T12:00:00+00:00",
    "rules": [
        _wf_rule("golden_cross", 0.58, by_ticker={
            "AAPL": _wf_ticker(40, 8, 0.66),   # enough evaluations and windows
            "MSFT": _wf_ticker(40, 5, 0.70),   # too few windows: one regime
            "NVDA": _wf_ticker(25, 9, 0.80),   # too few evaluations
        }),
        _wf_rule("death_cross", 0.3, aggregate=None),
        _wf_rule("breakout", 0.9, aggregate=0.9, occurrences=19),  # confirmation left too few
        _wf_rule("concentration_risk", 0.9),
    ],
}
SINGLE = {
    "start_date": "2025-01-01", "end_date": "2025-12-31", "generated_at": "2026-10-06T12:00:00+00:00",
    "rules": [{
        "rule_name": "golden_cross", "hit_rate": 0.7, "suggested_confidence": 0.7,
        "sample_size_warning": False,
        "by_ticker": {"AAPL": {"evaluated": 30, "suggested_confidence": 0.72},
                      "MSFT": {"evaluated": 29, "suggested_confidence": 0.9}},
    }],
}


def test_apply_prefers_the_newer_walkforward_report(tmp_path):
    store = _store(tmp_path)
    _write(store.walkforward_path, WALKFORWARD)
    _write(store.report_path, SINGLE)

    result = store.apply_latest_report()

    assert result["source"] == "walkforward"
    assert store.weights["_default"]["golden_cross"] == 0.58
    assert store.weights["AAPL"] == {"golden_cross": 0.66}
    assert "MSFT" not in store.weights and "NVDA" not in store.weights
    assert result["overrides"] == {"AAPL": {"golden_cross": {"old": None, "new": 0.66}}}
    # No graded events, too few confirmed occurrences, and portfolio-level
    # rules all keep their weights.
    assert store.weights["_default"]["death_cross"] == 0.75
    assert store.weights["_default"]["breakout"] == 0.75
    assert any("breakout" in w for w in result["warnings"])
    assert store.weights["_default"]["concentration_risk"] == 0.8
    assert store.metadata["source"] == "walkforward"
    assert store.metadata["report_window"] == {"start": "2024-01-01", "end": "2026-01-01"}
    assert store.metadata["calibrated_at"] == result["calibrated_at"]


def test_apply_uses_the_single_window_report_when_it_is_newer(tmp_path):
    store = _store(tmp_path)
    _write(store.walkforward_path, {**WALKFORWARD, "generated_at": "2026-10-01T00:00:00+00:00"})
    _write(store.report_path, SINGLE)

    result = store.apply_latest_report()

    assert result["source"] == "single_window"
    assert store.weights["_default"]["golden_cross"] == 0.7
    assert store.weights["AAPL"] == {"golden_cross": 0.72}  # evaluated >= 30
    assert "MSFT" not in store.weights  # 29 is one short


def test_apply_with_only_a_walkforward_report(tmp_path):
    store = _store(tmp_path)
    _write(store.walkforward_path, WALKFORWARD)
    assert store.apply_latest_report()["source"] == "walkforward"


def test_apply_without_any_report_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        _store(tmp_path).apply_latest_report()


def test_applied_weights_are_saved_to_data_and_reload_on_startup(tmp_path):
    store = _store(tmp_path)
    _write(store.walkforward_path, WALKFORWARD)
    store.apply_latest_report()

    fresh = _store(tmp_path)  # a restarted API: committed defaults in memory
    assert fresh.load_live_weights()

    assert fresh.weights == store.weights
    assert fresh.metadata == store.metadata
    assert json.loads(store.weights_path.read_text())["metadata"]["source"] == "walkforward"


def test_reload_keeps_defaults_for_rules_the_file_predates(tmp_path):
    store = _store(tmp_path)
    _write(store.weights_path, {
        "weights": {"_default": {"golden_cross": 0.5}, "AAPL": {"golden_cross": 0.6}},
        "metadata": {"calibrated_at": "2026-10-07T00:00:00+00:00", "source": "walkforward",
                     "report_window": {"start": "2024-01-01", "end": "2026-01-01"}},
    })

    assert store.load_live_weights()

    assert store.weights["_default"]["golden_cross"] == 0.5
    assert store.weights["_default"]["breakout"] == 0.75  # not in the file: default
    assert store.weights["AAPL"] == {"golden_cross": 0.6}


def test_missing_or_malformed_weights_file_keeps_the_defaults(tmp_path):
    store = _store(tmp_path)
    assert not store.load_live_weights()
    _write(store.weights_path, {"weights": {"_default": {"golden_cross": "high"}}, "metadata": {}})
    assert not store.load_live_weights()
    assert store.weights["_default"]["golden_cross"] == 0.75


def test_calibrating_never_touches_the_committed_defaults_file():
    from src.backtest.persistence import STORE

    assert STORE.weights_path == Path("data") / "confidence_weights.json"
    source = Path(confidence_config.__file__).read_text()
    assert "data/confidence_weights.json" in source


# ---------- live lookup ----------


def test_engine_prefers_a_ticker_override(monkeypatch):
    monkeypatch.setitem(confidence_config.CONFIDENCE_WEIGHTS, "AAPL", {"golden_cross": 0.66})

    assert signal_engine._conf("golden_cross", "AAPL") == 0.66
    assert signal_engine._conf("golden_cross", "NO-SUCH-TICKER") == confidence_config.CONFIDENCE_WEIGHTS["_default"]["golden_cross"]
    assert signal_engine._conf("breakout", "AAPL") == confidence_config.CONFIDENCE_WEIGHTS["_default"]["breakout"]
    assert signal_engine._conf("unknown_rule", "AAPL", fallback=0.42) == 0.42


# ---------- automatic calibration (after every backfill) ----------


def test_recalibrating_a_rule_replaces_its_ticker_overrides(tmp_path):
    store = _store(tmp_path)
    store.weights["MSFT"] = {"golden_cross": 0.9}  # stale: MSFT no longer qualifies
    store.weights["TSLA"] = {"breakout": 0.5}      # breakout isn't recalibrated: kept
    _write(store.walkforward_path, WALKFORWARD)

    result = store.apply_latest_report()

    assert "MSFT" not in store.weights
    assert store.weights["TSLA"] == {"breakout": 0.5}
    assert store.weights["AAPL"] == {"golden_cross": 0.66}
    assert result["removed_overrides"] == {"MSFT": ["golden_cross"]}


async def test_calibrate_runs_a_walkforward_over_all_stored_history_and_applies_it(engine, history, tmp_path):
    from src.backtest.calibration import calibrate

    store = _store(tmp_path)
    result = await calibrate(engine, ["STBL", "SWNG", "LATE"], store=store, today=END)

    assert result["start_date"] == "2024-01-01"  # the earliest stored bar
    assert result["windows"] == 12
    assert result["source"] == "walkforward"
    assert store.walkforward_path.exists()
    # Breakout: pooled 50% across STBL+SWNG+LATE with some instability -> recalibrated.
    assert store.weights["_default"]["breakout"] == pytest.approx(result["updated"]["breakout"]["new"])
    assert store.weights["_default"]["breakout"] < 0.75
    # Crosses never fire here (no daily cross confirms) -> defaults untouched.
    assert store.weights["_default"]["golden_cross"] == 0.75
    assert store.metadata["source"] == "walkforward"


async def test_calibrate_without_history_is_reported_not_raised_by_the_api(engine, tmp_path):
    from src.backtest.calibration import calibrate

    with pytest.raises(ValueError, match="no price history"):
        await calibrate(engine, ["NOPE"], store=_store(tmp_path))


def test_calibration_summary_line():
    from src.cli.launcher import _Calibration, calibration_summary

    cal = _Calibration.model_validate({
        "windows": 47,
        "updated": {"breakout": {"old": 0.39, "new": 0.41}},
        "overrides": {"AAPL": {"breakout": {"old": None, "new": 0.5}}},
        "removed_overrides": {"MSFT": ["golden_cross"]},
        "skipped": ["golden_cross: only 0 confirmed occurrences", "drawdown_alert: portfolio-level rule"],
    })
    assert calibration_summary(cal) == (
        "[calibrate] walk-forward over 47 windows: breakout 0.39→0.41; "
        "1 ticker override(s) set, 1 removed; unchanged (too little evidence): golden_cross"
    )
    assert calibration_summary(_Calibration(not_run="no price history")) == "[calibrate] not run: no price history"
