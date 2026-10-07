"""Walk-forward validation, the two-tier confidence config and calibration."""
import json
import runpy
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from sqlalchemy import insert

from src.backtest import confidence_config
from src.backtest.persistence import CalibrationStore, render_config
from src.backtest.runner import BacktestRunner, _Tally, summarize_windows, walkforward_windows
from src.models import Indicator, PriceBar
from src.signals import engine as signal_engine

# ---------- synthetic history ----------
#
# Business days 2024-01-01 .. 2025-12-31. EMA-9 sits below EMA-21 except on
# chosen "cross days", where it pokes above: a golden cross on that bar (and a
# death cross the bar after, which these tests ignore). The close five bars
# later decides the outcome: 103 is a win for an entry rule, 97 a loss.

START, END = date(2024, 1, 1), date(2026, 1, 1)


def _business_days(first: date, last: date) -> list[date]:
    days, d = [], first
    while d <= last:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _cross_days(days: list[date]) -> list[date]:
    """Two per month, ~2 weeks apart, so forward windows never overlap."""
    out = []
    for year, month in {(d.year, d.month) for d in days}:
        in_month = [d for d in days if (d.year, d.month) == (year, month)]
        out += [in_month[2], in_month[12]]
    return sorted(out)


def _midnight(d: date) -> datetime:
    return datetime.combine(d, time(0), tzinfo=timezone.utc)


async def _seed(engine, ticker: str, days: list[date], wins: dict[date, bool]) -> None:
    """One ticker's bars + indicators; `wins` maps cross days to their outcome."""
    index = {d: i for i, d in enumerate(days)}
    close = {d: 100.0 for d in days}
    for cross, win in wins.items():
        close[days[index[cross] + 5]] = 103.0 if win else 97.0
    bars, indicators = [], []
    for d in days:
        bars.append({
            "ticker": ticker, "timestamp": _midnight(d), "open": close[d], "high": close[d],
            "low": close[d], "close": close[d], "volume": 1000.0, "adjusted_close": close[d],
        })
        indicators.append({
            "ticker": ticker, "timestamp": _midnight(d),
            "ema_9": 3.0 if d in wins else 1.0, "ema_21": 2.0,
        })
    async with engine.begin() as conn:
        await conn.execute(insert(PriceBar), bars)
        await conn.execute(insert(Indicator), indicators)


@pytest.fixture
async def history(engine):
    days = _business_days(START, END - timedelta(days=1))
    crosses = _cross_days(days)
    in_2025 = [d for d in crosses if d.year == 2025]

    # STBL: in every 2025 month, one cross wins and one loses -> 50% each window.
    stable = {d: i % 2 == 0 for i, d in enumerate(crosses)}
    # SWNG: whole months right, then whole months wrong -> 100%, 0%, 100%, ...
    swing = {d: d.month % 2 == 1 for d in crosses}
    await _seed(engine, "STBL", days, stable)
    await _seed(engine, "SWNG", days, swing)

    # LATE: history from July 2024, always right; only the windows whose train
    # period it covers (train_start >= Jul 2024: tests Jul-Dec 2025) count.
    late_days = [d for d in days if d >= date(2024, 7, 1)]
    await _seed(engine, "LATE", late_days, {d: True for d in crosses if d >= date(2024, 7, 1)})

    # A 5-minute bar with a cross on it: daily-only replay must not see it.
    stray = datetime(2025, 3, 20, 14, 30, tzinfo=timezone.utc)
    async with engine.begin() as conn:
        await conn.execute(insert(PriceBar), [{
            "ticker": "STBL", "timestamp": stray, "open": 100.0, "high": 100.0, "low": 100.0,
            "close": 100.0, "volume": 1000.0, "adjusted_close": 100.0,
        }])
        await conn.execute(insert(Indicator), [{"ticker": "STBL", "timestamp": stray, "ema_9": 3.0, "ema_21": 2.0}])
    return in_2025


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
    golden = _rule(report, "golden_cross")

    stable = golden.by_ticker["STBL"]
    # 2025 crosses only (2024 is train-only), and not the stray 5-minute bar.
    assert stable.total_occurrences == 24
    assert stable.window_hit_rates == [0.5] * 12
    assert stable.stability_score == 1
    assert stable.suggested_confidence == pytest.approx(0.5)

    swing = golden.by_ticker["SWNG"]
    assert swing.window_hit_rates == [1.0, 0.0] * 6
    assert swing.aggregate_hit_rate == pytest.approx(0.5)
    assert swing.stability_score == 0
    assert swing.suggested_confidence == pytest.approx(0.35)

    late = golden.by_ticker["LATE"]
    assert late.windows_evaluated == 6  # Jul-Dec 2025; its Jan-Jun crosses lack a full train window
    assert late.total_evaluated == 12
    assert late.suggested_confidence == 0.95

    assert golden.total_evaluated == 24 + 24 + 12
    assert golden.windows_evaluated == 12
    assert _rule(report, "concentration_risk").total_evaluated == 0


async def test_single_window_backtest_reports_by_ticker_and_ignores_intraday(engine, history):
    report = await BacktestRunner(engine).run(["STBL", "SWNG"], START, END)
    golden = _rule(report, "golden_cross")

    assert set(golden.by_ticker) == {"STBL", "SWNG"}
    assert golden.by_ticker["STBL"].occurrences == 48  # 2 a month for 24 months, no stray
    assert golden.by_ticker["STBL"].hit_rate == pytest.approx(0.5)


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
    return CalibrationStore(tmp_path / "data", tmp_path / "confidence_config.py", weights, metadata)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _wf_rule(name, suggested, aggregate=0.6, by_ticker=None):
    return {"rule_name": name, "aggregate_hit_rate": aggregate, "suggested_confidence": suggested,
            "sample_size_warning": False, "by_ticker": by_ticker or {}}


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
    # No graded events and portfolio-level rules keep their weights.
    assert store.weights["_default"]["death_cross"] == 0.75
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


def test_applied_config_file_round_trips(tmp_path):
    store = _store(tmp_path)
    _write(store.walkforward_path, WALKFORWARD)
    store.apply_latest_report()

    loaded = runpy.run_path(str(store.config_path))

    assert loaded["CONFIDENCE_WEIGHTS"] == store.weights
    assert loaded["WEIGHTS_METADATA"] == store.metadata
    assert list(loaded["CONFIDENCE_WEIGHTS"])[0] == "_default"


def test_committed_config_is_what_the_writer_produces():
    path = Path(confidence_config.__file__)
    assert path.read_text() == render_config(
        confidence_config.CONFIDENCE_WEIGHTS, confidence_config.WEIGHTS_METADATA,
    ), "regenerate confidence_config.py through persistence.render_config"


# ---------- live lookup ----------


def test_engine_prefers_a_ticker_override(monkeypatch):
    monkeypatch.setitem(confidence_config.CONFIDENCE_WEIGHTS, "AAPL", {"golden_cross": 0.66})

    assert signal_engine._conf("golden_cross", "AAPL") == 0.66
    assert signal_engine._conf("golden_cross", "MSFT") == confidence_config.CONFIDENCE_WEIGHTS["_default"]["golden_cross"]
    assert signal_engine._conf("breakout", "AAPL") == confidence_config.CONFIDENCE_WEIGHTS["_default"]["breakout"]
    assert signal_engine._conf("unknown_rule", "AAPL", fallback=0.42) == 0.42
