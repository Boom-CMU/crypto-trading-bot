"""
tests/test_neutral_score.py — Unit tests for neutral_score.py (Task 2)

Run: pytest tests/test_neutral_score.py -v
"""
from __future__ import annotations

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from neutral_score import compute_neutral_score


# ─────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────

def make_ohlcv(closes, volumes=None):
    """Build minimal OHLCV DataFrame from a close array."""
    n = len(closes)
    closes = np.array(closes, dtype=float)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1D"),
        "open":   closes * 0.998,
        "high":   closes * 1.005,
        "low":    closes * 0.995,
        "close":  closes,
        "volume": volumes if volumes is not None else np.full(n, 1_000_000.0),
    })


@pytest.fixture
def bullish_trend():
    """Linear uptrend +0.5%/day for 100 days."""
    return make_ohlcv([100 * (1.005 ** i) for i in range(100)])


@pytest.fixture
def bearish_trend():
    """Linear downtrend -0.5%/day for 100 days."""
    return make_ohlcv([100 * (0.995 ** i) for i in range(100)])


@pytest.fixture
def sideways():
    """Mean-reverting noise around 100."""
    np.random.seed(42)
    closes = 100 + np.random.randn(100) * 0.5
    return make_ohlcv(closes)


@pytest.fixture
def trend_then_pump():
    """Uptrend then 24h +20% pump (overextension)."""
    base = [100 * (1.003 ** i) for i in range(99)]
    base.append(base[-1] * 1.20)
    return make_ohlcv(base)


# ─────────────────────────────────────────────────────────────
#  Direction tests (spec-required)
# ─────────────────────────────────────────────────────────────

def test_bullish_data_produces_positive_score(bullish_trend):
    result = compute_neutral_score(bullish_trend)
    assert result["score"] > 0.5, f"Expected strong bullish, got {result['score']}"


def test_bearish_data_produces_negative_score(bearish_trend):
    result = compute_neutral_score(bearish_trend)
    assert result["score"] < -0.5, f"Expected strong bearish, got {result['score']}"


def test_sideways_data_produces_neutral_score(sideways):
    result = compute_neutral_score(sideways)
    assert abs(result["score"]) < 0.2, f"Expected neutral, got {result['score']}"


# ─────────────────────────────────────────────────────────────
#  Symmetry test (spec-required)
# ─────────────────────────────────────────────────────────────

def test_mirror_symmetry():
    """Bullish and bearish mirror should produce negated scores (±0.1)."""
    bull_closes = [100 * (1.005 ** i) for i in range(100)]
    bear_closes = [100 * (0.995 ** i) for i in range(100)]
    bull_score = compute_neutral_score(make_ohlcv(bull_closes))["score"]
    bear_score = compute_neutral_score(make_ohlcv(bear_closes))["score"]
    assert abs(bull_score + bear_score) < 0.15, (
        f"Symmetry broken: bull={bull_score}, bear={bear_score}"
    )


# ─────────────────────────────────────────────────────────────
#  Component tests (spec-required)
# ─────────────────────────────────────────────────────────────

def test_mean_reversion_dampens_after_pump(trend_then_pump, bullish_trend):
    """After 24h pump, score must be lower than equivalent steady trend."""
    pumped = compute_neutral_score(trend_then_pump)["score"]
    normal = compute_neutral_score(bullish_trend)["score"]
    assert pumped < normal, (
        f"Mean reversion not working: pumped={pumped} should < normal={normal}"
    )


def test_volume_confirmation_amplifies_trend():
    """A trend with a recent volume spike should score higher than the same trend with flat volume."""
    closes   = [100 * (1.003 ** i) for i in range(100)]
    low_vol  = make_ohlcv(closes, volumes=np.full(100, 500_000.0))
    high_vol = make_ohlcv(closes, volumes=[500_000.0] * 80 + [3_000_000.0] * 20)
    low_score  = compute_neutral_score(low_vol)["score"]
    high_score = compute_neutral_score(high_vol)["score"]
    assert high_score > low_score, (
        f"Volume amplification failed: high={high_score}, low={low_score}"
    )


def test_components_returned():
    df = make_ohlcv([100 * (1.003 ** i) for i in range(100)])
    result = compute_neutral_score(df)
    assert "components" in result
    for key in ["trend", "momentum", "mean_rev", "volume"]:
        assert key in result["components"], f"Missing component: {key}"
        val = result["components"][key]
        assert -1.0 <= val <= 1.0, f"Component {key}={val} out of [-1, +1]"


# ─────────────────────────────────────────────────────────────
#  Edge cases (spec-required)
# ─────────────────────────────────────────────────────────────

def test_score_bounded():
    """Score must stay in [-1, +1] regardless of extreme input."""
    extreme = make_ohlcv([100 * (1.10 ** i) for i in range(100)])
    result  = compute_neutral_score(extreme)
    assert -1.0 <= result["score"] <= 1.0


def test_insufficient_data_returns_neutral():
    """Fewer than MIN_BARS → score=0.0 with warning key."""
    small  = make_ohlcv([100, 101, 102])
    result = compute_neutral_score(small)
    assert result["score"] == 0.0
    assert result.get("warning") is not None


def test_no_llm_called(monkeypatch):
    """neutral_score must not make any HTTP requests."""
    def boom(*a, **kw):
        raise RuntimeError("LLM/HTTP should not be called from neutral_score")

    import requests
    monkeypatch.setattr(requests, "post", boom)
    monkeypatch.setattr(requests, "get",  boom)

    df = make_ohlcv([100 * (1.003 ** i) for i in range(100)])
    compute_neutral_score(df)   # must not raise


# ─────────────────────────────────────────────────────────────
#  Additional edge cases (Task 2 working protocol: ≥3 per task)
# ─────────────────────────────────────────────────────────────

def test_bearish_pump_also_dampens():
    """A crash (-20% last bar) should produce a less extreme negative score."""
    base = [100 * (0.997 ** i) for i in range(99)]
    base.append(base[-1] * 0.80)   # -20% crash
    crash  = make_ohlcv(base)
    normal = make_ohlcv([100 * (0.997 ** i) for i in range(100)])
    crash_score  = compute_neutral_score(crash)["score"]
    normal_score = compute_neutral_score(normal)["score"]
    # After a -20% crash, mean_rev pushes score back toward 0, so crash score > normal score
    assert crash_score > normal_score, (
        f"Expected crash_score ({crash_score}) > normal_score ({normal_score})"
    )


def test_mean_rev_symmetric():
    """Equal-magnitude pump and dump should produce equal-magnitude mean_rev components."""
    pump_base = [100 * (1.003 ** i) for i in range(99)]
    pump_base.append(pump_base[-1] * 1.15)   # +15% pump
    dump_base = [100 * (0.997 ** i) for i in range(99)]
    dump_base.append(dump_base[-1] * 0.85)   # -15% dump (symmetric)

    pump_mr = compute_neutral_score(make_ohlcv(pump_base))["components"]["mean_rev"]
    dump_mr = compute_neutral_score(make_ohlcv(dump_base))["components"]["mean_rev"]
    # Pump → negative mean_rev; dump → positive mean_rev; magnitudes should match
    assert abs(pump_mr + dump_mr) < 0.15, (
        f"mean_rev asymmetric: pump={pump_mr}, dump={dump_mr}"
    )


def test_components_bounded_for_all_fixtures(bullish_trend, bearish_trend, sideways, trend_then_pump):
    """All component values must stay in [-1, +1] for every fixture."""
    for name, df in [
        ("bullish", bullish_trend),
        ("bearish", bearish_trend),
        ("sideways", sideways),
        ("pump",    trend_then_pump),
    ]:
        result = compute_neutral_score(df)
        for comp, val in result["components"].items():
            assert -1.0 <= val <= 1.0, (
                f"[{name}] component {comp}={val} out of bounds"
            )


def test_score_with_exact_min_bars():
    """Exactly MIN_BARS bars should return a valid score (no warning)."""
    from neutral_score import MIN_BARS
    closes = [100 * (1.002 ** i) for i in range(MIN_BARS)]
    result = compute_neutral_score(make_ohlcv(closes))
    assert "warning" not in result
    assert -1.0 <= result["score"] <= 1.0


def test_constant_price_neutral():
    """Flat price series should produce a score near 0."""
    closes = [100.0] * 100
    result = compute_neutral_score(make_ohlcv(closes))
    assert abs(result["score"]) < 0.15, f"Flat price gave score={result['score']}"


def test_insufficient_data_components_are_zero():
    """When data is insufficient, all components must be 0.0."""
    result = compute_neutral_score(make_ohlcv([100, 101]))
    for key, val in result["components"].items():
        assert val == 0.0, f"Component {key} should be 0 for insufficient data, got {val}"
