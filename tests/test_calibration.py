"""
tests/test_calibration.py — Unit tests for calibration.py (Task 1a)

Run: pytest tests/test_calibration.py -v
"""
from __future__ import annotations

import math
import random
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from calibration import (
    _pct_returns,
    _rolling_std,
    _interpolate_sigma,
    _compute_sigma_for_coin,
    _compute_global_fallback,
    get_sigma,
    apply_isotonic,
    HORIZONS,
    MIN_BARS,
)


# ─────────────────────────────────────────────────────────────
#  Shared fixtures
# ─────────────────────────────────────────────────────────────

def make_closes(n: int, daily_return: float = 0.0, noise: float = 0.0) -> list[float]:
    """Synthetic price series. Seeded for reproducibility."""
    rng = random.Random(42)
    closes = [100.0]
    for _ in range(n - 1):
        r = daily_return + rng.gauss(0, noise)
        closes.append(max(0.01, closes[-1] * (1 + r)))
    return closes


def make_hl(closes: list[float], spread: float = 0.005) -> tuple[list[float], list[float]]:
    highs = [c * (1 + spread) for c in closes]
    lows  = [c * (1 - spread) for c in closes]
    return highs, lows


# ─────────────────────────────────────────────────────────────
#  _pct_returns
# ─────────────────────────────────────────────────────────────

def test_pct_returns_length_period_1():
    closes = [100.0, 102.0, 104.0, 103.0, 105.0]
    assert len(_pct_returns(closes, 1)) == 4


def test_pct_returns_length_period_3():
    closes = [100.0, 102.0, 104.0, 103.0, 105.0]
    assert len(_pct_returns(closes, 3)) == 2


def test_pct_returns_correct_value():
    closes = [100.0, 104.0]
    r = _pct_returns(closes, 1)
    assert r[0] == pytest.approx(0.04)


def test_pct_returns_constant_price_is_zero():
    closes = [100.0] * 20
    assert all(x == 0.0 for x in _pct_returns(closes, 1))


def test_pct_returns_period_3_correct_value():
    closes = [100.0, 101.0, 102.0, 103.0]
    r = _pct_returns(closes, 3)
    assert len(r) == 1
    assert r[0] == pytest.approx(0.03)


# ─────────────────────────────────────────────────────────────
#  _rolling_std
# ─────────────────────────────────────────────────────────────

def test_rolling_std_none_when_insufficient():
    assert _rolling_std([0.01, 0.02], window=90) is None


def test_rolling_std_constant_returns_zero():
    result = _rolling_std([0.01] * 100, window=90)
    assert result == pytest.approx(0.0, abs=1e-12)


def test_rolling_std_uses_only_last_window():
    # First 200 values are 0; last 90 values have std ~0.01
    rng = random.Random(0)
    tail = [rng.gauss(0, 0.01) for _ in range(90)]
    values = [0.0] * 200 + tail
    result = _rolling_std(values, window=90)
    assert result == pytest.approx(_rolling_std(tail, window=90))


def test_rolling_std_positive_for_noisy_data():
    rng = random.Random(1)
    values = [rng.gauss(0, 0.02) for _ in range(100)]
    result = _rolling_std(values, window=90)
    assert result is not None
    assert result > 0


# ─────────────────────────────────────────────────────────────
#  _interpolate_sigma
# ─────────────────────────────────────────────────────────────

SAMPLE_DATA = {
    "sigma_1d":  0.020,
    "sigma_3d":  0.040,
    "sigma_7d":  0.060,
    "sigma_14d": 0.090,
    "sigma_daily": 0.020,
}


def test_interpolate_exact_horizon_1():
    assert _interpolate_sigma(1, SAMPLE_DATA) == pytest.approx(0.020)


def test_interpolate_exact_horizon_3():
    assert _interpolate_sigma(3, SAMPLE_DATA) == pytest.approx(0.040)


def test_interpolate_exact_horizon_14():
    assert _interpolate_sigma(14, SAMPLE_DATA) == pytest.approx(0.090)


def test_interpolate_midpoint_between_3_and_7():
    # horizon=5: t=(5-3)/(7-3)=0.5 → 0.040 + 0.5*(0.060-0.040) = 0.050
    result = _interpolate_sigma(5, SAMPLE_DATA)
    assert result == pytest.approx(0.050)


def test_interpolate_clamps_below_minimum():
    assert _interpolate_sigma(0, SAMPLE_DATA) == pytest.approx(0.020)


def test_interpolate_clamps_above_maximum():
    assert _interpolate_sigma(30, SAMPLE_DATA) == pytest.approx(0.090)


def test_interpolate_result_between_brackets():
    result = _interpolate_sigma(10, SAMPLE_DATA)
    assert SAMPLE_DATA["sigma_7d"] < result < SAMPLE_DATA["sigma_14d"]


# ─────────────────────────────────────────────────────────────
#  _compute_sigma_for_coin
# ─────────────────────────────────────────────────────────────

def test_compute_sigma_returns_none_for_short_data():
    closes = make_closes(50)
    highs, lows = make_hl(closes)
    assert _compute_sigma_for_coin(closes, highs, lows) is None


def test_compute_sigma_returns_dict_for_sufficient_data():
    closes = make_closes(MIN_BARS + 20, daily_return=0.001, noise=0.02)
    highs, lows = make_hl(closes)
    result = _compute_sigma_for_coin(closes, highs, lows)
    assert result is not None


def test_compute_sigma_all_horizon_keys_present():
    closes = make_closes(MIN_BARS + 20, daily_return=0.001, noise=0.02)
    highs, lows = make_hl(closes)
    result = _compute_sigma_for_coin(closes, highs, lows)
    assert result is not None
    for h in HORIZONS:
        assert f"sigma_{h}d" in result
        assert result[f"sigma_{h}d"] > 0


def test_compute_sigma_daily_aliases_sigma_1d():
    closes = make_closes(MIN_BARS + 20, daily_return=0.001, noise=0.02)
    highs, lows = make_hl(closes)
    result = _compute_sigma_for_coin(closes, highs, lows)
    assert result is not None
    assert result["sigma_daily"] == result["sigma_1d"]


def test_compute_sigma_grows_with_horizon():
    """For random-walk-like data, realized vol should grow with horizon."""
    closes = make_closes(MIN_BARS + 50, daily_return=0.0, noise=0.025)
    highs, lows = make_hl(closes)
    result = _compute_sigma_for_coin(closes, highs, lows)
    assert result is not None
    assert result["sigma_3d"]  > result["sigma_1d"]
    assert result["sigma_7d"]  > result["sigma_3d"]
    assert result["sigma_14d"] > result["sigma_7d"]


def test_compute_sigma_atr_is_positive():
    closes = make_closes(MIN_BARS + 20, daily_return=0.001, noise=0.02)
    highs, lows = make_hl(closes)
    result = _compute_sigma_for_coin(closes, highs, lows)
    assert result is not None
    assert result["atr_14_avg"] is not None
    assert result["atr_14_avg"] > 0


def test_compute_sigma_constant_prices_near_zero():
    closes = [100.0] * (MIN_BARS + 20)
    highs  = [100.5] * (MIN_BARS + 20)
    lows   = [99.5]  * (MIN_BARS + 20)
    result = _compute_sigma_for_coin(closes, highs, lows)
    assert result is not None
    assert result["sigma_daily"] == pytest.approx(0.0, abs=1e-12)


def test_compute_sigma_includes_metadata():
    closes = make_closes(MIN_BARS + 20, daily_return=0.001, noise=0.02)
    highs, lows = make_hl(closes)
    result = _compute_sigma_for_coin(closes, highs, lows)
    assert result is not None
    assert "data_source" in result
    assert "computed_at" in result


# ─────────────────────────────────────────────────────────────
#  _compute_global_fallback
# ─────────────────────────────────────────────────────────────

def test_global_fallback_is_median_of_sigma_1d():
    per_coin = {
        "BTC": {"sigma_1d": 0.010, "sigma_3d": 0.020, "sigma_7d": 0.030, "sigma_14d": 0.040},
        "ETH": {"sigma_1d": 0.020, "sigma_3d": 0.040, "sigma_7d": 0.060, "sigma_14d": 0.080},
        "SOL": {"sigma_1d": 0.030, "sigma_3d": 0.060, "sigma_7d": 0.090, "sigma_14d": 0.120},
    }
    fallback = _compute_global_fallback(per_coin)
    assert fallback["sigma_1d"] == pytest.approx(0.020)   # median of [0.01, 0.02, 0.03]
    assert fallback["sigma_3d"] == pytest.approx(0.040)   # median of [0.02, 0.04, 0.06]


def test_global_fallback_has_all_horizons():
    per_coin = {
        "BTC": {"sigma_1d": 0.02, "sigma_3d": 0.04, "sigma_7d": 0.06, "sigma_14d": 0.09},
    }
    fallback = _compute_global_fallback(per_coin)
    for h in HORIZONS:
        assert f"sigma_{h}d" in fallback


def test_global_fallback_is_positive():
    per_coin = {
        "X": {"sigma_1d": 0.03, "sigma_3d": 0.05, "sigma_7d": 0.08, "sigma_14d": 0.11},
    }
    fallback = _compute_global_fallback(per_coin)
    assert all(v > 0 for v in fallback.values())


# ─────────────────────────────────────────────────────────────
#  get_sigma
# ─────────────────────────────────────────────────────────────

CALIB_FIXTURE = {
    "per_coin": {
        "BTC": {
            "sigma_daily": 0.020,
            "sigma_1d":    0.020,
            "sigma_3d":    0.040,
            "sigma_7d":    0.060,
            "sigma_14d":   0.090,
        },
    },
    "_global_fallback": {
        "sigma_1d":  0.030,
        "sigma_3d":  0.050,
        "sigma_7d":  0.075,
        "sigma_14d": 0.105,
    },
}


def test_get_sigma_known_coin_exact():
    assert get_sigma("BTC", 3, CALIB_FIXTURE) == pytest.approx(0.040)


def test_get_sigma_known_coin_interpolated():
    s = get_sigma("BTC", 5, CALIB_FIXTURE)
    assert CALIB_FIXTURE["per_coin"]["BTC"]["sigma_3d"] < s < CALIB_FIXTURE["per_coin"]["BTC"]["sigma_7d"]


def test_get_sigma_unknown_coin_uses_fallback():
    s = get_sigma("UNKNOWN_COIN", 3, CALIB_FIXTURE)
    assert s == pytest.approx(0.050)


def test_get_sigma_case_insensitive():
    assert get_sigma("btc", 3, CALIB_FIXTURE) == get_sigma("BTC", 3, CALIB_FIXTURE)


def test_get_sigma_returns_positive():
    assert get_sigma("BTC", 7, CALIB_FIXTURE) > 0
    assert get_sigma("MISSING", 7, CALIB_FIXTURE) > 0


# ─────────────────────────────────────────────────────────────
#  apply_isotonic
# ─────────────────────────────────────────────────────────────

def test_apply_isotonic_linear_midpoint_is_exact_half():
    """Core invariant: raw_prob=0.5 → output=0.5 exactly."""
    calib = {"method": "linear_scale", "scale": 0.6}
    assert apply_isotonic(0.5, calib) == pytest.approx(0.5)


def test_apply_isotonic_linear_shrinks_above_half():
    calib = {"method": "linear_scale", "scale": 0.6}
    # 0.5 + (0.7 - 0.5) * 0.6 = 0.62
    assert apply_isotonic(0.7, calib) == pytest.approx(0.62)


def test_apply_isotonic_linear_shrinks_below_half():
    calib = {"method": "linear_scale", "scale": 0.6}
    # 0.5 + (0.3 - 0.5) * 0.6 = 0.38
    assert apply_isotonic(0.3, calib) == pytest.approx(0.38)


def test_apply_isotonic_symmetric_around_half():
    """Shrinkage must be symmetric: distance above 0.5 == distance below 0.5."""
    calib = {"method": "linear_scale", "scale": 0.6}
    above = apply_isotonic(0.7, calib) - 0.5
    below = 0.5 - apply_isotonic(0.3, calib)
    assert abs(above - below) < 1e-10


def test_apply_isotonic_isotonic_mode_interpolates():
    calib = {
        "method": "isotonic",
        "knots":  [[0.50, 0.48], [0.60, 0.55], [0.70, 0.62], [0.80, 0.68]],
    }
    result = apply_isotonic(0.65, calib)
    assert 0.55 < result < 0.62


def test_apply_isotonic_isotonic_clamps_below():
    calib = {"method": "isotonic", "knots": [[0.5, 0.48], [0.7, 0.62]]}
    assert apply_isotonic(0.2, calib) == pytest.approx(0.48)


def test_apply_isotonic_isotonic_clamps_above():
    calib = {"method": "isotonic", "knots": [[0.5, 0.48], [0.7, 0.62]]}
    assert apply_isotonic(0.9, calib) == pytest.approx(0.62)


def test_apply_isotonic_unknown_method_passthrough():
    calib = {"method": "future_method_v3"}
    assert apply_isotonic(0.65, calib) == pytest.approx(0.65)
