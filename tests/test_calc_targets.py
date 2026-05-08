"""
tests/test_calc_targets.py — Unit tests for Task 5 (_calc_targets_new)

Run: pytest tests/test_calc_targets.py -v
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analyzer import _calc_targets_new, _signal_to_direction


# ─────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────

def make_data(price=100.0, atr=2.0, symbol="BTC") -> dict:
    return {
        "symbol": symbol,
        "price":      {"current": price},
        "technicals": {"atr_14": atr},
    }


def make_calib(k=2.5, j=1.5, calib_atr=None, symbol="BTC") -> dict:
    calib = {
        "atr_multipliers": {"target": k, "stop": j},
        "per_coin": {},
    }
    if calib_atr is not None:
        calib["per_coin"][symbol.upper()] = {"atr_14_avg": calib_atr}
    return calib


# ─────────────────────────────────────────────────────────────
#  Basic structure
# ─────────────────────────────────────────────────────────────

def test_returns_dict_with_required_keys():
    result = _calc_targets_new(make_data(), "long", make_calib())
    for key in ("entry", "target", "target_pct", "inval", "inval_pct", "rr", "atr", "direction"):
        assert key in result, f"Missing key: {key}"


def test_no_hardcoded_timeframe_labels():
    """No 'ชม.' or 'วัน' timeframe strings anywhere in the result."""
    result = _calc_targets_new(make_data(), "long", make_calib())
    result_str = str(result)
    assert "ชม." not in result_str
    assert "ภายใน" not in result_str


# ─────────────────────────────────────────────────────────────
#  Direction: long
# ─────────────────────────────────────────────────────────────

def test_long_target_above_entry():
    r = _calc_targets_new(make_data(price=100.0, atr=2.0), "long", make_calib(k=2.5, j=1.5))
    assert r["target"] > r["entry"]


def test_long_inval_below_entry():
    r = _calc_targets_new(make_data(price=100.0, atr=2.0), "long", make_calib(k=2.5, j=1.5))
    assert r["inval"] < r["entry"]


def test_long_target_price_correct():
    # entry=100, atr=2, k=2.5 → target = 100 + 2.5×2 = 105
    r = _calc_targets_new(make_data(price=100.0, atr=2.0), "long", make_calib(k=2.5))
    assert r["target"] == pytest.approx(105.0)


def test_long_inval_price_correct():
    # entry=100, atr=2, j=1.5 → inval = 100 - 1.5×2 = 97
    r = _calc_targets_new(make_data(price=100.0, atr=2.0), "long", make_calib(j=1.5))
    assert r["inval"] == pytest.approx(97.0)


# ─────────────────────────────────────────────────────────────
#  Direction: short
# ─────────────────────────────────────────────────────────────

def test_short_target_below_entry():
    r = _calc_targets_new(make_data(price=100.0, atr=2.0), "short", make_calib(k=2.5, j=1.5))
    assert r["target"] < r["entry"]


def test_short_inval_above_entry():
    r = _calc_targets_new(make_data(price=100.0, atr=2.0), "short", make_calib(k=2.5, j=1.5))
    assert r["inval"] > r["entry"]


def test_short_target_price_correct():
    # entry=100, atr=2, k=2.5 → target = 100 - 2.5×2 = 95
    r = _calc_targets_new(make_data(price=100.0, atr=2.0), "short", make_calib(k=2.5))
    assert r["target"] == pytest.approx(95.0)


# ─────────────────────────────────────────────────────────────
#  R:R calculation
# ─────────────────────────────────────────────────────────────

def test_rr_equals_k_over_j():
    """R:R = reward/risk = k×ATR / j×ATR = k/j"""
    k, j = 2.5, 1.5
    r = _calc_targets_new(make_data(price=100.0, atr=3.0), "long", make_calib(k=k, j=j))
    assert r["rr"] == pytest.approx(k / j, abs=0.1)   # rounded to 1 decimal


def test_rr_below_1_5_returns_neutral_action():
    """k/j < 1.5 → action='neutral' with reason."""
    calib = make_calib(k=1.0, j=1.0)   # R:R = 1.0 < 1.5
    r = _calc_targets_new(make_data(), "long", calib)
    assert r["action"] == "neutral"
    assert "reason" in r
    assert r["rr"] < 1.5


def test_rr_above_1_5_returns_directional_action():
    calib = make_calib(k=2.5, j=1.5)   # R:R = 1.67 > 1.5
    r = _calc_targets_new(make_data(), "long", calib)
    assert r["action"] == "long"


# ─────────────────────────────────────────────────────────────
#  Calibration multipliers
# ─────────────────────────────────────────────────────────────

def test_uses_calibration_k_multiplier():
    calib_k3 = make_calib(k=3.0)
    calib_k2 = make_calib(k=2.0)
    r3 = _calc_targets_new(make_data(atr=2.0), "long", calib_k3)
    r2 = _calc_targets_new(make_data(atr=2.0), "long", calib_k2)
    assert r3["target"] > r2["target"]


def test_uses_calibration_j_multiplier():
    calib_j2 = make_calib(j=2.0)
    calib_j1 = make_calib(j=1.0)
    rj2 = _calc_targets_new(make_data(atr=2.0), "long", calib_j2)
    rj1 = _calc_targets_new(make_data(atr=2.0), "long", calib_j1)
    assert rj2["inval"] < rj1["inval"]   # bigger stop = lower invalidation price


def test_default_multipliers_k_25_j_15():
    calib = {"atr_multipliers": {"target": 2.5, "stop": 1.5}, "per_coin": {}}
    r = _calc_targets_new(make_data(price=100.0, atr=2.0), "long", calib)
    assert r["target"] == pytest.approx(105.0)
    assert r["inval"]  == pytest.approx(97.0)


# ─────────────────────────────────────────────────────────────
#  ATR sanity cap
# ─────────────────────────────────────────────────────────────

def test_atr_sanity_cap_applied():
    """live_atr > 3×calib_atr → capped to 3×calib_atr."""
    calib = make_calib(k=2.5, j=1.5, calib_atr=2.0, symbol="BTC")
    # live_atr = 20 > 3×2 = 6 → should be capped at 6
    data  = make_data(price=100.0, atr=20.0, symbol="BTC")
    r     = _calc_targets_new(data, "long", calib)
    capped_atr = 3.0 * 2.0   # 6.0
    assert r["atr"] == pytest.approx(capped_atr)
    assert r["target"] == pytest.approx(100.0 + 2.5 * capped_atr)


def test_normal_atr_not_capped():
    """live_atr <= 3×calib_atr → not capped."""
    calib = make_calib(k=2.5, j=1.5, calib_atr=5.0, symbol="BTC")
    data  = make_data(price=100.0, atr=3.0, symbol="BTC")
    r     = _calc_targets_new(data, "long", calib)
    assert r["atr"] == pytest.approx(3.0)


def test_missing_calib_atr_no_cap():
    """No calibration ATR for coin → use live ATR as-is."""
    calib = make_calib(k=2.5, j=1.5)   # no per_coin atr
    data  = make_data(price=100.0, atr=50.0, symbol="UNKNOWN")
    r     = _calc_targets_new(data, "long", calib)
    assert r["atr"] == pytest.approx(50.0)


# ─────────────────────────────────────────────────────────────
#  _signal_to_direction
# ─────────────────────────────────────────────────────────────

def test_buy_signals_map_to_long():
    assert _signal_to_direction("Buy")        == "long"
    assert _signal_to_direction("Strong Buy") == "long"


def test_sell_signals_map_to_short():
    assert _signal_to_direction("Sell")        == "short"
    assert _signal_to_direction("Strong Sell") == "short"


def test_hold_maps_to_neutral():
    assert _signal_to_direction("Hold") == "neutral"


def test_unknown_signal_defaults_to_neutral():
    assert _signal_to_direction("???") == "neutral"
