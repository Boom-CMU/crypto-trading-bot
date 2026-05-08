"""
tests/test_calc_expected_pct.py — Unit tests for Task 3 (refactored calc_expected_pct)

Run: pytest tests/test_calc_expected_pct.py -v
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtest import calc_expected_pct, _calc_expected_pct_legacy


# ─────────────────────────────────────────────────────────────
#  Minimal calibration fixture (no file I/O)
# ─────────────────────────────────────────────────────────────

def make_calib(sigma_daily=0.025, sigma_1d=0.025, sigma_3d=0.043,
               sigma_7d=0.067, sigma_14d=0.091) -> dict:
    coin_data = {
        "sigma_daily": sigma_daily,
        "sigma_1d":    sigma_1d,
        "sigma_3d":    sigma_3d,
        "sigma_7d":    sigma_7d,
        "sigma_14d":   sigma_14d,
        "atr_14_avg":  1850.0,
    }
    return {
        "per_coin": {"BTC": coin_data},
        "_global_fallback": {
            "sigma_1d": 0.030, "sigma_3d": 0.050,
            "sigma_7d": 0.075, "sigma_14d": 0.105,
        },
        "prob_up_calibration": {"method": "linear_scale", "scale": 0.6},
        "atr_multipliers": {"target": 2.5, "stop": 1.5},
        "veto_threshold": 0.4,
    }


# ─────────────────────────────────────────────────────────────
#  Return structure
# ─────────────────────────────────────────────────────────────

def test_returns_dict_with_upper_and_lower():
    result = calc_expected_pct("BTC", 3, rsi=55, return_24h_frac=0.01,
                               calibration=make_calib())
    assert isinstance(result, dict)
    assert "upper" in result
    assert "lower" in result


def test_upper_is_positive_lower_is_negative():
    result = calc_expected_pct("BTC", 3, rsi=55, return_24h_frac=0.0,
                               calibration=make_calib())
    assert result["upper"] > 0
    assert result["lower"] < 0


# ─────────────────────────────────────────────────────────────
#  Symmetric bounds (no dampening)
# ─────────────────────────────────────────────────────────────

def test_symmetric_bounds_no_dampening():
    """Without any dampening triggers, upper == |lower|."""
    result = calc_expected_pct("BTC", 3, rsi=55, return_24h_frac=0.005,
                               calibration=make_calib())
    assert abs(result["upper"]) == pytest.approx(abs(result["lower"]), abs=0.01)


def test_bounds_scale_with_sigma():
    """Larger sigma coin → wider bounds."""
    calib_wide   = make_calib(sigma_3d=0.080)
    calib_narrow = make_calib(sigma_3d=0.020)
    wide   = calc_expected_pct("BTC", 3, rsi=55, return_24h_frac=0.005, calibration=calib_wide)
    narrow = calc_expected_pct("BTC", 3, rsi=55, return_24h_frac=0.005, calibration=calib_narrow)
    assert wide["upper"] > narrow["upper"]
    assert abs(wide["lower"]) > abs(narrow["lower"])


def test_bounds_equal_2_sigma_times_100():
    """Upper = 2 × sigma_3d × 100 when no dampening."""
    sigma_3d = 0.043
    calib    = make_calib(sigma_3d=sigma_3d, sigma_daily=0.025)
    # No dampening: return_24h_frac=0.005 (< 1.5×0.025=0.0375), rsi=55
    result = calc_expected_pct("BTC", 3, rsi=55, return_24h_frac=0.005, calibration=calib)
    expected_upper = 2.0 * sigma_3d * 100
    assert result["upper"] == pytest.approx(expected_upper, abs=0.05)


# ─────────────────────────────────────────────────────────────
#  Mean-reversion dampening
# ─────────────────────────────────────────────────────────────

def test_mean_reversion_halves_bounds():
    """When |return_24h| > 1.5×sigma_daily, both bounds shrink by ×0.5."""
    calib = make_calib(sigma_daily=0.020, sigma_3d=0.043)
    # Trigger: |return| = 0.05 > 1.5×0.020 = 0.030
    undamped = calc_expected_pct("BTC", 3, rsi=55, return_24h_frac=0.005, calibration=calib)
    damped   = calc_expected_pct("BTC", 3, rsi=55, return_24h_frac=0.050, calibration=calib)
    assert damped["upper"] == pytest.approx(undamped["upper"] * 0.5, abs=0.05)
    assert damped["lower"] == pytest.approx(undamped["lower"] * 0.5, abs=0.05)


def test_mean_reversion_symmetric_for_positive_and_negative_returns():
    """Positive and negative equal-magnitude 24h returns produce same dampening."""
    calib = make_calib(sigma_daily=0.020, sigma_3d=0.043)
    pos = calc_expected_pct("BTC", 3, rsi=55, return_24h_frac=+0.06, calibration=calib)
    neg = calc_expected_pct("BTC", 3, rsi=55, return_24h_frac=-0.06, calibration=calib)
    assert pos["upper"] == pytest.approx(neg["upper"], abs=0.01)
    assert pos["lower"] == pytest.approx(neg["lower"], abs=0.01)


def test_no_mean_reversion_below_threshold():
    """Small return (< 1.5σ) must NOT trigger dampening."""
    calib    = make_calib(sigma_daily=0.025, sigma_3d=0.043)
    baseline = calc_expected_pct("BTC", 3, rsi=55, return_24h_frac=0.0, calibration=calib)
    small    = calc_expected_pct("BTC", 3, rsi=55, return_24h_frac=0.010, calibration=calib)
    # 0.010 < 1.5×0.025=0.0375 — no dampening
    assert small["upper"] == pytest.approx(baseline["upper"], abs=0.01)


# ─────────────────────────────────────────────────────────────
#  RSI dampening — symmetric, only reduces, never boosts
# ─────────────────────────────────────────────────────────────

def test_rsi_overbought_reduces_upper():
    """RSI > 70 must shrink upper (upside dampened), lower unchanged."""
    calib  = make_calib(sigma_3d=0.043, sigma_daily=0.025)
    normal = calc_expected_pct("BTC", 3, rsi=55, return_24h_frac=0.005, calibration=calib)
    hot    = calc_expected_pct("BTC", 3, rsi=75, return_24h_frac=0.005, calibration=calib)
    assert hot["upper"] < normal["upper"]
    assert hot["lower"] == pytest.approx(normal["lower"], abs=0.01)


def test_rsi_oversold_reduces_lower_magnitude():
    """RSI < 30 must shrink |lower| (downside dampened), upper unchanged."""
    calib     = make_calib(sigma_3d=0.043, sigma_daily=0.025)
    normal    = calc_expected_pct("BTC", 3, rsi=55,  return_24h_frac=0.005, calibration=calib)
    oversold  = calc_expected_pct("BTC", 3, rsi=25,  return_24h_frac=0.005, calibration=calib)
    assert abs(oversold["lower"]) < abs(normal["lower"])
    assert oversold["upper"] == pytest.approx(normal["upper"], abs=0.01)


def test_rsi_dampening_scale_factor():
    """RSI > 70 applies 0.85× to upper; RSI < 30 applies 0.85× to lower."""
    calib  = make_calib(sigma_3d=0.043, sigma_daily=0.025)
    base   = calc_expected_pct("BTC", 3, rsi=55, return_24h_frac=0.0, calibration=calib)
    hot    = calc_expected_pct("BTC", 3, rsi=75, return_24h_frac=0.0, calibration=calib)
    cold   = calc_expected_pct("BTC", 3, rsi=20, return_24h_frac=0.0, calibration=calib)
    assert hot["upper"]   == pytest.approx(base["upper"]  * 0.85, abs=0.05)
    assert cold["lower"]  == pytest.approx(base["lower"]  * 0.85, abs=0.05)


def test_no_rsi_boost_at_neutral():
    """Neutral RSI (50) must give identical result to RSI=None (no boost allowed)."""
    calib    = make_calib(sigma_3d=0.043, sigma_daily=0.025)
    no_rsi   = calc_expected_pct("BTC", 3, rsi=None, return_24h_frac=0.0, calibration=calib)
    mid_rsi  = calc_expected_pct("BTC", 3, rsi=50,   return_24h_frac=0.0, calibration=calib)
    assert mid_rsi["upper"] == pytest.approx(no_rsi["upper"], abs=0.01)
    assert mid_rsi["lower"] == pytest.approx(no_rsi["lower"], abs=0.01)


# ─────────────────────────────────────────────────────────────
#  No code path produces |upper| > |lower| without justification
# ─────────────────────────────────────────────────────────────

def test_no_upper_exceeds_lower_without_rsi_justification():
    """Without RSI < 30 trigger, |upper| must never exceed |lower|."""
    calib = make_calib(sigma_3d=0.043, sigma_daily=0.025)
    for rsi in [None, 35, 50, 55, 65, 70, 72]:
        result = calc_expected_pct("BTC", 3, rsi=rsi,
                                   return_24h_frac=0.005, calibration=calib)
        assert result["upper"] <= abs(result["lower"]) + 0.01, (
            f"RSI={rsi}: upper={result['upper']} > |lower|={abs(result['lower'])}"
        )


def test_upper_can_exceed_lower_when_oversold():
    """RSI < 30 (oversold) is the only justified case for |upper| > |lower|."""
    calib  = make_calib(sigma_3d=0.043, sigma_daily=0.025)
    result = calc_expected_pct("BTC", 3, rsi=20, return_24h_frac=0.0, calibration=calib)
    assert result["upper"] > abs(result["lower"]), (
        "RSI<30 should give |upper| > |lower| (downside dampened)"
    )


# ─────────────────────────────────────────────────────────────
#  Unknown coin fallback
# ─────────────────────────────────────────────────────────────

def test_unknown_coin_uses_global_fallback():
    """A coin not in per_coin must still produce a valid range (global fallback)."""
    calib  = make_calib()   # only has "BTC"
    result = calc_expected_pct("UNKNOWN_COIN", 3, rsi=55,
                               return_24h_frac=0.005, calibration=calib)
    assert result["upper"] > 0
    assert result["lower"] < 0


def test_unknown_coin_range_uses_fallback_sigma():
    """Unknown coin upper ≈ 2 × global_fallback_sigma_3d × 100."""
    calib        = make_calib()
    fallback_3d  = calib["_global_fallback"]["sigma_3d"]  # 0.050
    result       = calc_expected_pct("ALTCOIN", 3, rsi=55,
                                     return_24h_frac=0.0, calibration=calib)
    expected_upper = 2.0 * fallback_3d * 100
    assert result["upper"] == pytest.approx(expected_upper, abs=0.1)


# ─────────────────────────────────────────────────────────────
#  USE_LEGACY_FORECASTER flag
# ─────────────────────────────────────────────────────────────

def test_legacy_function_returns_float():
    """_calc_expected_pct_legacy must return a float, not a dict."""
    result = _calc_expected_pct_legacy(
        chg_24h=4.0, chg_7d=10.0, rsi_4h=65.0, vol_spike=1.5, horizon=3
    )
    assert isinstance(result, float)


def test_legacy_function_bullish_bias_exists():
    """Legacy function must produce positive value for positive recent move (known bias)."""
    result = _calc_expected_pct_legacy(
        chg_24h=5.0, chg_7d=10.0, rsi_4h=60.0, vol_spike=1.5, horizon=3
    )
    assert result > 0


def test_legacy_forecaster_flag_dispatch(monkeypatch):
    """When USE_LEGACY_FORECASTER=True, _calc_expected_and_horizon returns float."""
    monkeypatch.setenv("USE_LEGACY_FORECASTER", "1")
    import importlib
    import config
    importlib.reload(config)

    import analyzer
    importlib.reload(analyzer)

    # Build a minimal data dict
    data = {
        "symbol": "BTC",
        "price":       {"current": 77000, "change_24h_pct": 1.0, "volume_24h_usdt": 1e9,
                        "low_24h": 75000, "high_24h": 78000},
        "technicals":  {"rsi_14": 55.0, "rsi_14_4h": 52.0, "volume_spike": 1.0,
                        "above_ma25": True, "above_ma99": True, "atr_14": 2000,
                        "price_structure": {"phase": "UPTREND_PULLBACK"}},
        "market_data": {"market_cap_usd": 1.5e12, "change_7d_pct": 2.0},
        "bitkub":      {"listed": True, "price_thb": 2_500_000},
        "opportunity_grade": "C", "opportunity_score": 50,
    }

    # Import the function from the reloaded module
    from analyzer import _calc_expected_and_horizon
    expected, horizon, src = _calc_expected_and_horizon(data)
    assert isinstance(expected, (float, type(None)))

    # Cleanup: restore env
    monkeypatch.delenv("USE_LEGACY_FORECASTER", raising=False)
    importlib.reload(config)
    importlib.reload(analyzer)
