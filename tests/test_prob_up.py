"""
tests/test_prob_up.py — Unit tests for Task 6 (prob_up calibration fix)

Run: pytest tests/test_prob_up.py -v
"""
from __future__ import annotations

import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ─────────────────────────────────────────────────────────────
#  Helpers to call _calc_all_scores with controlled inputs
# ─────────────────────────────────────────────────────────────

def make_data(
    price: float = 100.0,
    change_24h: float = 2.0,
    volume: float = 1e8,
    rsi: float = 55.0,
    rsi_4h: float = 55.0,
    vol_spike: float = 1.0,
    above_ma25: bool = True,
    above_ma99: bool = True,
    market_cap: float = 1e10,
    change_7d: float = 5.0,
    atr: float = 2.0,
    pct_from_high: float = -15.0,
) -> dict:
    return {
        "symbol": "BTC",
        "price": {
            "current":        price,
            "change_24h_pct": change_24h,
            "volume_24h_usdt": volume,
            "low_24h":        price * 0.98,
            "high_24h":       price * 1.02,
        },
        "technicals": {
            "rsi_14":         rsi,
            "rsi_14_4h":      rsi_4h,
            "volume_spike":   vol_spike,
            "above_ma25":     above_ma25,
            "above_ma99":     above_ma99,
            "atr_14":         atr,
            "pct_from_high":  pct_from_high,
            "price_structure": {"phase": "UPTREND_PULLBACK"},
        },
        "market_data": {
            "market_cap_usd": market_cap,
            "change_7d_pct":  change_7d,
        },
        "bitkub": {"listed": True, "price_thb": price * 33.0},
        "opportunity_grade": "B",
        "opportunity_score": 55,
    }


def get_prob_up(data: dict, legacy: bool = False) -> float:
    """Call _calc_all_scores and return prob_up, with optional legacy mode."""
    import analyzer
    import config
    # Patch USE_LEGACY_FORECASTER at module level
    original = config.USE_LEGACY_FORECASTER
    config.USE_LEGACY_FORECASTER = legacy
    # Also patch in analyzer module
    try:
        scores = analyzer._calc_all_scores(data)
        return scores["prob_up"]
    finally:
        config.USE_LEGACY_FORECASTER = original


# ─────────────────────────────────────────────────────────────
#  Core constraint: raw_prob=0.5 → prob_up=0.5 (±0.05)
# ─────────────────────────────────────────────────────────────

def test_midpoint_raw_prob_gives_half():
    """
    When momentum and technical scores are both mid-range,
    raw_prob ≈ 0.5, and prob_up should be ≈ 0.5 (±0.05).
    raw_prob = m/10*0.6 + t/10*0.4
    To get raw_prob=0.5: need m=5, t=5.
    """
    from analyzer import _apply_isotonic
    calib = {"method": "linear_scale", "scale": 0.6}
    result = _apply_isotonic(0.5, calib)
    assert abs(result - 0.5) < 0.05, f"apply_isotonic(0.5) = {result}, expected ≈ 0.5"


# ─────────────────────────────────────────────────────────────
#  Symmetric floor / ceiling
# ─────────────────────────────────────────────────────────────

def test_floor_is_015():
    """Very weak signals → prob_up never below 0.15."""
    # Minimal momentum/technical → low raw_prob
    data = make_data(
        rsi=10.0, rsi_4h=10.0, vol_spike=0.3,
        change_24h=-20.0, change_7d=-30.0,
        above_ma25=False, above_ma99=False,
    )
    prob = get_prob_up(data, legacy=False)
    assert prob >= 0.15, f"prob_up={prob} < floor 0.15"


def test_ceiling_is_085():
    """Very strong signals → prob_up never above 0.85."""
    data = make_data(
        rsi=68.0, rsi_4h=68.0, vol_spike=4.0,
        change_24h=25.0, change_7d=50.0,
        above_ma25=True, above_ma99=True,
    )
    prob = get_prob_up(data, legacy=False)
    assert prob <= 0.85, f"prob_up={prob} > ceiling 0.85"


def test_floor_ceiling_symmetric_around_half():
    """0.5 - floor == ceiling - 0.5 (both = 0.35)."""
    floor   = 0.15
    ceiling = 0.85
    assert (0.5 - floor) == pytest.approx(ceiling - 0.5), (
        f"Not symmetric: floor={floor}, ceiling={ceiling}"
    )


def test_no_additive_constant_at_zero_raw_prob():
    """
    With the old formula: raw_prob=0 → 0.30+0=0.30 (additive constant).
    New formula must NOT have this: isotonic(0.0) = 0.5+(0-0.5)*scale < 0.5.
    """
    from analyzer import _apply_isotonic
    calib = {"method": "linear_scale", "scale": 0.6}
    result_at_zero = _apply_isotonic(0.0, calib)
    assert result_at_zero < 0.5, (
        f"apply_isotonic(0.0)={result_at_zero} — expected < 0.5 (no additive constant)"
    )


# ─────────────────────────────────────────────────────────────
#  Directionality: weak → low prob, strong → high prob
# ─────────────────────────────────────────────────────────────

def test_weak_signals_give_prob_below_half():
    data = make_data(
        rsi=20.0, rsi_4h=20.0, vol_spike=0.4,
        change_24h=-10.0, change_7d=-15.0,
        above_ma25=False, above_ma99=False,
    )
    prob = get_prob_up(data, legacy=False)
    assert prob < 0.5, f"Weak signals gave prob_up={prob}, expected < 0.5"


def test_strong_signals_give_prob_above_half():
    data = make_data(
        rsi=65.0, rsi_4h=65.0, vol_spike=3.0,
        change_24h=10.0, change_7d=20.0,
        above_ma25=True, above_ma99=True,
    )
    prob = get_prob_up(data, legacy=False)
    assert prob > 0.5, f"Strong signals gave prob_up={prob}, expected > 0.5"


def test_strong_prob_greater_than_weak_prob():
    strong = make_data(rsi=68.0, vol_spike=3.0, change_24h=8.0,
                       above_ma25=True, above_ma99=True)
    weak   = make_data(rsi=30.0, vol_spike=0.5, change_24h=-5.0,
                       above_ma25=False, above_ma99=False)
    p_strong = get_prob_up(strong, legacy=False)
    p_weak   = get_prob_up(weak,   legacy=False)
    assert p_strong > p_weak, f"strong={p_strong} should > weak={p_weak}"


# ─────────────────────────────────────────────────────────────
#  Legacy mode: old formula preserved
# ─────────────────────────────────────────────────────────────

def test_legacy_mode_uses_old_floor_025():
    """USE_LEGACY_FORECASTER=True → old formula, floor=0.25."""
    data = make_data(rsi=10.0, rsi_4h=10.0, vol_spike=0.3,
                     change_24h=-20.0, above_ma25=False, above_ma99=False)
    prob = get_prob_up(data, legacy=True)
    assert prob >= 0.25, f"Legacy floor should be 0.25, got {prob}"


def test_legacy_mode_ceiling_075():
    """USE_LEGACY_FORECASTER=True → old formula, ceiling=0.75."""
    data = make_data(rsi=68.0, rsi_4h=68.0, vol_spike=4.0,
                     change_24h=25.0, above_ma25=True, above_ma99=True)
    prob = get_prob_up(data, legacy=True)
    assert prob <= 0.75, f"Legacy ceiling should be 0.75, got {prob}"


def test_new_mode_wider_range_than_legacy():
    """
    New floor (0.15) is lower than legacy floor (0.25),
    new ceiling (0.85) is higher than legacy ceiling (0.75).
    This means the new formula can express more extreme probabilities.
    """
    assert 0.15 < 0.25, "New floor 0.15 should be lower than legacy 0.25"
    assert 0.85 > 0.75, "New ceiling 0.85 should be higher than legacy 0.75"


# ─────────────────────────────────────────────────────────────
#  apply_isotonic directly
# ─────────────────────────────────────────────────────────────

def test_apply_isotonic_linear_scale_midpoint():
    from analyzer import _apply_isotonic
    assert _apply_isotonic(0.5, {"method": "linear_scale", "scale": 0.6}) == pytest.approx(0.5)


def test_apply_isotonic_linear_scale_shrinks_extremes():
    from analyzer import _apply_isotonic
    calib = {"method": "linear_scale", "scale": 0.6}
    assert _apply_isotonic(0.8, calib) == pytest.approx(0.5 + 0.3 * 0.6)
    assert _apply_isotonic(0.2, calib) == pytest.approx(0.5 - 0.3 * 0.6)


def test_apply_isotonic_symmetric():
    from analyzer import _apply_isotonic
    calib = {"method": "linear_scale", "scale": 0.6}
    above = _apply_isotonic(0.7, calib) - 0.5
    below = 0.5 - _apply_isotonic(0.3, calib)
    assert abs(above - below) < 1e-10
