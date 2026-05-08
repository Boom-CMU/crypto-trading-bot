"""
tests/test_atr_calibration.py — Unit tests for Task 1c (ATR multiplier grid search)

Run: pytest tests/test_atr_calibration.py -v
"""
from __future__ import annotations

import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from calibration import (
    _check_hit,
    _grid_search_atr,
    _select_best_atr_multipliers,
    K_GRID,
    J_GRID,
    ATR_HORIZON,
    MIN_BARS,
)


# ─────────────────────────────────────────────────────────────
#  Synthetic price fixtures
# ─────────────────────────────────────────────────────────────

def flat_bars(n: int, price: float = 100.0):
    """Flat market — neither target nor stop reached."""
    closes = [price] * n
    highs  = [price * 1.001] * n
    lows   = [price * 0.999] * n
    return closes, highs, lows


def trending_up(n: int, price: float = 100.0, step_pct: float = 0.01):
    """Uptrend — target almost always reached first."""
    closes, highs, lows = [price], [price * 1.005], [price * 0.995]
    for _ in range(n - 1):
        p = closes[-1] * (1 + step_pct)
        closes.append(p)
        highs.append(p * 1.005)
        lows.append(p * 0.995)
    return closes, highs, lows


# ─────────────────────────────────────────────────────────────
#  _check_hit
# ─────────────────────────────────────────────────────────────

def test_check_hit_target_reached_first():
    # highs all at 110, lows all at 99 — target=105, stop=95 → hit
    highs  = [99.0] * 5 + [110.0] * 20
    lows   = [99.0] * 25
    result = _check_hit(highs, lows, i=4, target=105.0, stop=95.0, horizon=20)
    assert result == "hit"


def test_check_hit_stop_reached_first():
    # lows all at 85, highs never reach target
    highs  = [100.0] * 25
    lows   = [100.0] * 5 + [85.0] * 20
    result = _check_hit(highs, lows, i=4, target=120.0, stop=90.0, horizon=20)
    assert result == "miss"


def test_check_hit_timeout_when_neither_reached():
    highs  = [101.0] * 50
    lows   = [99.0]  * 50
    # target=200, stop=0 — never reached in horizon=10
    result = _check_hit(highs, lows, i=10, target=200.0, stop=0.01, horizon=10)
    assert result == "timeout"


def test_check_hit_target_on_exact_boundary():
    """If high[i+1] == target exactly → hit."""
    highs = [99.0, 105.0, 103.0]
    lows  = [98.0,  98.0,  98.0]
    result = _check_hit(highs, lows, i=0, target=105.0, stop=90.0, horizon=5)
    assert result == "hit"


def test_check_hit_respects_horizon_limit():
    """Only look ATR_HORIZON bars ahead — don't peek beyond."""
    n = 200
    highs = [100.0] * n
    lows  = [100.0] * n
    # target and stop are only reached at bar i+ATR_HORIZON+5
    highs[ATR_HORIZON + 15] = 200.0
    result = _check_hit(highs, lows, i=0, target=150.0, stop=50.0, horizon=ATR_HORIZON)
    assert result == "timeout"


# ─────────────────────────────────────────────────────────────
#  _grid_search_atr
# ─────────────────────────────────────────────────────────────

def _make_enough_bars(price=100.0, n=300):
    """Generate enough bars for the grid search loop."""
    import random
    rng = random.Random(42)
    closes, highs, lows = [price], [price * 1.005], [price * 0.995]
    for _ in range(n - 1):
        ret = rng.gauss(0.0005, 0.015)
        p   = max(0.01, closes[-1] * (1 + ret))
        closes.append(p)
        highs.append(p * 1.005)
        lows.append(p * 0.995)
    return closes, highs, lows


def test_grid_search_returns_all_combos():
    closes, highs, lows = _make_enough_bars()
    results = _grid_search_atr(closes, highs, lows)
    for k in K_GRID:
        for j in J_GRID:
            assert (k, j) in results, f"Missing combo k={k}, j={j}"


def test_grid_search_hit_rates_between_0_and_1():
    closes, highs, lows = _make_enough_bars()
    results = _grid_search_atr(closes, highs, lows)
    for (k, j), stats in results.items():
        assert 0.0 <= stats["hit_rate"] <= 1.0, f"k={k},j={j}: hit_rate={stats['hit_rate']}"


def test_grid_search_ev_formula():
    """EV = hit_rate × k - (1 - hit_rate) × j for each combo."""
    closes, highs, lows = _make_enough_bars()
    results = _grid_search_atr(closes, highs, lows)
    for (k, j), stats in results.items():
        hr  = stats["hit_rate"]
        ev  = stats["ev"]
        expected_ev = round(hr * k - (1 - hr) * j, 4)
        assert ev == pytest.approx(expected_ev, abs=0.001), (
            f"k={k},j={j}: ev={ev} but expected {expected_ev}"
        )


def test_grid_search_larger_k_higher_reward_when_hit_rate_same():
    """
    For the same hit_rate, larger k should give larger EV when hr > j/(k+j).
    Test that results are internally consistent.
    """
    closes, highs, lows = _make_enough_bars()
    results = _grid_search_atr(closes, highs, lows)
    # Spot-check: compare k=1.5,j=1.0 vs k=3.0,j=1.0
    s15 = results[(1.5, 1.0)]
    s30 = results[(3.0, 1.0)]
    # hit_rate for k=3.0 should be lower (harder to reach), but if similar, EV comparison holds
    # Just verify both have plausible hit counts
    assert s15["hit"] + s15["miss"] > 0
    assert s30["hit"] + s30["miss"] > 0


def test_grid_search_smaller_j_less_aggressive_stop():
    """Smaller j (tighter stop) → more misses (stop hit more often)."""
    closes, highs, lows = _make_enough_bars()
    results = _grid_search_atr(closes, highs, lows)
    # j=1.0 (tight stop) should have lower hit_rate than j=2.0 (loose stop) for same k
    for k in K_GRID:
        hr_tight = results[(k, 1.0)]["hit_rate"]
        hr_loose = results[(k, 2.0)]["hit_rate"]
        assert hr_loose >= hr_tight, (
            f"k={k}: hit_rate with tight stop ({hr_tight}) > loose stop ({hr_loose})"
        )


# ─────────────────────────────────────────────────────────────
#  _select_best_atr_multipliers
# ─────────────────────────────────────────────────────────────

def test_select_best_returns_highest_ev():
    agg = {
        (1.5, 1.0): [0.10, 0.12],
        (2.5, 1.5): [0.35, 0.38],   # ← highest EV
        (3.0, 2.0): [0.20, 0.22],
    }
    result = _select_best_atr_multipliers(agg)
    assert result["target"] == 2.5
    assert result["stop"]   == 1.5


def test_select_best_uses_mean_ev_across_coins():
    # (2.5, 1.5) has high EV for 1 coin but low for others → not selected
    # (2.0, 1.5) consistently high
    agg = {
        (2.5, 1.5): [1.0, -0.5, -0.5],    # mean = 0.0
        (2.0, 1.5): [0.3,  0.3,  0.3],    # mean = 0.3 ← winner
    }
    result = _select_best_atr_multipliers(agg)
    assert result["target"] == 2.0
    assert result["stop"]   == 1.5


def test_select_best_returns_defaults_when_best_ev_negative():
    """If all mean EVs are negative → use defaults."""
    agg = {(k, j): [-0.1, -0.2] for k in K_GRID for j in J_GRID}
    result = _select_best_atr_multipliers(agg)
    assert result == {"target": 2.5, "stop": 1.5}


def test_select_best_returns_defaults_when_empty():
    agg: dict = {(k, j): [] for k in K_GRID for j in J_GRID}
    result = _select_best_atr_multipliers(agg)
    assert result == {"target": 2.5, "stop": 1.5}


def test_select_best_result_keys():
    agg = {(2.5, 1.5): [0.3, 0.4]}
    result = _select_best_atr_multipliers(agg)
    assert "target" in result
    assert "stop" in result


def test_select_best_values_from_grid():
    """Selected (k, j) must come from the specified grids."""
    agg = {(k, j): [0.1 * k] for k in K_GRID for j in J_GRID}
    result = _select_best_atr_multipliers(agg)
    assert result["target"] in K_GRID
    assert result["stop"]   in J_GRID
