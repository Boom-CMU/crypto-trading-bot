"""
tests/test_reconciliation.py — Unit tests for Task 7 (Reconciliation Gate)

Run: pytest tests/test_reconciliation.py -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from reconciliation import reconcile, VETO_LOG_FILE


# ─────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────

def make_forecast(
    direction="long",
    confidence=0.70,
    symbol="BTC",
    reasoning="test",
    bull_prob=0.45,
    bear_prob=0.35,
    base_prob=0.20,
) -> dict:
    return {
        "symbol":            symbol,
        "direction":         direction,
        "confidence":        confidence,
        "reasoning":         reasoning,
        "bull_case":         {"thesis": "up", "key_evidence": [], "probability": bull_prob},
        "bear_case":         {"thesis": "dn", "key_evidence": [], "probability": bear_prob},
        "base_case":         {"thesis": "flat",                    "probability": base_prob},
        "target_price":      110.0,
        "invalidation_price": 90.0,
    }


def make_calib(veto_threshold=0.4) -> dict:
    return {
        "veto_threshold":    veto_threshold,
        "atr_multipliers":   {"target": 2.5, "stop": 1.5},
        "prob_up_calibration": {"method": "linear_scale", "scale": 0.6},
    }


# ─────────────────────────────────────────────────────────────
#  Veto: long blocked by bearish structure
# ─────────────────────────────────────────────────────────────

def test_veto_long_bearish_structure():
    result = reconcile(make_forecast("long", 0.70), neutral_score=-0.55, calibration=make_calib(0.4))
    assert result["action"] == "HOLD"
    assert "bearish" in result["reason"]


def test_veto_long_requires_confidence_above_06():
    """Confidence ≤ 0.6 → no veto even with strongly bearish structure."""
    result = reconcile(make_forecast("long", 0.60), neutral_score=-0.55, calibration=make_calib(0.4))
    assert result["action"] != "HOLD"


def test_veto_long_requires_neutral_below_neg_threshold():
    """neutral_score just above -threshold → no veto."""
    result = reconcile(make_forecast("long", 0.70), neutral_score=-0.39, calibration=make_calib(0.4))
    assert result["action"] != "HOLD"


# ─────────────────────────────────────────────────────────────
#  Veto: short blocked by bullish structure
# ─────────────────────────────────────────────────────────────

def test_veto_short_bullish_structure():
    result = reconcile(make_forecast("short", 0.70), neutral_score=+0.55, calibration=make_calib(0.4))
    assert result["action"] == "HOLD"
    assert "bullish" in result["reason"]


def test_veto_short_requires_confidence_above_06():
    result = reconcile(make_forecast("short", 0.60), neutral_score=+0.55, calibration=make_calib(0.4))
    assert result["action"] != "HOLD"


# ─────────────────────────────────────────────────────────────
#  Veto symmetry
# ─────────────────────────────────────────────────────────────

def test_veto_symmetric_long_vs_short():
    """Long vetoed by -0.5 mirror equals short vetoed by +0.5."""
    long_veto  = reconcile(make_forecast("long",  0.70), -0.50, make_calib())
    short_veto = reconcile(make_forecast("short", 0.70), +0.50, make_calib())
    assert long_veto["action"]  == "HOLD"
    assert short_veto["action"] == "HOLD"


def test_no_veto_for_neutral_direction():
    """direction='neutral' should never trigger a veto."""
    f = make_forecast("neutral", 0.70, bull_prob=0.33, bear_prob=0.33, base_prob=0.34)
    result = reconcile(f, neutral_score=-0.90, calibration=make_calib())
    assert result["action"] != "HOLD"


# ─────────────────────────────────────────────────────────────
#  Confidence reduction for mild disagreement
# ─────────────────────────────────────────────────────────────

def test_confidence_reduced_on_mild_disagreement():
    """long + neutral_score slightly negative (within threshold) → confidence × 0.6."""
    f = make_forecast("long", 0.70)
    result = reconcile(f, neutral_score=-0.20, calibration=make_calib(veto_threshold=0.4))
    assert result["action"] != "HOLD"
    assert result["confidence"] == pytest.approx(0.70 * 0.6, abs=0.001)


def test_confidence_reduced_adds_note_to_reasoning():
    f = make_forecast("long", 0.70)
    result = reconcile(f, neutral_score=-0.20, calibration=make_calib())
    assert "confidence reduced" in result.get("reasoning", "")


def test_no_confidence_reduction_when_aligned():
    """long + bullish neutral_score → no reduction."""
    f = make_forecast("long", 0.70)
    result = reconcile(f, neutral_score=+0.30, calibration=make_calib())
    assert result["confidence"] == pytest.approx(0.70)
    assert "confidence reduced" not in result.get("reasoning", "")


def test_no_confidence_reduction_when_neutral_score_near_zero():
    """Neutral score in (-0.1, +0.1) → no sign conflict, no reduction."""
    f = make_forecast("long", 0.70)
    result = reconcile(f, neutral_score=0.05, calibration=make_calib())
    assert result["confidence"] == pytest.approx(0.70)


# ─────────────────────────────────────────────────────────────
#  Pass-through (no veto, no reduction)
# ─────────────────────────────────────────────────────────────

def test_passthrough_long_bullish_structure():
    result = reconcile(make_forecast("long", 0.70), neutral_score=+0.30, calibration=make_calib())
    assert result["action"] == "LONG"


def test_passthrough_preserves_all_ai_fields():
    f = make_forecast("long", 0.65)
    result = reconcile(f, neutral_score=+0.20, calibration=make_calib())
    assert result["target_price"] == f["target_price"]
    assert result["invalidation_price"] == f["invalidation_price"]
    assert result["neutral_score"] == pytest.approx(+0.20)


# ─────────────────────────────────────────────────────────────
#  Input immutability
# ─────────────────────────────────────────────────────────────

def test_input_forecast_not_mutated():
    """reconcile() must not modify the original ai_forecast dict."""
    f = make_forecast("long", 0.70)
    orig_confidence = f["confidence"]
    orig_reasoning  = f["reasoning"]
    reconcile(f, neutral_score=-0.20, calibration=make_calib())
    assert f["confidence"] == orig_confidence
    assert f["reasoning"]  == orig_reasoning


# ─────────────────────────────────────────────────────────────
#  Veto log file
# ─────────────────────────────────────────────────────────────

def test_veto_writes_to_log(tmp_path, monkeypatch):
    """Veto events must be appended to veto_log.jsonl."""
    import reconciliation
    log_path = str(tmp_path / "veto_log.jsonl")
    monkeypatch.setattr(reconciliation, "VETO_LOG_FILE", log_path)

    reconcile(make_forecast("long", 0.70), -0.55, make_calib())
    assert os.path.exists(log_path)
    with open(log_path) as f:
        line = f.readline()
    event = json.loads(line)
    assert event["type"] == "veto"
    assert event["direction"] == "long"


def test_confidence_reduction_writes_to_log(tmp_path, monkeypatch):
    import reconciliation
    log_path = str(tmp_path / "veto_log.jsonl")
    monkeypatch.setattr(reconciliation, "VETO_LOG_FILE", log_path)

    reconcile(make_forecast("long", 0.70), -0.20, make_calib())
    with open(log_path) as f:
        event = json.loads(f.readline())
    assert event["type"] == "confidence_reduction"
    assert event["confidence_before"] == pytest.approx(0.70)
    assert event["confidence_after"]  == pytest.approx(0.70 * 0.6, abs=0.001)


def test_passthrough_writes_no_log(tmp_path, monkeypatch):
    """Clean pass-through (no veto, no reduction) → no log written."""
    import reconciliation
    log_path = str(tmp_path / "veto_log.jsonl")
    monkeypatch.setattr(reconciliation, "VETO_LOG_FILE", log_path)

    reconcile(make_forecast("long", 0.70), +0.30, make_calib())
    assert not os.path.exists(log_path)


# ─────────────────────────────────────────────────────────────
#  Custom veto threshold
# ─────────────────────────────────────────────────────────────

def test_higher_threshold_harder_to_veto():
    """threshold=0.8 → -0.5 neutral doesn't veto."""
    result = reconcile(make_forecast("long", 0.70), -0.50, make_calib(veto_threshold=0.8))
    assert result["action"] != "HOLD"


def test_lower_threshold_easier_to_veto():
    """threshold=0.2 → -0.25 neutral triggers veto."""
    result = reconcile(make_forecast("long", 0.70), -0.25, make_calib(veto_threshold=0.2))
    assert result["action"] == "HOLD"
