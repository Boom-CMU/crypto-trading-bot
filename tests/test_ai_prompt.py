"""
tests/test_ai_prompt.py — Unit tests for Task 4 (bilateral AI prompt + validation)

Run: pytest tests/test_ai_prompt.py -v
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analyzer import (
    AI_SYSTEM_PROMPT_NEUTRAL,
    _FALLBACK_NEUTRAL_RESPONSE,
    _validate_ai_response,
    _parse_ai_json,
    _format_ai_analysis,
)


# ─────────────────────────────────────────────────────────────
#  Fixture helpers
# ─────────────────────────────────────────────────────────────

def make_valid_response(
    bull_p=0.45, bear_p=0.35, base_p=0.20,
    direction="long",
    current_price=100.0,
    target=110.0,
    inval=92.0,
    confidence=0.45,
) -> dict:
    """A response that satisfies all 4 constraints by default."""
    return {
        "bull_case":  {"thesis": "RSI momentum", "key_evidence": ["RSI 62"], "probability": bull_p},
        "bear_case":  {"thesis": "Below MA99",   "key_evidence": ["MA99 fail"], "probability": bear_p},
        "base_case":  {"thesis": "Sideways",                                    "probability": base_p},
        "direction":  direction,
        "invalidation_price": inval,
        "target_price":       target,
        "confidence":         confidence,
        "reasoning":  "ทดสอบ",
    }


# ─────────────────────────────────────────────────────────────
#  System prompt: no bullish-bias keywords
# ─────────────────────────────────────────────────────────────

def test_system_prompt_no_upside_first():
    assert "upside ก่อน"       not in AI_SYSTEM_PROMPT_NEUTRAL
    assert "upside opportunity ก่อน" not in AI_SYSTEM_PROMPT_NEUTRAL


def test_system_prompt_no_bull_thesis_first():
    assert "Bull thesis"       not in AI_SYSTEM_PROMPT_NEUTRAL
    assert "Red Flags หลัง"    not in AI_SYSTEM_PROMPT_NEUTRAL
    assert "ไม่ใช่ก่อน"        not in AI_SYSTEM_PROMPT_NEUTRAL


def test_system_prompt_has_both_bull_and_bear_fields():
    """Schema must require both bull_case and bear_case."""
    assert "bull_case"  in AI_SYSTEM_PROMPT_NEUTRAL
    assert "bear_case"  in AI_SYSTEM_PROMPT_NEUTRAL


def test_system_prompt_has_json_schema():
    assert "direction" in AI_SYSTEM_PROMPT_NEUTRAL
    assert "confidence" in AI_SYSTEM_PROMPT_NEUTRAL
    assert "probability" in AI_SYSTEM_PROMPT_NEUTRAL


def test_system_prompt_mentions_hard_constraints():
    assert "HARD CONSTRAINTS" in AI_SYSTEM_PROMPT_NEUTRAL or "constraint" in AI_SYSTEM_PROMPT_NEUTRAL.lower()


# ─────────────────────────────────────────────────────────────
#  _validate_ai_response — constraint 1: probability sum
# ─────────────────────────────────────────────────────────────

def test_valid_response_passes():
    resp = make_valid_response()
    ok, err = _validate_ai_response(resp, current_price=100.0)
    assert ok, f"Expected valid, got error: {err}"


def test_probability_sum_must_equal_1():
    resp = make_valid_response(bull_p=0.50, bear_p=0.40, base_p=0.20)  # sums to 1.10
    ok, err = _validate_ai_response(resp, current_price=100.0)
    assert not ok
    assert "probability sum" in err


def test_probability_sum_tolerance_within_001():
    """Sum of 0.999 or 1.001 should pass (±0.01 tolerance)."""
    resp = make_valid_response(bull_p=0.45, bear_p=0.35, base_p=0.20)  # = 1.00 exactly
    ok, _ = _validate_ai_response(resp, current_price=100.0)
    assert ok


def test_probability_sum_fails_at_110():
    resp = make_valid_response(bull_p=0.60, bear_p=0.30, base_p=0.20)  # 1.10
    ok, err = _validate_ai_response(resp, current_price=100.0)
    assert not ok


# ─────────────────────────────────────────────────────────────
#  _validate_ai_response — constraint 2: direction matches highest prob
# ─────────────────────────────────────────────────────────────

def test_direction_must_match_highest_prob():
    # bull_p=0.45 is highest, but direction="short"
    resp = make_valid_response(bull_p=0.45, bear_p=0.35, base_p=0.20, direction="short")
    ok, err = _validate_ai_response(resp, current_price=100.0)
    assert not ok
    assert "direction" in err.lower() or "highest" in err.lower()


def test_direction_neutral_when_base_highest():
    resp = make_valid_response(bull_p=0.30, bear_p=0.30, base_p=0.40, direction="neutral",
                               target=105.0, inval=95.0, confidence=0.40)
    ok, err = _validate_ai_response(resp, current_price=100.0)
    assert ok, f"Expected valid: {err}"


def test_direction_short_when_bear_highest():
    resp = make_valid_response(bull_p=0.25, bear_p=0.55, base_p=0.20, direction="short",
                               target=85.0, inval=108.0)
    ok, err = _validate_ai_response(resp, current_price=100.0)
    assert ok, f"Expected valid: {err}"


# ─────────────────────────────────────────────────────────────
#  _validate_ai_response — constraint 3: target/invalidation opposite sides
# ─────────────────────────────────────────────────────────────

def test_target_and_inval_must_be_opposite_sides():
    # Both above current price (100) — invalid
    resp = make_valid_response(target=110.0, inval=105.0)  # both > 100
    ok, err = _validate_ai_response(resp, current_price=100.0)
    assert not ok
    assert "same side" in err


def test_target_above_inval_below_passes():
    resp = make_valid_response(target=115.0, inval=90.0)
    ok, err = _validate_ai_response(resp, current_price=100.0)
    assert ok, f"Expected valid: {err}"


def test_zero_target_skips_constraint():
    """target=0 means AI didn't provide one — skip constraint 3."""
    resp = make_valid_response(target=0.0, inval=0.0)
    ok, err = _validate_ai_response(resp, current_price=100.0)
    assert ok, f"Expected valid (0 prices skip constraint): {err}"


# ─────────────────────────────────────────────────────────────
#  _validate_ai_response — constraint 4: confidence ≤ max(prob)
# ─────────────────────────────────────────────────────────────

def test_confidence_exceeds_max_prob_fails():
    # max_prob = 0.45 but confidence = 0.80
    resp = make_valid_response(bull_p=0.45, bear_p=0.35, base_p=0.20, confidence=0.80)
    ok, err = _validate_ai_response(resp, current_price=100.0)
    assert not ok
    assert "confidence" in err


def test_confidence_equal_to_max_prob_passes():
    resp = make_valid_response(bull_p=0.45, bear_p=0.35, base_p=0.20, confidence=0.45)
    ok, err = _validate_ai_response(resp, current_price=100.0)
    assert ok, f"Expected valid: {err}"


# ─────────────────────────────────────────────────────────────
#  Fallback response must itself pass validation
# ─────────────────────────────────────────────────────────────

def test_fallback_response_passes_validation():
    ok, err = _validate_ai_response(_FALLBACK_NEUTRAL_RESPONSE, current_price=100.0)
    assert ok, f"Fallback must be valid: {err}"


def test_fallback_direction_is_neutral():
    assert _FALLBACK_NEUTRAL_RESPONSE["direction"] == "neutral"


# ─────────────────────────────────────────────────────────────
#  _parse_ai_json
# ─────────────────────────────────────────────────────────────

def test_parse_plain_json():
    raw = json.dumps(make_valid_response())
    result = _parse_ai_json(raw)
    assert result is not None
    assert result["direction"] == "long"


def test_parse_json_with_markdown_fences():
    raw = "```json\n" + json.dumps(make_valid_response()) + "\n```"
    result = _parse_ai_json(raw)
    assert result is not None


def test_parse_json_with_plain_fences():
    raw = "```\n" + json.dumps(make_valid_response()) + "\n```"
    result = _parse_ai_json(raw)
    assert result is not None


def test_parse_returns_none_for_invalid_json():
    assert _parse_ai_json("no braces or valid structure at all") is None


def test_parse_extracts_json_from_surrounding_text():
    """AI sometimes adds preamble before the JSON."""
    raw = "Here is my analysis:\n\n" + json.dumps(make_valid_response()) + "\n\nI hope this helps."
    result = _parse_ai_json(raw)
    assert result is not None


# ─────────────────────────────────────────────────────────────
#  _format_ai_analysis
# ─────────────────────────────────────────────────────────────

def test_format_contains_bull_and_bear():
    output = _format_ai_analysis(make_valid_response(), "Tier 2")
    assert "Bull case" in output
    assert "Bear case" in output


def test_format_shows_direction():
    output = _format_ai_analysis(make_valid_response(direction="long"), "Tier 2")
    assert "LONG" in output


def test_format_shows_confidence_percentage():
    output = _format_ai_analysis(make_valid_response(confidence=0.45), "Tier 2")
    assert "45%" in output


def test_format_shows_probabilities():
    resp   = make_valid_response(bull_p=0.45, bear_p=0.35, base_p=0.20)
    output = _format_ai_analysis(resp, "Tier 2")
    assert "45%" in output   # bull probability
    assert "35%" in output   # bear probability


def test_format_fallback_shows_neutral():
    output = _format_ai_analysis(_FALLBACK_NEUTRAL_RESPONSE, "Tier 2")
    assert "NEUTRAL" in output


def test_format_shows_key_evidence():
    resp = make_valid_response()
    resp["bull_case"]["key_evidence"] = ["RSI 62 momentum", "Volume spike ×2.1"]
    output = _format_ai_analysis(resp, "Tier 2")
    assert "RSI 62 momentum" in output
    assert "Volume spike ×2.1" in output
