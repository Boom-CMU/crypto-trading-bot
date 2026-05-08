"""
reconciliation.py — Reconciliation Gate (Task 7)

reconcile(ai_forecast, neutral_score, calibration) → final signal dict

Principle: AI produces forecast; Neutral Score has veto power when the AI's
direction contradicts market structure beyond the calibrated threshold.

Veto rules (symmetric):
  direction="long"  + confidence > 0.6 + neutral_score < -threshold → HOLD
  direction="short" + confidence > 0.6 + neutral_score > +threshold → HOLD

Mild disagreement (sign mismatch, no veto): confidence × 0.6

All veto and confidence-reduction events are appended to veto_log.jsonl
for post-deploy pattern analysis.
"""
from __future__ import annotations

import copy
import json
import logging
import os
from datetime import datetime, timezone

from config import OUTPUT_DIR, LOG_LEVEL

logging.basicConfig(level=getattr(logging, LOG_LEVEL, "INFO"),
                    format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

VETO_LOG_FILE = os.path.join(OUTPUT_DIR, "veto_log.jsonl")


# ─────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────

def _log_event(event: dict) -> None:
    """Append one JSON object per line to veto_log.jsonl."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        with open(VETO_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("veto_log write failed: %s", e)


# ─────────────────────────────────────────────────────────────
#  Core gate
# ─────────────────────────────────────────────────────────────

def reconcile(
    ai_forecast:   dict,
    neutral_score: float,
    calibration:   dict,
) -> dict:
    """
    Apply the reconciliation gate.

    Parameters
    ----------
    ai_forecast   : validated AI JSON (from Task 4 _ai_call_with_retry).
                    Must have keys: direction, confidence, reasoning.
    neutral_score : float in [-1, +1] from compute_neutral_score() (Task 2).
    calibration   : calibration_data.json dict (supplies veto_threshold).

    Returns
    -------
    dict with keys:
      action        : "LONG" | "SHORT" | "NEUTRAL" | "HOLD"
      reason        : str (present when action="HOLD")
      neutral_score : float (echoed for downstream display)
      + all fields from ai_forecast (deep-copied, possibly confidence-reduced)
    """
    forecast   = copy.deepcopy(ai_forecast)
    threshold  = float(calibration.get("veto_threshold", 0.4))
    direction  = forecast.get("direction", "neutral")
    confidence = float(forecast.get("confidence", 0.0))
    symbol     = forecast.get("symbol", "")
    ts         = datetime.now(timezone.utc).isoformat()

    # ── Hard veto — symmetric both directions ────────────────────
    if direction == "long" and confidence > 0.6 and neutral_score < -threshold:
        _log_event({
            "type":          "veto",
            "timestamp":     ts,
            "symbol":        symbol,
            "direction":     direction,
            "confidence":    confidence,
            "neutral_score": neutral_score,
            "threshold":     threshold,
            "reason":        "vetoed: bearish structure",
        })
        log.info("🔴 VETO %s — long blocked (neutral=%.3f < -%.2f)",
                 symbol, neutral_score, threshold)
        return {
            "action":        "HOLD",
            "reason":        "vetoed: bearish structure",
            "ai_forecast":   forecast,
            "neutral_score": neutral_score,
        }

    if direction == "short" and confidence > 0.6 and neutral_score > +threshold:
        _log_event({
            "type":          "veto",
            "timestamp":     ts,
            "symbol":        symbol,
            "direction":     direction,
            "confidence":    confidence,
            "neutral_score": neutral_score,
            "threshold":     threshold,
            "reason":        "vetoed: bullish structure",
        })
        log.info("🔴 VETO %s — short blocked (neutral=%.3f > +%.2f)",
                 symbol, neutral_score, threshold)
        return {
            "action":        "HOLD",
            "reason":        "vetoed: bullish structure",
            "ai_forecast":   forecast,
            "neutral_score": neutral_score,
        }

    # ── Mild disagreement → confidence reduction ─────────────────
    sign_ai      = +1 if direction == "long" else (-1 if direction == "short" else 0)
    sign_neutral = +1 if neutral_score > 0.1 else (-1 if neutral_score < -0.1 else 0)

    if sign_ai != 0 and sign_neutral != 0 and sign_ai != sign_neutral:
        prev_conf           = confidence
        forecast["confidence"] = round(confidence * 0.6, 4)
        forecast["reasoning"]  = (
            forecast.get("reasoning", "") +
            " [confidence reduced: mild structure disagreement]"
        )
        _log_event({
            "type":              "confidence_reduction",
            "timestamp":         ts,
            "symbol":            symbol,
            "direction":         direction,
            "confidence_before": prev_conf,
            "confidence_after":  forecast["confidence"],
            "neutral_score":     neutral_score,
        })
        log.info("⚠️  CONF REDUCED %s — %s vs neutral=%.3f  %.2f→%.2f",
                 symbol, direction, neutral_score, prev_conf, forecast["confidence"])

    return {
        "action":        direction.upper(),
        **forecast,
        "neutral_score": neutral_score,
    }


# ─────────────────────────────────────────────────────────────
#  Display helper
# ─────────────────────────────────────────────────────────────

def _fmt_price_local(p) -> str:
    """Local price formatter (avoids circular import with analyzer)."""
    if p is None or p == 0:
        return "N/A"
    p = float(p)
    if p >= 10000: return f"${p:,.0f}"
    if p >= 1000:  return f"${p:,.2f}"
    if p >= 1:     return f"${p:.4f}"
    if p >= 0.001: return f"${p:.6f}"
    return f"${p:.8f}"


def format_reconcile_section(result: dict, final_targets: dict | None = None) -> str:
    """Display block appended after AI analysis. Shows single reconciled target."""
    action        = result.get("action", "")
    ns            = result.get("neutral_score", 0.0)
    reason        = result.get("reason", "")
    ai_dir        = result.get("direction", "")
    ai_conf       = float(result.get("confidence", 0.0))

    ns_label = "📈 bullish" if ns > 0.1 else ("📉 bearish" if ns < -0.1 else "⚖️ neutral")

    if action == "HOLD":
        status = f"🔴 HOLD — {reason}"
    elif "confidence reduced" in result.get("reasoning", ""):
        status = f"⚠️  {action} (confidence reduced — mild disagreement)"
    elif ai_conf < 0.5:
        # Fix 7: Low confidence warning
        status = f"⚠️ {action} (Low Confidence — {round(ai_conf * 100)}%)"
    else:
        status = f"✅ {action}"

    lines = [
        "\n────────────────────────────────────────────────────────────",
        "🔀 RECONCILIATION:",
        f"   AI Direction  : {ai_dir.upper()}  (confidence {round(ai_conf * 100)}%)",
        f"   Neutral Score : {ns:+.3f}  {ns_label}",
        f"   Final Signal  : {status}",
    ]

    # Fix 1 & 2: Show single reconciled target/invalidation (ATR-based)
    if final_targets is not None:
        tgt   = final_targets.get("target")
        inv   = final_targets.get("inval")
        t_pct = final_targets.get("target_pct", 0)
        i_pct = final_targets.get("inval_pct", 0)
        if tgt and inv:
            lines += [
                f"   Final Target   : {_fmt_price_local(tgt)}  ({t_pct:+.1f}%)",
                f"   Final Inval.   : {_fmt_price_local(inv)}  ({i_pct:+.1f}%)",
            ]

    return "\n".join(lines) + "\n"
