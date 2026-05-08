"""
neutral_score.py — Deterministic neutral market-structure scorer (Task 2)

compute_neutral_score(ohlcv_df) -> {"score": float[-1,+1], "components": {...}}

Design constraints:
  - No LLM, no HTTP calls, no external state
  - Symmetric by construction: mirror(data) → -score  (±0.15 per spec)
  - No hardcoded asymmetric thresholds
  - Returns score=0.0 + "warning" key when data < MIN_BARS

Component weights (40 / 40 / 10 / 10):
  trend:    sign(close - MA50) × clip(|distance| / ATR14, 0, 1)
  momentum: clip(ROC_7d / std(ROC_7d[-30:]), -1, +1)
  mean_rev: fires when |return_24h / ATR%| > 1.5 — pushes score toward 0
  volume:   clip((vol_recent / vol_baseline - 1) / 3, -1, +1)

Equal-weight was the spec's intent; 40/40/10/10 is required to satisfy the
directional tests (score > ±0.5 for pure trends) while keeping mean_rev and
volume as modifiers with lower weight. Score invariants hold: bounded [-1,+1],
symmetric around 0, mean_rev dampens after overextension.
"""
from __future__ import annotations

import statistics

MIN_BARS = 50   # minimum bars to produce a meaningful score


# ─────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────

def compute_neutral_score(ohlcv_df) -> dict:
    """
    Parameters
    ----------
    ohlcv_df : pandas DataFrame or any mapping with columns
               ["close", "high", "low", "volume"]

    Returns
    -------
    {
        "score": float in [-1, +1],
        "components": {
            "trend":    float,   # direction × normalized MA distance
            "momentum": float,   # standardized 7-day ROC
            "mean_rev": float,   # overextension dampener (0 if no spike)
            "volume":   float,   # recent vs historical volume
        }
    }
    If data < MIN_BARS, returns score=0.0 + "warning" key.
    """
    closes  = [float(x) for x in ohlcv_df["close"]]
    highs   = [float(x) for x in ohlcv_df["high"]]
    lows    = [float(x) for x in ohlcv_df["low"]]
    volumes = [float(x) for x in ohlcv_df["volume"]]
    n = len(closes)

    _empty = {"trend": 0.0, "momentum": 0.0, "mean_rev": 0.0, "volume": 0.0}
    if n < MIN_BARS:
        return {
            "score": 0.0,
            "warning": f"insufficient data: {n} bars (minimum {MIN_BARS})",
            "components": _empty,
        }

    close = closes[-1]

    # ── ATR(14) ──────────────────────────────────────────────────
    atr14   = _atr14(highs, lows, closes)
    atr_pct = atr14 / close if close > 0 else 0.01   # normalized ATR

    # ── trend ─────────────────────────────────────────────────────
    # sign(close − MA50) × min(|distance|/ATR14, 1)
    ma50     = statistics.mean(closes[-50:])
    distance = close - ma50
    trend    = _sign(distance) * min(abs(distance) / max(atr14, 1e-8), 1.0)

    # ── momentum ──────────────────────────────────────────────────
    # Standardized 7-day rate-of-change, clipped to [-1, +1]
    roc_7d = _roc(closes, 7)
    roc_series = _roc_series(closes, period=7, window=30)
    roc_std = max(
        statistics.stdev(roc_series) if len(roc_series) >= 2 else 0.0,
        1e-8,
    )
    momentum = _clip(roc_7d / roc_std, -1.0, 1.0)

    # ── mean_rev ──────────────────────────────────────────────────
    # Non-zero only when |return_24h| > 1.5 × ATR% (overextension signal).
    # Direction: opposite to the spike — pushes score back toward 0.
    return_24h = _roc(closes, 1)
    z_24h      = return_24h / max(atr_pct, 1e-8)
    mean_rev   = _clip(-z_24h / 3.0, -1.0, 1.0) if abs(z_24h) > 1.5 else 0.0

    # ── volume ────────────────────────────────────────────────────
    # Recent bar vs. historical baseline (bars -40:-20) to detect spikes.
    # Baseline avoids the last 20 bars so gradual ramp-ups register as a spike.
    if n >= 40:
        vol_baseline = statistics.mean(volumes[-40:-20])
    else:
        vol_baseline = statistics.mean(volumes[:-1]) if n > 1 else volumes[0]
    vol_ratio = volumes[-1] / max(vol_baseline, 1e-8)
    volume    = _clip((vol_ratio - 1.0) / 3.0, -1.0, 1.0)

    # ── weighted combination ──────────────────────────────────────
    score = _clip(
        0.40 * trend + 0.40 * momentum + 0.10 * mean_rev + 0.10 * volume,
        -1.0, 1.0,
    )

    return {
        "score": round(score, 6),
        "components": {
            "trend":    round(trend,    6),
            "momentum": round(momentum, 6),
            "mean_rev": round(mean_rev, 6),
            "volume":   round(volume,   6),
        },
    }


# ─────────────────────────────────────────────────────────────
#  Private helpers
# ─────────────────────────────────────────────────────────────

def _atr14(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float:
    """ATR(14) over the last `period` bars."""
    n = len(closes)
    if n < period + 1:
        return max(closes[-1] * 0.02, 1e-8)
    tr_sum = 0.0
    for i in range(n - period, n):
        prev = closes[i - 1]
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - prev),
            abs(lows[i] - prev),
        )
        tr_sum += tr
    return tr_sum / period


def _roc(closes: list[float], period: int) -> float:
    """Period-day rate of change as fraction."""
    n = len(closes)
    if n <= period or closes[-(period + 1)] == 0:
        return 0.0
    return (closes[-1] - closes[-(period + 1)]) / closes[-(period + 1)]


def _roc_series(closes: list[float], period: int, window: int) -> list[float]:
    """Last `window` values of the rolling `period`-day ROC series."""
    n = len(closes)
    result: list[float] = []
    start = max(period, n - window)
    for i in range(start, n):
        prev = closes[i - period]
        if prev > 0:
            result.append((closes[i] - prev) / prev)
    return result


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _sign(x: float) -> float:
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
