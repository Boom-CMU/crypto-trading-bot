"""
analyzer.py — CRYPTO OPPORTUNITY SCANNER (3-Tier Analysis Engine)
Usage:
  python analyzer.py --symbol BTC
  python analyzer.py --symbol ETH --fetch

Tier 1: Claude  — ANTHROPIC_API_KEY
Tier 2: Groq    — GROQ_API_KEY (free)
Tier 3: Technical rules (no API needed)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime

from config import (
    ANTHROPIC_API_KEY, CLAUDE_MODEL,
    GROQ_API_KEY, GROQ_MODEL,
    OUTPUT_DIR, LOG_LEVEL, get_analysis_tier,
)
from data_fetcher import (
    fetch_crypto, save_json, _normalize_symbol,
    _calc_full_opportunity_score, _calc_opportunity_grade,
    _binance_get,
)

try:
    from backtest import (
        lookup_confidence, _rsi_zone, _vol_spike_band, _ma_align_label,
        calc_expected_pct, _calc_expected_pct_legacy, estimate_horizon,
    )
    _BACKTEST_AVAILABLE = True
except ImportError:
    _BACKTEST_AVAILABLE = False

try:
    from calibration import apply_isotonic as _apply_isotonic
    _CALIBRATION_APPLY_AVAILABLE = True
except ImportError:
    _CALIBRATION_APPLY_AVAILABLE = False
    def _apply_isotonic(raw_prob: float, calib: dict) -> float:  # type: ignore[misc]
        """Fallback linear shrinkage when calibration module unavailable."""
        return 0.5 + (raw_prob - 0.5) * 0.6

try:
    from neutral_score import compute_neutral_score as _compute_neutral_score_fn
    _NEUTRAL_SCORE_AVAILABLE = True
except ImportError:
    _NEUTRAL_SCORE_AVAILABLE = False

try:
    from reconciliation import reconcile as _reconcile, format_reconcile_section
    _RECONCILIATION_AVAILABLE = True
except ImportError:
    _RECONCILIATION_AVAILABLE = False

try:
    from trading_time import (
        recommend_trading_time, get_horizon_from_timeframe, TIMEFRAME_PRESETS,
    )
    _TIMING_AVAILABLE = True
except ImportError:
    _TIMING_AVAILABLE = False

logging.basicConfig(level=getattr(logging, LOG_LEVEL, "INFO"),
                    format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ตั้งค่า horizon และ timeframe จาก CLI — ทั้ง 3 tier ใช้ค่านี้ร่วมกัน
_current_horizon: int | None = None
_current_timeframe: str | None = None   # "scalp" | "swing" | "position" | ...

# ─────────────────────────────────────────────────────────────
#  Neutral AI system prompt (Task 4)
#  Replaces all bullish-first prompts. Both tiers share this constant.
# ─────────────────────────────────────────────────────────────
AI_SYSTEM_PROMPT_NEUTRAL = (
    "You are a neutral crypto market analyst. Your only loyalty is to accuracy.\n\n"
    "For every setup, you MUST analyze both bull AND bear scenarios with equal rigor "
    "BEFORE choosing a direction.\n\n"
    "You MUST respond with valid JSON matching this schema exactly:\n"
    "{\n"
    '  "bull_case":  {"thesis": str, "key_evidence": [str], "probability": float},\n'
    '  "bear_case":  {"thesis": str, "key_evidence": [str], "probability": float},\n'
    '  "base_case":  {"thesis": str, "probability": float},\n'
    '  "direction":  "long" | "short" | "neutral",\n'
    '  "invalidation_price": float,\n'
    '  "target_price": float,\n'
    '  "confidence": float,\n'
    '  "reasoning":  str\n'
    "}\n\n"
    "HARD CONSTRAINTS — response will be rejected and retried if violated:\n"
    "1. bull_case.probability + bear_case.probability + base_case.probability == 1.0 (±0.01)\n"
    "2. direction must match the highest-probability case "
    "(bull→long, bear→short, base→neutral)\n"
    "3. invalidation_price must be on the opposite side of current price from target_price\n"
    "4. confidence must not exceed max(bull, bear, base) probability\n"
    "5. If you cannot find concrete evidence for BOTH bull and bear, "
    "return direction='neutral'\n\n"
    "ใส่ข้อความใน reasoning และ thesis เป็นภาษาไทย\n"
    "In key_evidence, provide qualitative pattern descriptions — "
    "do NOT simply list indicator price values already shown in the data "
    "(e.g., write 'price above all MAs — bull structure' not 'MA25: $0.039').\n"
    "Return ONLY the JSON object — no markdown, no extra text."
)

# Returned when all retries exhausted or AI unavailable
_FALLBACK_NEUTRAL_RESPONSE: dict = {
    "bull_case":  {"thesis": "ไม่มีข้อมูลเพียงพอ", "key_evidence": [], "probability": 0.33},
    "bear_case":  {"thesis": "ไม่มีข้อมูลเพียงพอ", "key_evidence": [], "probability": 0.33},
    "base_case":  {"thesis": "AI validation failed — defaulting to neutral", "probability": 0.34},
    "direction":  "neutral",
    "invalidation_price": 0.0,
    "target_price": 0.0,
    "confidence": 0.33,
    "reasoning":  "ไม่สามารถวิเคราะห์ได้ — ระบบ fallback เป็น neutral",
}


# ─────────────────────────────────────────────────────────────
#  Neutral score helper (Task 7 — feeds reconciliation gate)
# ─────────────────────────────────────────────────────────────

def _get_neutral_score(data: dict) -> float:
    """
    Fetch recent daily klines for the symbol and compute neutral structure
    score in [-1, +1].  Returns 0.0 on any error so gate is never blocked
    by a connectivity issue.
    """
    if not _NEUTRAL_SCORE_AVAILABLE:
        return 0.0
    try:
        sym   = data.get("symbol", "BTC")
        pair, _ = _normalize_symbol(sym)
        klines  = _binance_get("/klines", {"symbol": pair, "interval": "1d", "limit": 100})
        if not klines:
            return 0.0
        ohlcv = {
            "close":  [float(k[4]) for k in klines],
            "high":   [float(k[2]) for k in klines],
            "low":    [float(k[3]) for k in klines],
            "volume": [float(k[5]) for k in klines],
        }
        return float(_compute_neutral_score_fn(ohlcv).get("score", 0.0))
    except Exception as e:
        log.warning("neutral_score fetch failed for %s: %s",
                    data.get("symbol", "?"), e)
        return 0.0


# ─────────────────────────────────────────────────────────────
#  Build AI analysis prompt (Tier 1 / Tier 2)
# ─────────────────────────────────────────────────────────────
def _build_prompt(data: dict, expected_pct: float | None = None, horizon: int = 7, confidence: float | None = None) -> str:
    sym    = data["symbol"]
    p      = data.get("price", {})
    tech   = data.get("technicals", {})
    mkt    = data.get("market_data", {})
    struct = tech.get("price_structure", {})
    bk     = data.get("bitkub", {})

    mc = mkt.get("market_cap_usd", 0) or 0
    mc_str = f"${mc/1e9:.2f}B" if mc >= 1e9 else f"${mc/1e6:.2f}M" if mc >= 1e6 else "N/A"

    opp_score = data.get("opportunity_score", 0)
    opp_grade = data.get("opportunity_grade", "?")
    vol_spike = tech.get("volume_spike", 1.0) or 1.0
    rsi_4h    = tech.get("rsi_14_4h", "N/A")

    if bk.get("listed"):
        bitkub_str = f"✅ มีใน Bitkub — ราคา ฿{bk.get('price_thb', 0):,.4f} THB"
    else:
        bitkub_str = "❌ ไม่มีใน Bitkub — ต้องซื้อผ่าน Binance/exchange อื่น"

    # Format forecast for prompt (handles both legacy float and new range dict)
    if isinstance(expected_pct, dict):
        forecast_str = f"ช่วงคาด {expected_pct.get('lower', 0):+.1f}% ถึง {expected_pct.get('upper', 0):+.1f}%"
    elif expected_pct is not None:
        forecast_str = f"{'ขึ้น' if expected_pct >= 0 else 'ลง'} {expected_pct:+.1f}%"
    else:
        forecast_str = "N/A"

    prompt = f"""🚀 OPPORTUNITY SCAN: {sym}/USDT

=== OPPORTUNITY METRICS ===
Grade: [{opp_grade}] | Score: {opp_score}/100
Volume Spike: ×{vol_spike:.2f} (vs 14-day avg)
{bitkub_str}

=== PRICE MOMENTUM ===
ราคาปัจจุบัน : ${p.get('current', 'N/A')}
24h change   : {p.get('change_24h_pct', 'N/A')}%
7d change    : {mkt.get('change_7d_pct', 'N/A')}%
24h High/Low : ${p.get('high_24h', 'N/A')} / ${p.get('low_24h', 'N/A')}
Volume 24h   : ${p.get('volume_24h_usdt', 0):,} USDT
Period High  : ${tech.get('high_period', 'N/A')} ({tech.get('pct_from_high', 'N/A')}% from high)
Market Cap   : {mc_str}

=== TECHNICAL INDICATORS ===
RSI Daily(14): {tech.get('rsi_14', 'N/A')} | RSI 4h: {rsi_4h}
MA7 : ${tech.get('ma7', 'N/A')}
MA25: ${tech.get('ma25', 'N/A')}  {'✅ ABOVE' if tech.get('above_ma25') else '❌ BELOW' if tech.get('above_ma25') is False else 'N/A'}
MA99: ${tech.get('ma99', 'N/A')}  {'✅ ABOVE' if tech.get('above_ma99') else '❌ BELOW' if tech.get('above_ma99') is False else 'N/A'}
ATR(14): {tech.get('atr_14', 'N/A')}

=== PRICE STRUCTURE ===
Phase       : {struct.get('phase', 'N/A')}
Range       : {struct.get('range_pct', 'N/A')}%
Higher Lows : {struct.get('higher_lows', 'N/A')}
Volume Trend: {struct.get('volume_trend', 'N/A')}

=== VOLATILITY FORECAST ===
ช่วงราคาที่คาดในอีก {horizon} วัน (จาก realized vol): {forecast_str}
ความแม่นยำทิศทาง (จาก backtest): {f'~{confidence:.0f}%' if confidence is not None else 'ยังไม่มีข้อมูล'}

=== ANALYSIS REQUEST ===
Grade ปัจจุบัน: [{opp_grade}] score {opp_score}/100
ราคาปัจจุบัน: ${p.get('current', 0)} (ใช้เป็น reference สำหรับ target_price / invalidation_price)
Bitkub: {bitkub_str}
Forecast ช่วง: {forecast_str} ใน {horizon} วัน (ความแม่นยำทิศทาง: {f'~{confidence:.0f}%' if confidence is not None else '?'})

วิเคราะห์ข้อมูลข้างต้นและตอบเป็น JSON schema ที่กำหนดไว้ใน system prompt
ทั้ง bull_case และ bear_case ต้องมีหลักฐานจาก indicators จริงที่แสดงด้านบน
"""
    return prompt


# ─────────────────────────────────────────────────────────────
#  Price / Volume formatters
# ─────────────────────────────────────────────────────────────
def _fmt_price(p) -> str:
    if p is None or p == 0:
        return "N/A"
    p = float(p)
    if p >= 10000: return f"${p:,.0f}"
    if p >= 1000:  return f"${p:,.2f}"
    if p >= 1:     return f"${p:,.4f}"
    if p >= 0.001: return f"${p:.6f}"
    return f"${p:.8f}"


def _fmt_vol(v) -> str:
    if not v:
        return "N/A"
    v = float(v)
    if v >= 1e9: return f"${v/1e9:.2f}B"
    if v >= 1e6: return f"${v/1e6:.2f}M"
    if v >= 1e3: return f"${v/1e3:.1f}K"
    return f"${v:.0f}"


# ─────────────────────────────────────────────────────────────
#  Backtest confidence lookup
# ─────────────────────────────────────────────────────────────

def _ensure_backtest_fresh() -> None:
    """รัน backtest อัตโนมัติถ้าไฟล์ยังไม่มี หรืออายุเกิน 7 วัน"""
    if not _BACKTEST_AVAILABLE:
        return
    from backtest import BACKTEST_FILE, run_backtest
    import time

    needs_update = False
    reason = ""

    if not os.path.exists(BACKTEST_FILE):
        needs_update = True
        reason = "ยังไม่มีข้อมูล backtest"
    else:
        age_days = (time.time() - os.path.getmtime(BACKTEST_FILE)) / 86400
        if age_days >= 7:
            needs_update = True
            reason = f"ข้อมูล backtest อายุ {age_days:.0f} วัน"

    if needs_update:
        print(f"\n🔄 {reason} — กำลังอัปเดต backtest (~5 นาที)...")
        run_backtest(verbose=True)


def _get_setup_confidence(data: dict) -> float | None:
    """ดึง historical win rate ของ setup นี้จาก backtest_results.json"""
    if not _BACKTEST_AVAILABLE:
        return None
    tech       = data.get("technicals", {})
    grade      = data.get("opportunity_grade", "D")
    phase      = tech.get("price_structure", {}).get("phase", "UNKNOWN")
    rsi        = tech.get("rsi_14")
    vol_spike  = tech.get("volume_spike", 1.0) or 1.0
    above_ma25 = tech.get("above_ma25")
    above_ma99 = tech.get("above_ma99")
    return lookup_confidence(
        grade          = grade,
        phase          = phase,
        rsi_zone       = _rsi_zone(rsi),
        vol_spike_band = _vol_spike_band(vol_spike),
        ma_align       = _ma_align_label(above_ma25, above_ma99),
        score          = data.get("opportunity_score", 0),
    )


def _calc_expected_and_horizon(data: dict, user_horizon: int | None = None) -> tuple:
    """
    คำนวณ expected range/pct และ horizon สำหรับเหรียญนี้
    Returns: (expected, horizon, horizon_source)
      Legacy mode : expected = float (single point estimate, %)
      New mode    : expected = {"upper": float, "lower": float} (%, symmetric range)
    horizon_source = "user" | "auto"
    """
    if not _BACKTEST_AVAILABLE:
        return None, 7, "auto"

    from config import USE_LEGACY_FORECASTER

    tech      = data.get("technicals", {})
    p         = data.get("price", {})
    mkt       = data.get("market_data", {})
    chg_24h   = p.get("change_24h_pct", 0) or 0
    chg_7d    = mkt.get("change_7d_pct", 0) or 0
    rsi_4h    = tech.get("rsi_14_4h")
    rsi_14    = tech.get("rsi_14")
    vol_spike = tech.get("volume_spike", 1.0) or 1.0
    phase     = tech.get("price_structure", {}).get("phase", "UNKNOWN")
    symbol    = data.get("symbol", "UNKNOWN")

    if user_horizon:
        horizon        = user_horizon
        horizon_source = "user"
    else:
        horizon        = estimate_horizon(phase, vol_spike)
        horizon_source = "auto"

    if USE_LEGACY_FORECASTER:
        expected = _calc_expected_pct_legacy(chg_24h, chg_7d, rsi_4h, vol_spike, horizon)
        return expected, horizon, horizon_source

    # New mode: load calibration (file must already exist from Task 1a run)
    try:
        from calibration import CALIBRATION_FILE, load_calibration
        import os as _os
        if _os.path.exists(CALIBRATION_FILE):
            import json as _json
            with open(CALIBRATION_FILE, encoding="utf-8") as _f:
                _calib = _json.load(_f)
        else:
            _calib = None
    except Exception:
        _calib = None

    if _calib is None:
        # Calibration file not yet generated — fall back silently
        expected = _calc_expected_pct_legacy(chg_24h, chg_7d, rsi_4h, vol_spike, horizon)
        return expected, horizon, horizon_source

    rsi_for_range = rsi_14 if rsi_14 is not None else rsi_4h
    expected = calc_expected_pct(symbol, horizon, rsi_for_range, chg_24h / 100.0, _calib)
    return expected, horizon, horizon_source


# ─────────────────────────────────────────────────────────────
#  1h change (quick Binance kline fetch)
# ─────────────────────────────────────────────────────────────
def _fetch_1h_change(pair: str) -> float | None:
    klines = _binance_get("/klines", {"symbol": pair, "interval": "1h", "limit": 2})
    if klines and len(klines) >= 1:
        try:
            open_1h  = float(klines[-1][1])
            close_1h = float(klines[-1][4])
            if open_1h > 0:
                return round((close_1h - open_1h) / open_1h * 100, 2)
        except (ValueError, TypeError):
            pass
    return None


# ─────────────────────────────────────────────────────────────
#  Sub-score calculations (Risk Lover Mode)
# ─────────────────────────────────────────────────────────────
def _momentum_score(data: dict) -> float:
    """Momentum Score 1-10 — volume spike + price action + MA alignment"""
    tech       = data.get("technicals", {})
    p          = data.get("price", {})
    vol_spike  = tech.get("volume_spike", 1.0) or 1.0
    chg_24h    = p.get("change_24h_pct", 0) or 0
    above_ma99 = tech.get("above_ma99")
    above_ma25 = tech.get("above_ma25")
    rsi_4h     = tech.get("rsi_14_4h")

    score = 0.0

    # Volume spike (0-4): biggest signal for risk lover
    if vol_spike >= 3.0:   score += 4.0
    elif vol_spike >= 2.0: score += 3.0
    elif vol_spike >= 1.5: score += 2.0
    elif vol_spike >= 1.2: score += 1.0

    # 24h price momentum (0-3)
    if chg_24h >= 15:    score += 3.0
    elif chg_24h >= 7:   score += 2.0
    elif chg_24h >= 3:   score += 1.5
    elif chg_24h >= 1:   score += 0.5
    elif chg_24h <= -10: score -= 0.5

    # MA alignment (0-2)
    if above_ma99 is True and above_ma25 is True:
        score += 2.0
    elif above_ma99 is True:
        score += 1.5
    elif above_ma25 is True:
        score += 0.5

    # 4h intraday momentum (0-1)
    if rsi_4h is not None:
        if rsi_4h >= 60:   score += 1.0
        elif rsi_4h >= 50: score += 0.5

    return round(min(max(score, 1.0), 10.0), 1)


def _upside_score(data: dict) -> float:
    """Upside Potential Score 1-10 — ATH distance + market cap + 7d trend"""
    tech          = data.get("technicals", {})
    mkt           = data.get("market_data", {})
    pct_from_high = tech.get("pct_from_high", 0) or 0
    mc            = mkt.get("market_cap_usd") or 0
    chg_7d        = mkt.get("change_7d_pct", 0) or 0

    score = 0.0

    # ATH distance (0-4): further below ATH = more upside room
    ath_dist = abs(pct_from_high) if pct_from_high < 0 else 0
    if ath_dist >= 80:   score += 4.0
    elif ath_dist >= 60: score += 3.5
    elif ath_dist >= 40: score += 3.0
    elif ath_dist >= 20: score += 2.0
    elif ath_dist >= 10: score += 1.0
    else:                score += 0.5  # near ATH = limited upside

    # Market cap (0-3): smaller = more room to grow
    if mc > 0:
        if mc < 500e6:   score += 3.0
        elif mc < 2e9:   score += 2.5
        elif mc < 10e9:  score += 2.0
        elif mc < 50e9:  score += 1.0
        # >50B: 0 pts (BTC/ETH scale — limited asymmetric upside)
    else:
        score += 1.5  # unknown MC — assume mid-tier

    # 7d momentum direction (0-3): trend continuation
    if chg_7d >= 30:   score += 3.0
    elif chg_7d >= 15: score += 2.5
    elif chg_7d >= 5:  score += 2.0
    elif chg_7d >= 0:  score += 1.0

    return round(min(max(score, 1.0), 10.0), 1)


def _technical_score(data: dict) -> float:
    """Technical Score 1-10 — RSI zone + MA alignment + price structure"""
    tech       = data.get("technicals", {})
    struct     = tech.get("price_structure", {})
    rsi        = tech.get("rsi_14")
    above_ma25 = tech.get("above_ma25")
    above_ma99 = tech.get("above_ma99")
    phase      = struct.get("phase", "")

    score = 0.0

    # RSI zone (0-4): 50-75 ideal for momentum trade
    if rsi is not None:
        if 50 <= rsi <= 75:   score += 4.0
        elif 35 <= rsi < 50:  score += 2.5  # recovering
        elif 75 < rsi <= 82:  score += 2.0  # hot but not blown
        elif rsi < 35:        score += 2.0  # oversold reversal play
        # RSI > 82: 0 pts — extreme overbought
    else:
        score += 2.0  # no data, neutral

    # MA alignment (0-4)
    if above_ma25 is True and above_ma99 is True:
        score += 4.0
    elif above_ma99 is True:
        score += 3.0  # bull structure, short-term pullback
    elif above_ma25 is True:
        score += 1.5

    # Price structure (0-2)
    score += {
        "TIGHT_RANGE_HIGHER_LOWS":   2.0,
        "CONSOLIDATING_HIGHER_LOWS": 1.5,
        "UPTREND_PULLBACK":          1.5,
        "CONSOLIDATING_FLAT":        0.5,
        "VOLATILE_NO_STRUCTURE":     0.0,
    }.get(phase, 0.5)

    return round(min(max(score, 1.0), 10.0), 1)


def _risk_vol_score(data: dict) -> float:
    """Risk/Volatility Score 1-10 — for Risk Lover: volatility = opportunity"""
    tech    = data.get("technicals", {})
    p       = data.get("price", {})
    vol_24h = p.get("volume_24h_usdt", 0) or 0
    atr     = tech.get("atr_14") or 0
    price   = p.get("current", 0) or 0
    low_24h = p.get("low_24h", 0) or 0

    score = 0.0

    # Liquidity (0-4): must have enough volume to enter/exit cleanly
    if vol_24h >= 1e9:       score += 4.0
    elif vol_24h >= 100e6:   score += 3.5
    elif vol_24h >= 50e6:    score += 3.0
    elif vol_24h >= 10e6:    score += 2.0
    elif vol_24h >= 1e6:     score += 1.0

    # ATR% volatility (0-3): 3-10% sweet spot for crypto risk lover
    atr_pct = (atr / price * 100) if price > 0 and atr > 0 else 0
    if 3 <= atr_pct <= 10:   score += 3.0
    elif 2 <= atr_pct < 3:   score += 2.0
    elif atr_pct > 10:        score += 2.0  # very volatile = possible big move
    elif 1 <= atr_pct < 2:   score += 1.0

    # Near 24h support (0-3): buying near low = better R/R entry
    if price > 0 and low_24h > 0:
        pct_from_low = (price - low_24h) / price * 100
        if pct_from_low <= 2:    score += 3.0
        elif pct_from_low <= 5:  score += 2.0
        elif pct_from_low <= 10: score += 1.0

    return round(min(max(score, 1.0), 10.0), 1)


# ─────────────────────────────────────────────────────────────
#  Calibration loader for targets (module-level cache, no rebuild)
# ─────────────────────────────────────────────────────────────
_TARGETS_CALIBRATION: dict | None = None

def _get_calibration_for_targets() -> dict:
    """Load calibration once per session without triggering a rebuild."""
    global _TARGETS_CALIBRATION
    if _TARGETS_CALIBRATION is not None:
        return _TARGETS_CALIBRATION
    try:
        from calibration import CALIBRATION_FILE
        import json as _json
        if os.path.exists(CALIBRATION_FILE):
            with open(CALIBRATION_FILE, encoding="utf-8") as _f:
                _TARGETS_CALIBRATION = _json.load(_f)
                return _TARGETS_CALIBRATION
    except Exception:
        pass
    _TARGETS_CALIBRATION = {"atr_multipliers": {"target": 2.5, "stop": 1.5}, "per_coin": {}}
    return _TARGETS_CALIBRATION


def _signal_to_direction(signal: str) -> str:
    """Map composite signal string to long/short/neutral for target computation."""
    if signal in ("Strong Buy", "Buy"):
        return "long"
    if signal in ("Strong Sell", "Sell"):
        return "short"
    return "neutral"


def _calc_targets_new(data: dict, direction: str, calibration: dict) -> dict:
    """
    ATR-based single target + invalidation (Task 5).
    Uses calibration ATR multipliers; applies sanity cap if live ATR is anomalous.
    Returns dict with action="neutral" + reason if R:R < 1.5.
    No hardcoded timeframe labels.
    """
    p      = data.get("price", {})
    tech   = data.get("technicals", {})
    symbol = data.get("symbol", "UNKNOWN").upper()
    entry  = float(p.get("current", 0) or 0)

    live_atr = float(tech.get("atr_14") or entry * 0.03)

    # Sanity cap: prevent anomalous bars from blowing out targets
    calib_atr = calibration.get("per_coin", {}).get(symbol, {}).get("atr_14_avg")
    if calib_atr and live_atr > 3.0 * calib_atr:
        log.warning("ATR sanity cap %s: live=%.4f > 3×calib=%.4f — capping",
                    symbol, live_atr, calib_atr)
        live_atr = calib_atr * 3.0

    k = float(calibration.get("atr_multipliers", {}).get("target", 2.5))
    j = float(calibration.get("atr_multipliers", {}).get("stop",   1.5))

    if direction == "short":
        target = entry - k * live_atr
        inval  = entry + j * live_atr
    else:   # "long" or "neutral" — default bullish side for display
        target = entry + k * live_atr
        inval  = entry - j * live_atr

    reward = abs(target - entry)
    risk   = abs(inval  - entry)
    rr     = round(reward / risk, 1) if risk > 0 else 0.0

    def pct(t: float) -> float:
        return round((t - entry) / entry * 100, 1) if entry > 0 else 0.0

    base = {
        "entry":      entry,
        "target":     round(target, 8),
        "target_pct": pct(target),
        "inval":      round(inval, 8),
        "inval_pct":  pct(inval),
        "rr":         rr,
        "atr":        round(live_atr, 8),
        "direction":  direction,
        "k":          k,
        "j":          j,
    }
    if rr < 1.5:
        return {**base, "action": "neutral",
                "reason": f"R:R {rr:.1f} < 1.5 — รอ setup ที่ดีกว่า"}
    return {**base, "action": direction}


# ─────────────────────────────────────────────────────────────
#  Targets + Composite score
# ─────────────────────────────────────────────────────────────
def _calc_targets(data: dict) -> dict:
    """ATR-based trading targets; SL capped at 15% for crypto"""
    p      = data.get("price", {})
    tech   = data.get("technicals", {})
    struct = tech.get("price_structure", {})
    price  = p.get("current", 0) or 0
    atr    = tech.get("atr_14") or (price * 0.03)

    sl  = max(price - atr * 1.5, price * 0.85)
    tp1 = price + atr * 3
    tp2 = price + atr * 6
    tp3 = price + atr * 10

    def pct_gain(t):
        return round((t - price) / price * 100, 1) if price > 0 else 0

    sl_pct = round((price - sl) / price * 100, 1) if price > 0 else 0
    rr     = round((tp2 - price) / (price - sl), 1) if price > sl else 0

    # TP3 timeframe adapts to momentum strength
    phase     = struct.get("phase", "")
    vol_spike = tech.get("volume_spike", 1.0) or 1.0
    if vol_spike >= 2.0 and phase == "TIGHT_RANGE_HIGHER_LOWS":
        tp3_tf = "~1-2 วัน"
    elif vol_spike >= 1.5 or phase in ("CONSOLIDATING_HIGHER_LOWS", "UPTREND_PULLBACK"):
        tp3_tf = "~3-4 วัน"
    else:
        tp3_tf = "~5-7 วัน"

    return {
        "price": price, "atr": atr,
        "sl": sl,   "sl_pct": sl_pct,
        "tp1": tp1, "tp1_pct": pct_gain(tp1), "tp1_tf": "ภายใน 1 ชม.",
        "tp2": tp2, "tp2_pct": pct_gain(tp2), "tp2_tf": "ภายใน 24 ชม.",
        "tp3": tp3, "tp3_pct": pct_gain(tp3), "tp3_tf": tp3_tf,
        "rr": rr,
    }


def _calc_all_scores(data: dict) -> dict:
    """All sub-scores + asymmetric EV + composite (Risk Lover weighting)"""
    m = _momentum_score(data)
    u = _upside_score(data)
    t = _technical_score(data)
    r = _risk_vol_score(data)
    tg = _calc_targets(data)

    # Probability estimate from momentum + technical strength
    raw_prob = m / 10 * 0.6 + t / 10 * 0.4

    from config import USE_LEGACY_FORECASTER
    if USE_LEGACY_FORECASTER:
        # Old formula (biased upward — floor 0.30, ceiling 0.75, additive 0.30)
        prob_up = round(min(0.75, max(0.25, 0.30 + raw_prob * 0.40)), 2)
    else:
        # Task 6: symmetric calibrated probability
        # Constraints: floor/ceiling symmetric around 0.5 (0.15/0.85)
        #              no additive constant
        #              raw_prob=0.5 → prob_up=0.5 (guaranteed by apply_isotonic)
        _calib_pu   = _get_calibration_for_targets()
        _prob_calib = _calib_pu.get("prob_up_calibration",
                                    {"method": "linear_scale", "scale": 0.6})
        _calibrated = _apply_isotonic(raw_prob, _prob_calib)
        prob_up     = round(min(0.85, max(0.15, _calibrated)), 2)

    prob_down = round(1.0 - prob_up, 2)

    tp2_pct = tg["tp2_pct"]
    sl_pct  = tg["sl_pct"]

    # Asymmetric EV: high upside even at <50% probability counts
    ev = round((prob_up * tp2_pct) - (prob_down * sl_pct), 2)

    if ev > 30:    ev_score = 10.0
    elif ev > 15:  ev_score = 8.0
    elif ev > 8:   ev_score = 6.0
    elif ev > 3:   ev_score = 4.0
    elif ev > 0:   ev_score = 2.0
    else:          ev_score = 0.0

    # Weighted composite: Momentum 30% | Upside 30% | Technical 20% | EV 15% | Risk 5%
    composite = int(min(100, max(0, round(
        m * 3.0 + u * 3.0 + t * 2.0 + ev_score * 1.5 + r * 0.5
    ))))

    if composite >= 81:   label, signal = "High Conviction",  "Strong Buy"
    elif composite >= 66: label, signal = "Good Opportunity", "Buy"
    elif composite >= 51: label, signal = "Speculative",      "Hold"
    elif composite >= 31: label, signal = "High Risk",        "Sell"
    else:                 label, signal = "Skip",             "Strong Sell"

    return {
        "momentum": m, "upside": u, "technical": t, "risk_vol": r,
        "ev_score": ev_score, "ev": ev,
        "prob_up": prob_up, "prob_down": prob_down,
        "tp2_pct": tp2_pct, "sl_pct": sl_pct,
        "composite": composite, "label": label, "signal": signal,
    }


# ─────────────────────────────────────────────────────────────
#  Output consistency validator (General #1)
# ─────────────────────────────────────────────────────────────

def _validate_output_consistency(
    targets: dict,
    timing: dict | None,
    tech: dict,
) -> list[str]:
    """
    Detect conflicts between target/invalidation, entry type, and urgency.
    Logs each issue and returns a list of warning strings for display.
    """
    issues: list[str] = []
    rsi_4h = tech.get("rsi_14_4h")

    # RSI 4h overbought vs HIGH urgency (should be caught by trading_time.py fix)
    if timing is not None and rsi_4h is not None and rsi_4h > 80:
        if timing.get("urgency") == "HIGH":
            issues.append(
                f"RSI 4h={rsi_4h:.1f} overbought conflicts with urgency=HIGH — "
                "urgency auto-corrected to MEDIUM"
            )

    # Target and invalidation on same side of entry (invalid R/R geometry)
    entry  = float(targets.get("entry",  0) or 0)
    tgt    = float(targets.get("target", 0) or 0)
    inval  = float(targets.get("inval",  0) or 0)
    if entry > 0 and tgt != 0 and inval != 0:
        if (tgt > entry) == (inval > entry):
            issues.append(
                f"Target ({tgt:.6f}) and Invalidation ({inval:.6f}) "
                f"are on the same side of entry ({entry:.6f}) — check ATR calc"
            )

    # Very low R/R
    rr = float(targets.get("rr", 0) or 0)
    if 0 < rr < 1.0:
        issues.append(f"R/R ratio {rr:.1f} is below 1.0 — setup may be unfavorable")

    for issue in issues:
        log.warning("Output consistency: %s", issue)

    return issues


# ─────────────────────────────────────────────────────────────
#  Unified trading card (identical format across all tiers)
# ─────────────────────────────────────────────────────────────
def _build_forecast_lines(
    expected_pct,           # float (legacy) | dict {"upper","lower"} (new) | None
    horizon: int,
    horizon_source: str,
    confidence: float | None,
) -> list[str]:
    """สร้าง lines แสดงผล forecast ก่อน Composite Score"""
    if expected_pct is None or not _BACKTEST_AVAILABLE:
        return [""]

    hz_label = f"{horizon} วัน ({'กำหนดเอง' if horizon_source == 'user' else 'auto'})"
    conf_str = f"~{confidence:.0f}%" if confidence is not None else "?"

    if isinstance(expected_pct, dict):
        upper = expected_pct.get("upper", 0)
        lower = expected_pct.get("lower", 0)
        return [
            "────────────────────────────────────────────────────────────",
            f"💡 ช่วงคาด 2σ (horizon: {hz_label}):",
            f"   ช่วงที่คาด        : {lower:+.1f}% ถึง {upper:+.1f}%",
            f"   ความแม่นยำทิศทาง : {conf_str}",
        ]

    # Legacy float
    direction = "ขึ้น" if expected_pct >= 0 else "ลง"
    return [
        "────────────────────────────────────────────────────────────",
        f"💡 ประมาณการ (horizon: {hz_label}):",
        f"   คาดว่าจะ{direction}      : {expected_pct:+.1f}%",
        f"   ความแม่นยำของการพยากรณ์: {conf_str}",
    ]


def _format_trading_card(
    data: dict,
    tier_name: str,
    scores: dict,
    targets: dict,
    chg_1h: float | None = None,
    confidence: float | None = None,
    expected_pct: float | None = None,
    horizon: int = 7,
    horizon_source: str = "auto",
    timing: dict | None = None,
) -> str:
    sym    = data["symbol"]
    p      = data.get("price", {})
    tech   = data.get("technicals", {})
    mkt    = data.get("market_data", {})
    struct = tech.get("price_structure", {})

    price         = targets.get("price") or targets.get("entry", 0)
    chg_24h       = p.get("change_24h_pct", 0) or 0
    chg_7d        = mkt.get("change_7d_pct", 0) or 0
    vol_24h       = p.get("volume_24h_usdt", 0) or 0
    pct_from_high = tech.get("pct_from_high", 0) or 0
    vol_spike     = tech.get("volume_spike", 1.0) or 1.0
    phase         = struct.get("phase", "")
    low_24h       = p.get("low_24h", 0) or 0
    high_24h      = p.get("high_24h", 0) or 0

    rsi_14_val   = tech.get("rsi_14")
    rsi_4h_val   = tech.get("rsi_14_4h")
    ma7_val      = tech.get("ma7")
    ma25_val     = tech.get("ma25")
    ma99_val     = tech.get("ma99")
    above_ma25_v = tech.get("above_ma25")
    above_ma99_v = tech.get("above_ma99")
    atr_14_val   = tech.get("atr_14")
    above_ma7_v  = (float(price) > float(ma7_val)) if (price and ma7_val) else None

    # Volume vs average string
    vol_vs_pct = round((vol_spike - 1) * 100)
    vol_vs_str = f"+{vol_vs_pct:.0f}%" if vol_vs_pct >= 0 else f"{vol_vs_pct:.0f}%"

    chg_1h_str = f"{chg_1h:+.2f}%" if chg_1h is not None else "N/A"

    support    = low_24h if low_24h > 0 else price * 0.95
    resistance = high_24h if high_24h > 0 else price * 1.05

    # EV components
    prob_up   = scores["prob_up"]
    prob_down = scores["prob_down"]
    tp2_pct   = scores["tp2_pct"]
    sl_pct_v  = scores["sl_pct"]
    ev_up     = round(prob_up * tp2_pct, 2)
    ev_dn     = round(prob_down * sl_pct_v, 2)

    # Signal display
    signal_map = {
        "Strong Buy":  "🚀 Strong Buy",
        "Buy":         "✅ Buy",
        "Hold":        "👀 Hold (Speculative)",
        "Sell":        "⚠️ Sell",
        "Strong Sell": "❌ Strong Sell",
    }
    signal_disp = signal_map.get(scores["signal"], scores["signal"])

    # Inline helpers for Technical Indicators block
    def _rsi_flag_str(r):
        if r is None: return ""
        if r > 78:        return "⚠️ Overbought"
        if 50 <= r <= 68: return "🔥 Momentum zone"
        if r < 35:        return "📉 Oversold"
        return ""

    def _ma_chk_str(above):
        if above is True:  return "✅"
        if above is False: return "❌"
        return "—"

    _atr_display = _fmt_price(atr_14_val) if atr_14_val else "N/A"
    _atr_warning = ("  ⚠️ ATR unavailable — 2σ range estimate may be inaccurate"
                    if not atr_14_val else "")

    # Entry timing advice (quick one-liner for the trade card row)
    if phase == "TIGHT_RANGE_HIGHER_LOWS" and vol_spike >= 2.0:
        entry_timing_str = "เข้าได้เลย — breakout กำลังเกิด (volume ยืนยันแล้ว)"
    elif phase == "TIGHT_RANGE_HIGHER_LOWS" and vol_spike >= 1.5:
        entry_timing_str = "เข้าได้เลย — coil setup แน่น รอ candle ปิดยืนยัน"
    elif phase == "CONSOLIDATING_HIGHER_LOWS":
        entry_timing_str = f"รอ ~30-60 นาที — รอ volume spike ยืนยัน breakout"
    elif phase == "UPTREND_PULLBACK":
        entry_timing_str = f"รอ pullback ~15-30 นาที — retest แนวรับ {_fmt_price(support)}"
    elif vol_spike >= 2.0:
        entry_timing_str = "เข้าได้เลย — แต่ระวัง overextension จาก volume spike"
    elif vol_spike >= 1.3:
        entry_timing_str = "รอ ~15-30 นาที — volume เริ่มมา รอ confirm อีกครั้ง"
    else:
        entry_timing_str = "รอ ~1-2 ชม. — ยังไม่มี breakout สัญญาณ รอ volume confirm"

    # Fix 6: Detect pullback/retest setups to show dual-entry scenarios
    _has_pullback_setup = (
        phase == "UPTREND_PULLBACK"
        and ("retest" in entry_timing_str or "pullback" in entry_timing_str)
    )

    lines = [
        "════════════════════════════════════════════════════════════",
        f"🚀 {sym}/USDT — วิเคราะห์โดย {tier_name}",
        "════════════════════════════════════════════════════════════",
        f"💰 ราคา: {_fmt_price(price)}",
        f"   1h: {chg_1h_str}  |  7d: {chg_7d:+.2f}%",
        f"   Volume 24h: {_fmt_vol(vol_24h)}  |  vs Average: {vol_vs_str}",
        "",
        "────────────────────────────────────────────────────────────",
        "📊 Technical Indicators:",
        (f"   RSI(14)  : {rsi_14_val:.1f}  {_rsi_flag_str(rsi_14_val)}"
         f"  |  4h RSI: {rsi_4h_val:.1f}  {_rsi_flag_str(rsi_4h_val)}"
         if rsi_14_val is not None and rsi_4h_val is not None else
         f"   RSI(14)  : {rsi_14_val if rsi_14_val is not None else 'N/A'}  "
         f"{_rsi_flag_str(rsi_14_val)}  |  4h RSI: {rsi_4h_val if rsi_4h_val is not None else 'N/A'}  "
         f"{_rsi_flag_str(rsi_4h_val)}"),
        (f"   Vol Spike: ×{vol_spike:.2f}  "
         f"{'🔥 SPIKE!' if vol_spike >= 2.0 else ('↑ elevated' if vol_spike >= 1.3 else '')}"),
        f"   MA7  : {_fmt_price(ma7_val) if ma7_val else 'N/A'}  {_ma_chk_str(above_ma7_v)}",
        (f"   MA25 : {_fmt_price(ma25_val) if ma25_val else 'N/A'}  {_ma_chk_str(above_ma25_v)}"
         f"  |  MA99: {_fmt_price(ma99_val) if ma99_val else 'N/A'}  {_ma_chk_str(above_ma99_v)}"),
        f"   ATR(14) : {_atr_display}{_atr_warning}",
        f"   Phase   : {phase or 'N/A'}",
        "",
        "📈 คะแนนย่อย:",
        f"  Momentum        : {scores['momentum']:.1f}/10",
        f"  Upside Potential: {scores['upside']:.1f}/10",
        f"  Technical       : {scores['technical']:.1f}/10",
        f"  Risk/Volatility : {scores['risk_vol']:.1f}/10",
        "",
        "🎯 โอกาส:",
        f"  ขึ้น: {round(prob_up * 100):.0f}%  |  ลง: {round(prob_down * 100):.0f}%",
        f"  ถึง Target {_fmt_price(targets.get('tp2') or targets.get('target', 0))}: +{tp2_pct:.1f}%",
        f"  ATH Distance: {pct_from_high:.1f}%  ← upside ถ้าถึง ATH",
        "",
        f"⚡ Expected Value (Asymmetric): [+{ev_up:.2f}% / -{ev_dn:.2f}%]",
        "",
        f"🏆 คะแนนความคุ้มค่า: {scores['composite']}/100 — {scores['label']}"
        + (f"  |  ความแม่นยำ: ~{confidence:.0f}%" if confidence is not None
           else "  |  ความแม่นยำ: รัน python backtest.py ก่อน"),
        "",
        "────────────────────────────────────────────────────────────",
        "📋 คำแนะนำการเทรด:",
        f"  สัญญาณ    : {signal_disp}",
        (f"  Entry (A)  : {_fmt_price(price)} ← Market Entry (เข้าทันที)\n"
         f"  Entry (B)  : {_fmt_price(support)} ← Limit Entry (รอ retest แนวรับ)")
        if _has_pullback_setup else
        f"  Entry      : {_fmt_price(price)} ← Market Entry",
    ]

    # ── target / stop display — new single-target (Task 5) vs legacy ───
    if "target" in targets:
        dir_arrow = "↑" if targets.get("direction") == "long" else ("↓" if targets.get("direction") == "short" else "↕")
        lines += [
            f"  Target     : {_fmt_price(targets['target'])}  ({targets['target_pct']:+.1f}%)  {dir_arrow}",
            f"  Invalidation: {_fmt_price(targets['inval'])}  ({targets['inval_pct']:+.1f}%)",
            f"  R/R Ratio  : 1:{targets['rr']:.1f}",
        ]
        if targets.get("action") == "neutral" and targets.get("reason"):
            lines.append(f"  ⚠️  {targets['reason']}")
    else:
        lines += [
            f"  Stop Loss  : {_fmt_price(targets['sl'])}  (-{targets['sl_pct']:.1f}%) ← ไม่เกิน 15%",
            f"  TP1        : {_fmt_price(targets['tp1'])}  (+{targets['tp1_pct']:.1f}%) ← {targets['tp1_tf']}",
            f"  TP2        : {_fmt_price(targets['tp2'])}  (+{targets['tp2_pct']:.1f}%) ← {targets['tp2_tf']}",
            f"  TP3        : {_fmt_price(targets['tp3'])}  (+{targets['tp3_pct']:.1f}%) ← {targets['tp3_tf']}",
            f"  R/R Ratio  : 1:{targets['rr']:.1f}",
        ]

    lines += [
        "  Position   : ไม่เกิน 10% ของพอร์ต ต่อ coin 1 ตัว",
        f"  แนวรับ     : {_fmt_price(support)}  —  แนวต้าน: {_fmt_price(resistance)}",
        f"  จังหวะเข้า : {entry_timing_str}",
    ]

    # ── Trading Time Recommendation block ───────────────────────
    if timing is not None:
        t = timing
        lines += [
            "────────────────────────────────────────────────────────────",
            "⏰ แนะนำเวลาซื้อขาย (ICT = เวลาไทย UTC+7):",
            f"  กรอบเวลา    : {t['horizon_label']}",
            f"  เซสชั่นหลัก : {t['best_session']}  ★",
            f"  เซสชั่นรอง  : {t['secondary_session']}",
            f"  เงื่อนไขเข้า: {t['entry_condition']}",
            f"  หลีกเลี่ยง  : {t['avoid_window']}",
            f"  ระดับเร่งด่วน: {t['urgency_display']}",
            f"  หน้าต่างถัดไป: {t['best_session_short']} {t['next_window_str']}",
        ]

    _issues = _validate_output_consistency(targets, timing, tech)
    if _issues:
        lines.append("────────────────────────────────────────────────────────────")
        lines.append("⚠️ Output Consistency Notices:")
        for _issue in _issues:
            lines.append(f"   {_issue}")

    lines += [
        "════════════════════════════════════════════════════════════",
        "⚠️ Crypto มีความผันผวนสูง ใช้ประกอบการตัดสินใจเท่านั้น",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
#  Tier 3: Technical Rules (Risk Lover Mode)
# ─────────────────────────────────────────────────────────────
def _tier3_analysis(data: dict) -> str:
    sym    = data["symbol"]
    p      = data.get("price", {})
    tech   = data.get("technicals", {})
    mkt    = data.get("market_data", {})
    struct = tech.get("price_structure", {})
    bk     = data.get("bitkub", {})

    rsi       = tech.get("rsi_14")
    rsi_4h    = tech.get("rsi_14_4h")
    phase     = struct.get("phase", "")
    vol_trend = struct.get("volume_trend", "")
    vol_spike = tech.get("volume_spike", 1.0) or 1.0
    chg_7d    = mkt.get("change_7d_pct", 0) or 0

    # Compute scores, targets, 1h change
    scores                           = _calc_all_scores(data)
    pair, _                          = _normalize_symbol(sym)
    chg_1h                           = _fetch_1h_change(pair)
    confidence                       = _get_setup_confidence(data)
    expected_pct, horizon, hz_source = _calc_expected_and_horizon(data, _current_horizon)

    # Task 5: single ATR-based target using signal direction
    _calib3  = _get_calibration_for_targets()
    _dir3    = _signal_to_direction(scores["signal"])
    targets  = _calc_targets_new(data, _dir3, _calib3)

    _timing = None
    if _TIMING_AVAILABLE:
        _timing = recommend_trading_time(
            phase         = phase,
            vol_spike     = vol_spike,
            rsi_4h        = tech.get("rsi_14_4h"),
            horizon_days  = horizon,
            symbol        = sym,
            bitkub_listed = data.get("bitkub", {}).get("listed", False),
            horizon_source= hz_source,
        )

    card = _format_trading_card(
        data, "Tier 3 — Technical Rules", scores, targets,
        chg_1h, confidence, expected_pct, horizon, hz_source, _timing
    )

    # Build signal breakdown
    signals = []
    bull, bear = 0, 0

    if rsi is not None:
        if 50 <= rsi <= 68:
            signals.append(f"🔥 RSI {rsi:.1f} — Sweet spot! momentum กำลังสร้างตัว")
            bull += 2
        elif 68 < rsi <= 78:
            signals.append(f"⚡ RSI {rsi:.1f} — Momentum แรง (ยังไม่ over)")
            bull += 1
        elif rsi < 35:
            signals.append(f"🎣 RSI {rsi:.1f} — Oversold reversal play")
            bull += 1
        elif 35 <= rsi < 50:
            signals.append(f"⚡ RSI {rsi:.1f} — กำลังฟื้นตัว")
            bull += 1
        else:
            signals.append(f"⚠️ RSI {rsi:.1f} — Overbought ระวัง exhaustion")
            bear += 1

    if rsi_4h is not None:
        if rsi_4h > 50:
            signals.append(f"✅ RSI 4h {rsi_4h:.1f} — Intraday momentum บวก")
            bull += 1
        else:
            signals.append(f"❌ RSI 4h {rsi_4h:.1f} — Intraday อ่อนแรง")
            bear += 1

    if vol_spike >= 3.0:
        signals.append(f"🚀 Volume spike ×{vol_spike:.1f} — MAJOR accumulation signal!")
        bull += 3
    elif vol_spike >= 2.0:
        signals.append(f"🔥 Volume spike ×{vol_spike:.1f} — Strong buying interest")
        bull += 2
    elif vol_spike >= 1.5:
        signals.append(f"📊 Volume ×{vol_spike:.1f} — Above average")
        bull += 1
    elif vol_spike < 0.8:
        signals.append(f"😴 Volume ×{vol_spike:.1f} — ต่ำกว่าปกติ ไม่น่าสนใจ")
        bear += 1

    if tech.get("above_ma99") is True:
        signals.append("✅ ราคา > MA99 — Bull market structure")
        bull += 2
    elif tech.get("above_ma99") is False:
        signals.append("❌ ราคา < MA99 — Bear market structure")
        bear += 2

    if tech.get("above_ma25") is True:
        signals.append("✅ ราคา > MA25 — Short-term uptrend")
        bull += 1
    elif tech.get("above_ma25") is False:
        signals.append("❌ ราคา < MA25 — Short-term weakness")
        bear += 1

    if phase == "TIGHT_RANGE_HIGHER_LOWS":
        signals.append("💎 Phase: TIGHT_RANGE — Pre-breakout coiling! เข้าก่อน breakout")
        bull += 3
    elif phase == "CONSOLIDATING_HIGHER_LOWS":
        signals.append("🎯 Phase: CONSOLIDATING_HIGHER_LOWS — Accumulation setup")
        bull += 2
    elif phase == "UPTREND_PULLBACK":
        signals.append("📈 Phase: UPTREND_PULLBACK — Healthy dip in uptrend")
        bull += 1
    elif phase == "CONSOLIDATING_FLAT":
        signals.append("😐 Phase: CONSOLIDATING_FLAT — Sideways รอ catalyst")
    elif phase == "VOLATILE_NO_STRUCTURE":
        signals.append("⚠️ Phase: VOLATILE — ไม่มี structure ระวัง")
        bear += 1

    if vol_trend == "increasing":
        signals.append("📊 Volume trend: เพิ่มขึ้น — confirms momentum")
        bull += 1
    elif vol_trend == "decreasing":
        signals.append("📊 Volume trend: ลดลง — weak conviction")
        bear += 1

    if chg_7d >= 30:
        signals.append(f"🚀 7d: +{chg_7d:.1f}% — Trend แรง (ระวัง FOMO top)")
        bull += 1
    elif chg_7d >= 10:
        signals.append(f"✅ 7d: +{chg_7d:.1f}% — Momentum ดี")
        bull += 1
    elif chg_7d < -20:
        signals.append(f"❌ 7d: {chg_7d:.1f}% — Downtrend แรง")
        bear += 2

    signals_str = "\n  ".join(signals) if signals else "(ไม่มีสัญญาณชัดเจน)"

    if bk.get("listed"):
        bitkub_line = f"✅ Bitkub: ฿{bk.get('price_thb', 0):,.4f} THB"
    else:
        bitkub_line = "❌ ไม่มีใน Bitkub — ใช้ Binance/Gate.io"

    detail_lines = [
        "",
        "────────────────────────────────────────────────────────────",
        f"📊 SIGNAL BREAKDOWN (Tier 3):  Bull: {bull}  |  Bear: {bear}",
        f"  {signals_str}",
        "",
        "📍 PRICE STRUCTURE:",
        f"  Phase       : {phase or 'N/A'}",
        f"  Range       : {struct.get('range_pct', 'N/A')}%",
        f"  Higher Lows : {struct.get('higher_lows', 'N/A')}",
        f"  Volume Trend: {vol_trend or 'N/A'}",
        f"  Volume Spike: ×{vol_spike:.2f}",
        "",
        f"🇹🇭 {bitkub_line}",
        "",
        "⚠️ Tier 3 = technical only — ไม่รู้เรื่อง news/catalyst",
        "   เพิ่ม ANTHROPIC_API_KEY หรือ GROQ_API_KEY เพื่อ AI analysis",
    ]

    return card + "\n".join(detail_lines)


# ─────────────────────────────────────────────────────────────
#  AI response validation + formatting (Task 4)
# ─────────────────────────────────────────────────────────────

def _validate_ai_response(resp: dict, current_price: float) -> tuple[bool, str]:
    """
    Check 4 hard constraints from the spec.
    Returns (is_valid, error_message).
    """
    # Constraint 1: probabilities sum to 1.0 ±0.01
    try:
        p_sum = (resp["bull_case"]["probability"]
                 + resp["bear_case"]["probability"]
                 + resp["base_case"]["probability"])
        if abs(p_sum - 1.0) > 0.01:
            return False, f"probability sum = {p_sum:.3f} (must be 1.0 ±0.01)"
    except (KeyError, TypeError) as e:
        return False, f"missing probability field: {e}"

    # Constraint 2: direction matches highest-probability case
    try:
        probs = {
            "long":    resp["bull_case"]["probability"],
            "short":   resp["bear_case"]["probability"],
            "neutral": resp["base_case"]["probability"],
        }
        best = max(probs, key=probs.__getitem__)
        if resp.get("direction") != best:
            return False, (f"direction='{resp.get('direction')}' but "
                           f"highest prob is '{best}' ({probs[best]:.2f})")
    except (KeyError, TypeError) as e:
        return False, f"direction/probability check failed: {e}"

    # Constraint 3: invalidation_price on opposite side from target_price
    try:
        target = float(resp["target_price"])
        inval  = float(resp["invalidation_price"])
        if current_price > 0 and target != 0 and inval != 0:
            if (target > current_price) == (inval > current_price):
                return False, (f"target={target} and invalidation={inval} "
                               f"are on same side of price={current_price}")
    except (KeyError, TypeError, ValueError) as e:
        return False, f"target/invalidation check failed: {e}"

    # Constraint 4: confidence ≤ max(bull, bear, base) probability
    try:
        max_prob   = max(resp["bull_case"]["probability"],
                         resp["bear_case"]["probability"],
                         resp["base_case"]["probability"])
        confidence = float(resp["confidence"])
        if confidence > max_prob + 0.01:
            return False, (f"confidence={confidence:.2f} exceeds "
                           f"max_prob={max_prob:.2f}")
    except (KeyError, TypeError, ValueError) as e:
        return False, f"confidence check failed: {e}"

    return True, ""


def _parse_ai_json(raw: str) -> dict | None:
    """Extract and parse JSON from raw AI response (handles markdown code fences)."""
    import json as _json
    text = raw.strip()
    if "```" in text:
        start = text.find("```")
        end   = text.rfind("```")
        inner = text[start + 3:end].strip()
        if inner.startswith("json"):
            inner = inner[4:].strip()
        text = inner
    # Also handle leading/trailing non-JSON characters
    brace_start = text.find("{")
    brace_end   = text.rfind("}")
    if brace_start != -1 and brace_end != -1:
        text = text[brace_start:brace_end + 1]
    try:
        return _json.loads(text)
    except (_json.JSONDecodeError, ValueError):
        return None


def _format_ai_analysis(ai_json: dict, tier_name: str) -> str:
    """Convert validated AI JSON → readable Thai display text."""
    bull = ai_json.get("bull_case",  {})
    bear = ai_json.get("bear_case",  {})
    base = ai_json.get("base_case",  {})
    dir_ = ai_json.get("direction",  "neutral")
    conf = ai_json.get("confidence", 0.0)
    rsn  = ai_json.get("reasoning", "")

    dir_display = {
        "long":    "🟢 LONG",
        "short":   "🔴 SHORT",
        "neutral": "🟡 NEUTRAL",
    }.get(dir_, dir_.upper())

    bp  = round(bull.get("probability", 0) * 100)
    brp = round(bear.get("probability", 0) * 100)
    bap = round(base.get("probability", 0) * 100)

    def _filter_evidence(evs: list) -> list:
        """Remove raw MA price lines already shown in Technical Indicators."""
        filtered = []
        has_ma_note = False
        for ev in evs:
            s = ev.strip().upper()
            is_ma_price = (
                (s.startswith("MA7") or s.startswith("MA25") or s.startswith("MA99"))
                and ("$" in ev or "ABOVE" in s or "BELOW" in s)
            )
            if is_ma_price:
                if not has_ma_note:
                    filtered.append("(MA values — see Technical Indicators above)")
                    has_ma_note = True
            else:
                filtered.append(ev)
        return filtered

    lines = [f"📈 Bull case ({bp}%): {bull.get('thesis', 'N/A')}"]
    for ev in _filter_evidence(bull.get("key_evidence", [])):
        lines.append(f"   • {ev}")

    lines += ["", f"📉 Bear case ({brp}%): {bear.get('thesis', 'N/A')}"]
    for ev in _filter_evidence(bear.get("key_evidence", [])):
        lines.append(f"   • {ev}")

    lines += [
        "",
        f"⚖️  Base case ({bap}%): {base.get('thesis', 'N/A')}",
        "",
        f"🎯 Direction: {dir_display}  |  Confidence: {round(conf * 100)}%",
        "   Target & Invalidation: (see Reconciliation section below)",
    ]
    if rsn:
        lines += ["", f"💬 {rsn}"]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
#  Tier 2: Groq
# ─────────────────────────────────────────────────────────────
def _tier2_groq(data: dict) -> str:
    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError("groq ไม่ได้ติดตั้ง — รัน: pip install groq")

    # Pre-compute everything except targets (direction unknown until AI answers)
    scores                           = _calc_all_scores(data)
    pair, _                          = _normalize_symbol(data["symbol"])
    chg_1h                           = _fetch_1h_change(pair)
    confidence                       = _get_setup_confidence(data)
    expected_pct, horizon, hz_source = _calc_expected_and_horizon(data, _current_horizon)

    _tech2 = data.get("technicals", {})
    _timing = None
    if _TIMING_AVAILABLE:
        _timing = recommend_trading_time(
            phase         = _tech2.get("price_structure", {}).get("phase", ""),
            vol_spike     = _tech2.get("volume_spike", 1.0) or 1.0,
            rsi_4h        = _tech2.get("rsi_14_4h"),
            horizon_days  = horizon,
            symbol        = data["symbol"],
            bitkub_listed = data.get("bitkub", {}).get("listed", False),
            horizon_source= hz_source,
        )

    client        = Groq(api_key=GROQ_API_KEY)
    current_price = data.get("price", {}).get("current", 0) or 0
    user_prompt   = _build_prompt(data, expected_pct, horizon, confidence)
    last_raw      = ""

    ai_json = _FALLBACK_NEUTRAL_RESPONSE
    for attempt in range(3):   # initial + 2 retries
        try:
            messages = [
                {"role": "system", "content": AI_SYSTEM_PROMPT_NEUTRAL},
                {"role": "user",   "content": user_prompt},
            ]
            if attempt > 0 and last_raw:
                messages += [
                    {"role": "assistant", "content": last_raw},
                    {"role": "user",      "content":
                     f"ตอบกลับผิด constraint: {_last_groq_error}\n"
                     "กรุณาตอบใหม่เป็น JSON ที่ถูกต้องตาม schema"},
                ]
            resp    = client.chat.completions.create(
                model=GROQ_MODEL, messages=messages,
                temperature=0.4, max_tokens=1500,
            )
            last_raw = resp.choices[0].message.content
        except Exception as e:
            log.warning("Groq call failed (attempt %d): %s", attempt + 1, e)
            break

        parsed = _parse_ai_json(last_raw)
        if parsed is None:
            _last_groq_error = "invalid JSON"
            log.info("Groq retry %d — invalid JSON", attempt + 1)
            continue

        valid, err = _validate_ai_response(parsed, current_price)
        if valid:
            ai_json = parsed
            break
        _last_groq_error = err
        log.info("Groq retry %d — constraint: %s", attempt + 1, err)
    else:
        log.warning("Groq all retries exhausted — using neutral fallback")

    # Task 7: reconciliation gate
    _calib2      = _get_calibration_for_targets()
    _neutral2    = _get_neutral_score(data)
    ai_json["symbol"] = data.get("symbol", "")
    final_signal = (
        _reconcile(ai_json, _neutral2, _calib2)
        if _RECONCILIATION_AVAILABLE
        else {"action": ai_json.get("direction", "neutral").upper(), **ai_json,
              "neutral_score": _neutral2}
    )
    direction2 = final_signal.get("direction", "neutral")

    # Task 5: compute targets using reconciled direction
    targets = _calc_targets_new(data, direction2, _calib2)

    card = _format_trading_card(
        data, f"Tier 2 — Groq ({GROQ_MODEL})", scores, targets,
        chg_1h, confidence, expected_pct, horizon, hz_source, _timing
    )

    ai_text = _format_ai_analysis(ai_json, f"Groq ({GROQ_MODEL})")
    ai_section = (
        "\n────────────────────────────────────────────────────────────\n"
        "🦙 AI ANALYSIS (Groq Llama3):\n"
        f"{ai_text}\n"
    )
    recon_section = (
        format_reconcile_section(final_signal, targets)
        if _RECONCILIATION_AVAILABLE else ""
    )
    return card + ai_section + recon_section


# ─────────────────────────────────────────────────────────────
#  Tier 1: Claude
# ─────────────────────────────────────────────────────────────
def _tier1_claude(data: dict) -> str:
    import anthropic

    # Pre-compute everything except targets (direction unknown until AI answers)
    scores                           = _calc_all_scores(data)
    pair, _                          = _normalize_symbol(data["symbol"])
    chg_1h                           = _fetch_1h_change(pair)
    confidence                       = _get_setup_confidence(data)
    expected_pct, horizon, hz_source = _calc_expected_and_horizon(data, _current_horizon)

    _tech1 = data.get("technicals", {})
    _timing = None
    if _TIMING_AVAILABLE:
        _timing = recommend_trading_time(
            phase         = _tech1.get("price_structure", {}).get("phase", ""),
            vol_spike     = _tech1.get("volume_spike", 1.0) or 1.0,
            rsi_4h        = _tech1.get("rsi_14_4h"),
            horizon_days  = horizon,
            symbol        = data["symbol"],
            bitkub_listed = data.get("bitkub", {}).get("listed", False),
            horizon_source= hz_source,
        )

    client        = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    current_price = data.get("price", {}).get("current", 0) or 0
    user_prompt   = _build_prompt(data, expected_pct, horizon, confidence)
    last_raw      = ""

    ai_json = _FALLBACK_NEUTRAL_RESPONSE
    for attempt in range(3):   # initial + 2 retries
        try:
            msgs = [{"role": "user", "content": user_prompt}]
            if attempt > 0 and last_raw:
                msgs += [
                    {"role": "assistant", "content": last_raw},
                    {"role": "user",      "content":
                     f"ตอบกลับผิด constraint: {_last_claude_error}\n"
                     "กรุณาตอบใหม่เป็น JSON ที่ถูกต้องตาม schema"},
                ]
            resp     = client.messages.create(
                model=CLAUDE_MODEL, max_tokens=1500,
                system=AI_SYSTEM_PROMPT_NEUTRAL, messages=msgs,
            )
            last_raw = resp.content[0].text
        except Exception as e:
            log.warning("Claude call failed (attempt %d): %s", attempt + 1, e)
            break

        parsed = _parse_ai_json(last_raw)
        if parsed is None:
            _last_claude_error = "invalid JSON"
            log.info("Claude retry %d — invalid JSON", attempt + 1)
            continue

        valid, err = _validate_ai_response(parsed, current_price)
        if valid:
            ai_json = parsed
            break
        _last_claude_error = err
        log.info("Claude retry %d — constraint: %s", attempt + 1, err)
    else:
        log.warning("Claude all retries exhausted — using neutral fallback")

    # Task 7: reconciliation gate
    _calib1      = _get_calibration_for_targets()
    _neutral1    = _get_neutral_score(data)
    ai_json["symbol"] = data.get("symbol", "")
    final_signal1 = (
        _reconcile(ai_json, _neutral1, _calib1)
        if _RECONCILIATION_AVAILABLE
        else {"action": ai_json.get("direction", "neutral").upper(), **ai_json,
              "neutral_score": _neutral1}
    )
    direction1 = final_signal1.get("direction", "neutral")

    # Task 5: compute targets using reconciled direction
    targets = _calc_targets_new(data, direction1, _calib1)

    card = _format_trading_card(
        data, f"Tier 1 — Claude ({CLAUDE_MODEL})", scores, targets,
        chg_1h, confidence, expected_pct, horizon, hz_source, _timing
    )

    ai_text = _format_ai_analysis(ai_json, f"Claude ({CLAUDE_MODEL})")
    ai_section = (
        "\n────────────────────────────────────────────────────────────\n"
        f"🧠 AI ANALYSIS (Claude {CLAUDE_MODEL}):\n"
        f"{ai_text}\n"
    )
    recon_section1 = (
        format_reconcile_section(final_signal1, targets)
        if _RECONCILIATION_AVAILABLE else ""
    )
    return card + ai_section + recon_section1


# ─────────────────────────────────────────────────────────────
#  Main analyze
# ─────────────────────────────────────────────────────────────
def analyze(symbol: str, data: dict | None = None) -> dict:
    _ensure_backtest_fresh()
    _, base = _normalize_symbol(symbol)

    if data is None:
        json_path = os.path.join(OUTPUT_DIR, f"{base}.json")
        if os.path.exists(json_path):
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            log.info("Loaded data from %s", json_path)
        else:
            log.info("No cached data for %s — fetching...", base)
            data = fetch_crypto(symbol)
            save_json(data, base)

    if "error" in data:
        return {"symbol": base, "error": data["error"], "tier": "none"}

    tier = get_analysis_tier()
    analysis_text = ""
    used_tier = ""

    if tier == "tier1_claude":
        print(f"\n🧠 กำลังใช้ [Tier 1 — Claude {CLAUDE_MODEL}] วิเคราะห์อยู่...")
        try:
            analysis_text = _tier1_claude(data)
            used_tier = f"Tier 1 — Claude ({CLAUDE_MODEL})"
        except Exception as e:
            log.warning("Tier 1 failed: %s", e)
            tier = "tier2_groq" if GROQ_API_KEY else "tier3_technical"

    if tier == "tier2_groq" and not analysis_text:
        print(f"\n🦙 กำลังใช้ [Tier 2 — Groq Llama3 (ฟรี)] วิเคราะห์อยู่...")
        try:
            analysis_text = _tier2_groq(data)
            used_tier = f"Tier 2 — Groq ({GROQ_MODEL})"
        except Exception as e:
            log.warning("Tier 2 failed: %s", e)

    if not analysis_text:
        print("\n📊 กำลังใช้ [Tier 3 — Technical Rules] วิเคราะห์อยู่...")
        analysis_text = _tier3_analysis(data)
        used_tier = "Tier 3 — Technical Rules"

    # คำนวณ forecast สำหรับเก็บใน result (ใช้ค่าที่คำนวณไว้แล้วจาก tier functions)
    _exp, _hz, _hz_src = _calc_expected_and_horizon(data, _current_horizon)
    _conf = _get_setup_confidence(data)

    _timing_json = None
    if _TIMING_AVAILABLE:
        _t = data.get("technicals", {})
        _timing_json = recommend_trading_time(
            phase         = _t.get("price_structure", {}).get("phase", ""),
            vol_spike     = _t.get("volume_spike", 1.0) or 1.0,
            rsi_4h        = _t.get("rsi_14_4h"),
            horizon_days  = _hz,
            symbol        = base,
            bitkub_listed = data.get("bitkub", {}).get("listed", False),
            horizon_source= _hz_src,
        )

    # Build forecast block — compatible with both legacy (float) and new (dict) modes
    if isinstance(_exp, dict):
        _forecast_block = {
            "expected_pct":   _exp.get("upper"),   # backward-compat field (upper bound)
            "upper_pct":      _exp.get("upper"),
            "lower_pct":      _exp.get("lower"),
            "mode":           "new",
        }
    else:
        _forecast_block = {
            "expected_pct":   _exp,
            "upper_pct":      None,
            "lower_pct":      None,
            "mode":           "legacy",
        }
    _forecast_block.update({
        "horizon_days":   _hz,
        "horizon_source": _hz_src,
        "accuracy_pct":   _conf,
    })

    result = {
        "symbol":        base,
        "timestamp":     datetime.utcnow().isoformat() + "Z",
        "tier_used":     used_tier,
        "analysis":      analysis_text,
        "forecast":      _forecast_block,
        "neutral_score": None,   # populated by tier 1/2 via reconciliation
        "timing": _timing_json,
        "data_snapshot": {
            "price":            data.get("price", {}).get("current"),
            "change_24h_pct":   data.get("price", {}).get("change_24h_pct"),
            "change_7d_pct":    data.get("market_data", {}).get("change_7d_pct"),
            "rsi_14":           data.get("technicals", {}).get("rsi_14"),
            "rsi_14_4h":        data.get("technicals", {}).get("rsi_14_4h"),
            "volume_spike":     data.get("technicals", {}).get("volume_spike"),
            "phase":            data.get("technicals", {}).get("price_structure", {}).get("phase"),
            "opportunity_grade": data.get("opportunity_grade"),
            "opportunity_score": data.get("opportunity_score"),
            "bitkub_listed":    data.get("bitkub", {}).get("listed", False),
            "bitkub_price_thb": data.get("bitkub", {}).get("price_thb"),
        },
    }

    out_path = os.path.join(OUTPUT_DIR, f"{base}_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────
def main():
    _tf_choices = list(TIMEFRAME_PRESETS.keys()) if _TIMING_AVAILABLE else []

    parser = argparse.ArgumentParser(description="Crypto Opportunity Analyzer (3-tier AI)")
    parser.add_argument("--symbol",    type=str, required=True, help="Symbol e.g. BTC, ETH, SOLUSDT")
    parser.add_argument("--fetch",     action="store_true", help="Force re-fetch data first")
    parser.add_argument("--horizon",   type=int, default=None,
                        help="ขอบเขตเวลา (วัน) เช่น 1=24hr, 7=1สัปดาห์ (default: auto จาก setup)")
    parser.add_argument("--timeframe", type=str, default=None,
                        choices=_tf_choices or None,
                        metavar="TIMEFRAME",
                        help=f"กรอบเวลา: scalp(1d) short(3d) swing(7d) position(14d) monthly(30d)")
    args = parser.parse_args()

    global _current_horizon, _current_timeframe
    _current_horizon   = args.horizon
    _current_timeframe = args.timeframe

    # --timeframe sets horizon if --horizon not given explicitly
    if _TIMING_AVAILABLE and args.timeframe and not args.horizon:
        mapped = get_horizon_from_timeframe(args.timeframe)
        if mapped:
            _current_horizon = mapped
            print(f"⏰ Timeframe '{args.timeframe}' → horizon {mapped} วัน")

    data = None
    if args.fetch:
        print(f"📡 Fetching fresh data for {args.symbol}...")
        data = fetch_crypto(args.symbol)
        _, base = _normalize_symbol(args.symbol)
        save_json(data, base)

        from data_fetcher import print_crypto_summary
        print_crypto_summary(data)

    result = analyze(args.symbol, data)

    print()
    print(result["analysis"])

    print(f"\n✅ Analysis saved: {OUTPUT_DIR}/{result['symbol']}_analysis.json\n")


if __name__ == "__main__":
    main()
