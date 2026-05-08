"""
trading_time.py — Trading Time Window Recommendations (ICT = UTC+7)
แนะนำช่วงเวลาซื้อขายที่ดีที่สุด ตามลักษณะ setup และ timeframe
"""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from typing import Optional

_ICT = timezone(timedelta(hours=7))

# Timeframe presets → horizon days
TIMEFRAME_PRESETS: dict[str, int] = {
    "scalp":    1,
    "short":    3,
    "swing":    7,
    "position": 14,
    "monthly":  30,
}

# Session bounds (ICT hours, float; 24+ means next-day AM)
_SESSION_BOUNDS: dict[str, tuple[float, float]] = {
    "ASIA_OPEN":    (8.0,  10.0),
    "ASIA":        (10.0,  15.0),
    "LONDON_OPEN": (15.0,  17.0),
    "LONDON":      (17.0,  20.5),
    "US_OPEN":     (20.5,  22.0),
    "US":          (22.0,  29.0),   # 29 = 05:00 next day
    "DEAD_ZONE":   (5.0,   8.0),
}

_SESSION_LABELS: dict[str, tuple[str, str]] = {
    # key: (short_label, full_label)
    "ASIA_OPEN":    ("Asian Open",    "Asian Open (08:00-10:00 ICT)"),
    "ASIA":         ("Asian Session", "Asian Session (10:00-15:00 ICT)"),
    "LONDON_OPEN":  ("London Open",   "London Open (15:00-17:00 ICT)"),
    "LONDON":       ("London",        "London Session (17:00-20:30 ICT)"),
    "US_OPEN":      ("US Open",       "US Open (20:30-22:00 ICT)"),
    "US":           ("US Session",    "US Session (22:00-05:00 ICT)"),
    "DEAD_ZONE":    ("Dead Zone",     "Dead Zone (05:00-08:00 ICT)"),
}

# Bitkub-native coins: Thai traders more active during Asian hours
_BITKUB_NATIVE = frozenset({"KUB", "SIX", "JFIN", "ERN"})


def get_horizon_from_timeframe(tf: str) -> Optional[int]:
    """แปลง timeframe string → จำนวนวัน  e.g. "swing" → 7"""
    return TIMEFRAME_PRESETS.get(tf.lower())


# ─────────────────────────────────────────────────────────────
#  Internal time helpers
# ─────────────────────────────────────────────────────────────
def _ict_now_hour() -> float:
    t = datetime.now(_ICT)
    return t.hour + t.minute / 60.0


def _to_linear(h: float) -> float:
    """Convert ICT hour to linear (05:00 = base; hours before 05:00 → +24)"""
    return h if h >= 5.0 else h + 24.0


def _is_in_session(key: str) -> bool:
    now_lin = _to_linear(_ict_now_hour())
    bounds  = _SESSION_BOUNDS.get(key)
    if not bounds:
        return False
    s, e = bounds
    return s <= now_lin < e


def _hours_until(session_start: float) -> float:
    """Hours from now until next occurrence of session_start (ICT)"""
    now_lin    = _to_linear(_ict_now_hour())
    target_lin = _to_linear(session_start)
    diff = target_lin - now_lin
    if diff < 0:
        diff += 24.0
    return round(diff, 1)


def _fmt_hours(h: float) -> str:
    if h == 0.0:
        return "กำลังอยู่ใน session นี้แล้ว ✓"
    if h < 0.5:
        return "อีกไม่กี่นาที"
    if h < 1.0:
        return f"{int(h * 60)} นาที"
    return f"{h:.1f} ชม."


# ─────────────────────────────────────────────────────────────
#  Main recommendation function
# ─────────────────────────────────────────────────────────────
def recommend_trading_time(
    phase: str,
    vol_spike: float,
    rsi_4h: Optional[float],
    horizon_days: int,
    symbol: str = "",
    bitkub_listed: bool = False,
    horizon_source: str = "auto",
) -> dict:
    """
    Returns trading time recommendation based on setup characteristics.

    Returned keys:
      best_session        — full label e.g. "US Open (20:30-22:00 ICT)"
      best_session_short  — short label  e.g. "US Open"
      secondary_session   — fallback session full label
      entry_condition     — what to wait for before pulling the trigger
      avoid_window        — time window to avoid
      horizon_label       — human-friendly timeframe label
      urgency             — "HIGH" | "MEDIUM" | "LOW"
      urgency_display     — with emoji
      next_window_str     — e.g. "กำลังอยู่ใน session นี้แล้ว ✓" or "2.5 ชม."
    """
    phase     = phase or ""
    vol_spike = vol_spike or 1.0
    sym_upper = symbol.upper()
    is_native = sym_upper in _BITKUB_NATIVE

    # ── Urgency ────────────────────────────────────────────────
    if phase == "VOLATILE_NO_STRUCTURE":
        urgency = "LOW"
    elif phase == "TIGHT_RANGE_HIGHER_LOWS":
        urgency = "HIGH"
    elif phase == "CONSOLIDATING_HIGHER_LOWS" and vol_spike >= 2.0:
        urgency = "HIGH"
    elif vol_spike >= 3.0:
        urgency = "HIGH"
    elif phase in ("CONSOLIDATING_HIGHER_LOWS", "UPTREND_PULLBACK") or vol_spike >= 1.5:
        urgency = "MEDIUM"
    elif phase == "CONSOLIDATING_FLAT":
        urgency = "LOW"
    else:
        urgency = "MEDIUM"

    # Fix 3: RSI 4h > 80 (overbought) must not show HIGH urgency
    if rsi_4h is not None and rsi_4h > 80 and urgency == "HIGH":
        urgency = "MEDIUM"

    # ── Best session selection ──────────────────────────────────
    if is_native:
        # Thai-native Bitkub coins → Asian traders dominate
        best_key = "ASIA_OPEN"
        sec_key  = "US_OPEN"
    elif horizon_days <= 2:
        # Scalp / day-trade → US open has deepest liquidity
        best_key = "US_OPEN"
        sec_key  = "ASIA_OPEN"
    elif phase == "TIGHT_RANGE_HIGHER_LOWS":
        # Pre-breakout coil → needs US volume to ignite
        best_key = "US_OPEN"
        sec_key  = "LONDON_OPEN"
    elif phase == "UPTREND_PULLBACK":
        # Buy-the-dip → Asian open provides cheaper entry
        best_key = "ASIA_OPEN"
        sec_key  = "US_OPEN"
    elif phase == "CONSOLIDATING_FLAT":
        # Waiting game → either open is fine; London often breaks range
        best_key = "LONDON_OPEN"
        sec_key  = "US_OPEN"
    else:
        best_key = "US_OPEN"
        sec_key  = "ASIA_OPEN"

    # ── Entry condition ─────────────────────────────────────────
    _cond_map = {
        "TIGHT_RANGE_HIGHER_LOWS":   "รอ candle ปิดเหนือ resistance + volume ≥ 1.5×avg",
        "CONSOLIDATING_HIGHER_LOWS": "รอ volume spike ≥ 1.5× และ candle ปิดเหนือ resistance",
        "UPTREND_PULLBACK":          "เข้าที่แนวรับ / retest MA25 หรือ breakout level เดิม",
        "CONSOLIDATING_FLAT":        "รอ breakout พร้อม volume spike ≥ 2×avg ก่อนเข้า",
        "VOLATILE_NO_STRUCTURE":     "⚠️ ไม่แนะนำ — รอ structure ชัดเจนก่อน",
    }
    cond = _cond_map.get(phase, "รอ volume confirm + candlestick signal ยืนยัน")

    if rsi_4h is not None:
        if rsi_4h > 75:
            cond += "  | ⚠️ RSI 4h overbought — รอ pullback ก่อน"
        elif rsi_4h < 30:
            cond += "  | 🎣 RSI 4h oversold — possible reversal setup"
        elif 40 <= rsi_4h <= 60:
            cond += "  | RSI 4h neutral — รอทิศทาง breakout"

    # ── Horizon label ───────────────────────────────────────────
    src_tag = " (กำหนดเอง)" if horizon_source == "user" else " (auto)"
    if horizon_days == 1:
        h_label = f"Day Trade (1 วัน){src_tag}"
    elif horizon_days <= 3:
        h_label = f"Short-term ({horizon_days} วัน){src_tag}"
    elif horizon_days <= 7:
        h_label = f"Swing Trade ({horizon_days} วัน){src_tag}"
    elif horizon_days <= 14:
        h_label = f"Swing-Position ({horizon_days} วัน){src_tag}"
    else:
        h_label = f"Position Trade ({horizon_days} วัน){src_tag}"

    # ── Avoid window ────────────────────────────────────────────
    if phase == "VOLATILE_NO_STRUCTURE" or urgency == "LOW":
        avoid = "ทุก session จนกว่า structure จะชัดเจน"
    elif horizon_days <= 2:
        avoid = "05:00-08:00 ICT (dead zone) | หลีกเลี่ยง London fake-out 15:00-16:00 ICT"
    else:
        avoid = "05:00-08:00 ICT (low-volume dead zone)"

    # ── Next window countdown ───────────────────────────────────
    best_start = _SESSION_BOUNDS.get(best_key, (20.5, 22.0))[0]
    if _is_in_session(best_key):
        hours_to   = 0.0
        next_str   = "กำลังอยู่ใน session นี้แล้ว ✓"
    else:
        hours_to = _hours_until(best_start)
        next_str = f"ใน {_fmt_hours(hours_to)}"

    # ── Urgency display ─────────────────────────────────────────
    urgency_disp = {
        "HIGH":   "🔴 HIGH — เข้าได้เลย อย่ารอนาน",
        "MEDIUM": "🟡 MEDIUM — ตั้ง alert รอ signal",
        "LOW":    "🟢 LOW — รอ setup ดีกว่า",
    }.get(urgency, urgency)

    best_short = _SESSION_LABELS.get(best_key, ("", ""))[0]
    best_full  = _SESSION_LABELS.get(best_key, ("", best_key))[1]
    sec_full   = _SESSION_LABELS.get(sec_key,  ("", sec_key))[1]

    return {
        "best_session":       best_full,
        "best_session_short": best_short,
        "secondary_session":  sec_full,
        "entry_condition":    cond,
        "avoid_window":       avoid,
        "horizon_label":      h_label,
        "urgency":            urgency,
        "urgency_display":    urgency_disp,
        "hours_to_next":      hours_to,
        "next_window_str":    next_str,
    }
