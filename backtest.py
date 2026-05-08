"""
backtest.py — Historical Accuracy Backtester
ทดสอบความแม่นยำของการพยากรณ์ทิศทาง+ขนาด จากข้อมูล OHLCV จริงย้อนหลัง

Logic:
  สำหรับแต่ละ bar i:
    1. คำนวณ indicator ณ วันนั้น (ไม่มี look-ahead)
    2. ใช้ _calc_all_scores() จาก analyzer.py → composite score (0-100)
    3. ดูราคาปิดจริงหลัง horizon วัน → actual_pct
    4. Win  = actual_pct >= MIN_PROFIT_THRESHOLD (3%)
       Loss = actual_pct < MIN_PROFIT_THRESHOLD
  รวม win rate ตาม Grade / Phase / RSI zone / Volume spike / MA alignment / Score range
  บันทึกลง output/backtest_results.json

Usage:
  python backtest.py                        # backtest 35 เหรียญ, horizon auto
  python backtest.py --symbols BTC ETH SOL  # custom symbols
  python backtest.py --horizon 1            # กำหนด horizon 1 วัน (24hr trading)
  python backtest.py --horizon 7            # horizon 7 วัน
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone

from config import OUTPUT_DIR, LOG_LEVEL
from data_fetcher import (
    _binance_get,
    _calc_rsi,
    _analyze_ohlcv_structure,
)

logging.basicConfig(level=getattr(logging, LOG_LEVEL, "INFO"),
                    format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

BACKTEST_FILE        = os.path.join(OUTPUT_DIR, "backtest_results.json")
MIN_SETUPS           = 5    # setups ขั้นต่ำในกลุ่มถึงจะรายงาน win rate
ATR_WIN_MULTIPLIER   = 1.0  # win = actual_pct >= 1× ATR%
MIN_HISTORY_BARS     = 120  # bars ขั้นต่ำก่อนเริ่ม simulate

DEFAULT_SYMBOLS = [
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE",
    "AVAX", "MATIC", "DOT", "LINK", "UNI", "LTC", "ATOM",
    "NEAR", "APT", "SUI", "INJ", "ARB", "OP", "AAVE",
    "TRX", "XLM", "ICP", "FET", "RENDER",
    "SAND", "MANA", "GALA", "AXS", "GRT", "FTM", "CRV",
    "PEPE", "SHIB",
]


# ─────────────────────────────────────────────────────────────
#  Setup classification helpers (ใช้ร่วมกับ analyzer.py)
# ─────────────────────────────────────────────────────────────

def _rsi_zone(rsi: float | None) -> str:
    if rsi is None: return "unknown"
    if rsi < 35:    return "oversold"
    if rsi < 50:    return "recovering"
    if rsi <= 68:   return "momentum"
    if rsi <= 78:   return "hot"
    return "overbought"


def _vol_spike_band(spike: float) -> str:
    if spike >= 3.0: return "3x+"
    if spike >= 2.0: return "2-3x"
    if spike >= 1.5: return "1.5-2x"
    if spike >= 1.2: return "1.2-1.5x"
    return "<1.2x"


def _ma_align_label(above_ma25: bool | None, above_ma99: bool | None) -> str:
    if above_ma25 and above_ma99: return "both"
    if above_ma99:                return "ma99_only"
    if above_ma25:                return "ma25_only"
    return "neither"


def _opportunity_grade(composite: int) -> str:
    if composite >= 70: return "A"
    if composite >= 65: return "B+"
    if composite >= 60: return "B"
    if composite >= 55: return "C+"
    if composite >= 50: return "C"
    if composite >= 45: return "D+"
    if composite >= 35: return "D"
    return "F"


def _score_bucket(score: int) -> str:
    base = (score // 10) * 10
    return f"{base}-{base + 9}"


# ─────────────────────────────────────────────────────────────
#  Horizon + Expected % calculation
# ─────────────────────────────────────────────────────────────

def estimate_horizon(phase: str, vol_spike: float) -> int:
    """
    Auto-estimate holding horizon จาก setup ถ้า user ไม่กำหนด
    Momentum signal มีอายุสั้น: vol spike หมดใน 24 ชม., breakout วิ่งเร็ว
    """
    if vol_spike >= 2.0:
        return 1    # spike = วิ่งทันที ต้องเก็บผลใน 1 วัน
    if phase == "TIGHT_RANGE_HIGHER_LOWS":
        return 2    # coiling แน่น → breakout เร็ว
    if phase == "CONSOLIDATING_HIGHER_LOWS":
        return 3
    if phase == "UPTREND_PULLBACK":
        return 3
    if phase == "CONSOLIDATING_FLAT":
        return 5
    return 3        # default


def _calc_expected_pct_legacy(
    chg_24h: float,
    chg_7d: float,
    rsi_4h: float | None,
    vol_spike: float,
    horizon: int,
) -> float:
    """Legacy momentum extrapolation (bullish-biased). Kept for USE_LEGACY_FORECASTER A/B."""
    daily_7d = chg_7d / 7.0 if chg_7d else 0.0
    alpha = max(0.2, min(0.8, 1.0 - (horizon - 1) * 0.09))
    beta  = 1.0 - alpha
    daily_rate = alpha * chg_24h + beta * daily_7d
    raw        = daily_rate * horizon
    if vol_spike >= 3.0:   raw *= 1.30
    elif vol_spike >= 2.0: raw *= 1.20
    elif vol_spike >= 1.5: raw *= 1.10
    elif vol_spike < 0.8:  raw *= 0.80
    if horizon <= 3 and rsi_4h is not None:
        if rsi_4h >= 65:   raw *= 1.10
        elif rsi_4h < 40:  raw *= 0.85
    return round(max(-30.0, min(40.0, raw)), 2)


def calc_expected_pct(
    coin: str,
    horizon_days: int,
    rsi: float | None,
    return_24h_frac: float,
    calibration: dict,
) -> dict:
    """
    Returns symmetric volatility-based expected range in PERCENTAGE.
    {"upper": float, "lower": float}  e.g. upper=4.3 means +4.3%

    Bounds derive from 2× realized sigma (data-driven, never hardcoded ±30/40).
    Only dampens — never boosts — so |upper| can only shrink, never grow past 2σ.
    Justification for |upper| > |lower|: rsi < 30 dampens downside (oversold).
    """
    from calibration import get_sigma

    sigma  = get_sigma(coin, horizon_days, calibration)   # fraction e.g. 0.043
    upper  = +2.0 * sigma * 100   # → percentage
    lower  = -2.0 * sigma * 100

    # Mean-reversion dampening: large 24h move → continuation less likely
    daily_sigma = get_sigma(coin, 1, calibration)
    if abs(return_24h_frac) > 1.5 * daily_sigma:
        upper *= 0.5
        lower *= 0.5

    # RSI dampening — symmetric, only reduces relevant side, never boosts
    if rsi is not None:
        if rsi > 70:
            upper *= 0.85           # overbought → shrink upside range
        if rsi < 30:
            lower *= 0.85           # oversold → shrink downside range (magnitude)

    return {"upper": round(upper, 2), "lower": round(lower, 2)}


# ─────────────────────────────────────────────────────────────
#  Indicator helpers — ไม่มี look-ahead
# ─────────────────────────────────────────────────────────────

def _calc_atr14(highs: list[float], lows: list[float]) -> float | None:
    if len(highs) < 14:
        return None
    return sum(highs[i] - lows[i] for i in range(-14, 0)) / 14


def _calc_ma_flags(closes: list[float], price: float) -> tuple[bool | None, bool | None]:
    ma25 = sum(closes[-25:]) / 25 if len(closes) >= 25 else None
    ma99 = sum(closes[-99:]) / 99 if len(closes) >= 99 else None
    return (
        price > ma25 if ma25 is not None else None,
        price > ma99 if ma99 is not None else None,
    )


def _calc_rsi_4h_approx(closes_daily: list[float]) -> float | None:
    """
    ประมาณ RSI 4h จาก daily closes (6 bars ≈ 1 วัน)
    ใช้ 4 daily bars ล่าสุดแทน (crude approximation สำหรับ backtest)
    """
    if len(closes_daily) < 6:
        return None
    # ใช้ 4 closes ล่าสุดเป็น proxy ของ intraday
    from data_fetcher import _calc_rsi
    return _calc_rsi(closes_daily[-20:], period=4)


# ─────────────────────────────────────────────────────────────
#  Core backtester
# ─────────────────────────────────────────────────────────────

def backtest_symbol(
    symbol: str,
    klines: list,
    fixed_horizon: int | None = None,
    calibration: dict | None = None,
) -> list[dict]:
    """
    Walk through klines bar-by-bar ตั้งแต่ MIN_HISTORY_BARS
    Win = actual N-day return อยู่ใน [expected_pct − 5%, expected_pct + 5%]
    Loss = อยู่นอกช่วงนั้น
    """
    results = []
    max_horizon = fixed_horizon or 14  # ต้องมี buffer ข้างหน้าพอ

    for i in range(MIN_HISTORY_BARS, len(klines) - max_horizon - 1):
        hist    = klines[: i + 1]
        closes  = [float(k[4]) for k in hist]
        highs   = [float(k[2]) for k in hist]
        lows    = [float(k[3]) for k in hist]
        volumes = [float(k[5]) for k in hist]

        entry = closes[-1]
        if entry <= 0:
            continue

        # Indicators ณ bar i (ไม่มี look-ahead)
        rsi            = _calc_rsi(closes)
        rsi_4h_approx  = _calc_rsi_4h_approx(closes)
        above_ma25, above_ma99 = _calc_ma_flags(closes, entry)
        phase = _analyze_ohlcv_structure(
            closes[-30:], volumes[-30:]
        ).get("phase", "UNKNOWN")

        baseline  = volumes[max(0, len(volumes) - 15): -1]
        avg_vol   = sum(baseline) / len(baseline) if baseline else volumes[-1]
        vol_spike = volumes[-1] / avg_vol if avg_vol > 0 else 1.0

        chg_24h = (
            (closes[-1] - closes[-2]) / closes[-2] * 100
            if len(closes) >= 2 and closes[-2] > 0 else 0.0
        )
        chg_7d = (
            (closes[-1] - closes[-8]) / closes[-8] * 100
            if len(closes) >= 8 and closes[-8] > 0 else 0.0
        )

        # Horizon
        horizon = fixed_horizon if fixed_horizon else estimate_horizon(phase, vol_spike)

        # Expected % สำหรับ horizon นี้
        from config import USE_LEGACY_FORECASTER
        if USE_LEGACY_FORECASTER or calibration is None:
            expected = _calc_expected_pct_legacy(chg_24h, chg_7d, rsi_4h_approx, vol_spike, horizon)
        else:
            expected = calc_expected_pct(symbol, horizon, rsi, chg_24h / 100.0, calibration)

        # Actual return หลัง horizon วัน (ดูจาก closing price)
        if i + horizon >= len(klines):
            continue
        future_close = float(klines[i + horizon][4])
        actual_pct   = (future_close - entry) / entry * 100

        atr14    = _calc_atr14(highs, lows)
        atr_pct  = (atr14 / entry * 100) if atr14 and entry > 0 else 3.0
        win_threshold = round(atr_pct * ATR_WIN_MULTIPLIER, 2)

        # Win = actual_pct >= 1× ATR% (scale ตามความผันผวนของแต่ละเหรียญ)
        outcome = "win" if actual_pct >= win_threshold else "loss"

        # ใช้ _calc_all_scores จาก analyzer.py (lazy import หลีกเลี่ยง circular import)
        from analyzer import _calc_all_scores
        data_dict = {
            "symbol": symbol,
            "price": {
                "current":          entry,
                "change_24h_pct":   chg_24h,
                "volume_24h_usdt":  volumes[-1] * entry,
                "low_24h":          lows[-1],
                "high_24h":         highs[-1],
            },
            "technicals": {
                "rsi_14":           rsi,
                "rsi_14_4h":        rsi_4h_approx,
                "above_ma25":       above_ma25,
                "above_ma99":       above_ma99,
                "atr_14":           atr14,
                "volume_spike":     vol_spike,
                "pct_from_high":    0,
                "price_structure":  {"phase": phase},
                "macd":             None,
                "macd_signal":      None,
                "golden_cross":     None,
            },
            "market_data": {
                "market_cap_usd":   None,
                "change_7d_pct":    chg_7d,
            },
        }
        composite = _calc_all_scores(data_dict)["composite"]

        # Store expected_pct as scalar for backward compat; keep full range when available
        if isinstance(expected, dict):
            expected_pct_val   = expected.get("upper", 0)
            expected_range_val = expected
        else:
            expected_pct_val   = expected
            expected_range_val = None

        results.append({
            "symbol":         symbol,
            "grade":          _opportunity_grade(composite),
            "score":          composite,
            "score_bucket":   _score_bucket(composite),
            "phase":          phase,
            "rsi_zone":       _rsi_zone(rsi),
            "vol_spike_band": _vol_spike_band(vol_spike),
            "ma_align":       _ma_align_label(above_ma25, above_ma99),
            "horizon":        horizon,
            "expected_pct":   expected_pct_val,
            "expected_range": expected_range_val,
            "actual_pct":     round(actual_pct, 2),
            "outcome":        outcome,
        })

    return results


# ─────────────────────────────────────────────────────────────
#  Aggregation
# ─────────────────────────────────────────────────────────────

def _win_rate(items: list[dict]) -> tuple[float | None, int]:
    if len(items) < MIN_SETUPS:
        return None, len(items)
    wins = sum(1 for r in items if r["outcome"] == "win")
    return round(wins / len(items) * 100, 1), len(items)


def _aggregate(all_results: list[dict]) -> dict:
    by: dict[str, dict] = {
        "by_grade":       defaultdict(list),
        "by_phase":       defaultdict(list),
        "by_rsi_zone":    defaultdict(list),
        "by_vol_spike":   defaultdict(list),
        "by_ma_align":    defaultdict(list),
        "by_score_range": defaultdict(list),
    }
    for r in all_results:
        by["by_grade"][r["grade"]].append(r)
        by["by_phase"][r["phase"]].append(r)
        by["by_rsi_zone"][r["rsi_zone"]].append(r)
        by["by_vol_spike"][r["vol_spike_band"]].append(r)
        by["by_ma_align"][r["ma_align"]].append(r)
        by["by_score_range"][r["score_bucket"]].append(r)

    output: dict = {}
    for category, buckets in by.items():
        output[category] = {}
        for key, items in buckets.items():
            rate, n = _win_rate(items)
            if rate is not None:
                output[category][key] = {"win_rate": rate, "n": n}
    return output


# ─────────────────────────────────────────────────────────────
#  Main runner
# ─────────────────────────────────────────────────────────────

def run_backtest(
    symbols: list[str] | None = None,
    fixed_horizon: int | None = None,
    verbose: bool = True,
) -> dict:
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    # Load calibration once for new forecaster (skip if legacy mode or file missing)
    from config import USE_LEGACY_FORECASTER
    _calibration: dict | None = None
    if not USE_LEGACY_FORECASTER:
        try:
            from calibration import CALIBRATION_FILE
            if os.path.exists(CALIBRATION_FILE):
                import json as _json
                with open(CALIBRATION_FILE, encoding="utf-8") as _f:
                    _calibration = _json.load(_f)
        except Exception as _e:
            log.warning("Could not load calibration (%s) — using legacy forecaster", _e)

    all_results: list[dict] = []

    for sym in symbols:
        pair = f"{sym}USDT"
        if verbose:
            print(f"  📊 {sym:<8}", end=" ", flush=True)

        klines = _binance_get(
            "/klines", {"symbol": pair, "interval": "1d", "limit": 500}
        )
        if not klines or len(klines) < MIN_HISTORY_BARS + (fixed_horizon or 14) + 10:
            if verbose:
                print("⚠️  ข้อมูลไม่พอ ข้ามไป")
            continue

        results = backtest_symbol(sym, klines, fixed_horizon, _calibration)
        all_results.extend(results)

        if verbose:
            wins   = sum(1 for r in results if r["outcome"] == "win")
            wr_str = f"{wins / len(results) * 100:.1f}%" if results else "N/A"
            print(f"{len(results):>4} setups  |  accuracy: {wr_str}")

    aggregates  = _aggregate(all_results)
    total_wins  = sum(1 for r in all_results if r["outcome"] == "win")
    overall_acc = round(total_wins / len(all_results) * 100, 1) if all_results else None

    output = {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "symbols_tested":     symbols,
        "total_setups":       len(all_results),
        "overall_accuracy":   overall_acc,
        "fixed_horizon":      fixed_horizon,
        "atr_win_multiplier":   ATR_WIN_MULTIPLIER,
        "aggregates":         aggregates,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(BACKTEST_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    if verbose:
        _print_summary(aggregates, len(all_results), overall_acc, len(symbols), fixed_horizon)
        print(f"\n✅ บันทึกแล้ว: {BACKTEST_FILE}\n")

    return output


def _print_summary(
    aggregates: dict,
    total: int,
    overall_acc: float | None,
    n_coins: int,
    fixed_horizon: int | None,
) -> None:
    hz_str  = f"{fixed_horizon} วัน" if fixed_horizon else "auto (ตาม setup)"
    acc_str = f"{overall_acc:.1f}%" if overall_acc is not None else "N/A"
    print(f"\n  {'─' * 54}")
    print(f"  Coins    : {n_coins}  |  Total setups: {total}")
    print(f"  Horizon  : {hz_str}")
    print(f"  Overall win rate: {acc_str}")
    print(f"  (นิยาม Win = actual return >= {ATR_WIN_MULTIPLIER}× ATR% ของแต่ละ setup)")

    print(f"\n  Accuracy by Grade:")
    for grade in ["A", "B+", "B", "C+", "C", "D+", "D", "F"]:
        d = aggregates["by_grade"].get(grade)
        if d:
            bar = "█" * int(d["win_rate"] / 5)
            print(f"    [{grade}] {d['win_rate']:>5.1f}%  {bar:<20}  (n={d['n']})")

    print(f"\n  Accuracy by Phase:")
    for phase, d in sorted(
        aggregates["by_phase"].items(), key=lambda x: -x[1]["win_rate"]
    ):
        print(f"    {phase:<35}  {d['win_rate']:>5.1f}%  (n={d['n']})")

    print(f"\n  Accuracy by RSI Zone:")
    for zone, d in sorted(
        aggregates["by_rsi_zone"].items(), key=lambda x: -x[1]["win_rate"]
    ):
        print(f"    {zone:<15}  {d['win_rate']:>5.1f}%  (n={d['n']})")


# ─────────────────────────────────────────────────────────────
#  Confidence lookup — เรียกใช้โดย analyzer.py
# ─────────────────────────────────────────────────────────────

def lookup_confidence(
    grade: str,
    phase: str,
    rsi_zone: str,
    vol_spike_band: str,
    ma_align: str,
    score: int,
) -> float | None:
    """
    Weighted average ของ historical accuracy จาก category ที่ตรงกับ setup นี้
    Returns accuracy % หรือ None ถ้าไม่มีไฟล์ backtest
    """
    if not os.path.exists(BACKTEST_FILE):
        return None
    try:
        with open(BACKTEST_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    agg = data.get("aggregates", {})
    components: list[tuple[float, float]] = []

    def _add(category: str, key: str, weight: float) -> None:
        d = agg.get(category, {}).get(key)
        if d:
            components.append((d["win_rate"], weight))

    _add("by_grade",       grade,                3.0)
    _add("by_phase",       phase,                2.5)
    _add("by_score_range", _score_bucket(score), 2.0)
    _add("by_rsi_zone",    rsi_zone,             1.5)
    _add("by_vol_spike",   vol_spike_band,       1.5)
    _add("by_ma_align",    ma_align,             1.0)

    if not components:
        return None

    total_w  = sum(w for _, w in components)
    weighted = sum(rate * w for rate, w in components)
    return round(weighted / total_w, 1)


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Crypto Accuracy Backtester — ทดสอบความแม่นยำการพยากรณ์จากข้อมูลจริง"
    )
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Symbols to backtest (default: 35 coins)")
    parser.add_argument("--horizon", type=int, default=None,
                        help="กำหนด horizon (วัน) เช่น 1=24hr, 7=สัปดาห์ (default: auto)")
    args = parser.parse_args()

    syms   = args.symbols or DEFAULT_SYMBOLS
    hz_str = f"{args.horizon} วัน" if args.horizon else "auto (ตาม setup)"
    print(f"\n╔{'═' * 62}╗")
    print(f"║  📊 CRYPTO ACCURACY BACKTEST — Momentum Forecast Test      ║")
    print(f"╚{'═' * 62}╝")
    print(f"\n  Coins    : {len(syms)} symbols")
    print(f"  Horizon  : {hz_str}")
    print(f"  Win def  : actual return >= {ATR_WIN_MULTIPLIER}× ATR% (per-coin, per-setup)")
    print(f"  Score    : composite score จาก analyzer.py (0-100)")
    print(f"  Data     : up to 500 daily bars (~1.5 ปี) per coin\n")

    run_backtest(symbols=syms, fixed_horizon=args.horizon)


if __name__ == "__main__":
    main()
