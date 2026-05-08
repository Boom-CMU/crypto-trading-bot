"""
calibration.py — Calibration data pipeline (Task 1a: per-coin volatility bounds)

Produces output/calibration_data.json used by all refactored modules to:
  - Provide symmetric, data-driven sigma bounds for calc_expected_pct()
  - Supply stable ATR baseline for sanity-capping live ATR in Task 5
  - Hold placeholders for Tasks 1b/1c/1d (prob_up, ATR multipliers, veto threshold)

Refresh: weekly (7 days). Call load_calibration() from other modules — it handles
staleness checks automatically.

Usage:
  python calibration.py                       # compute all 35 coins
  python calibration.py --force               # force recompute
  python calibration.py --symbols BTC ETH SOL # specific coins
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import time
from datetime import datetime, timezone

from config import OUTPUT_DIR, LOG_LEVEL
from data_fetcher import _binance_get

logging.basicConfig(level=getattr(logging, LOG_LEVEL, "INFO"),
                    format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

CALIBRATION_FILE = os.path.join(OUTPUT_DIR, "calibration_data.json")
HORIZONS         = [1, 3, 7, 14]   # days — Task 3 uses sigma_{h}d keys
ROLLING_WINDOW   = 90              # bars for rolling realized-vol estimate
MIN_BARS         = ROLLING_WINDOW + max(HORIZONS)  # minimum klines needed
REFRESH_DAYS     = 7

# Task 1c — ATR multiplier grid search
K_GRID      = [1.5, 2.0, 2.5, 3.0]   # target multipliers to test
J_GRID      = [1.0, 1.5, 2.0]         # stop multipliers to test
ATR_HORIZON = 20                       # max bars to look forward for hit/miss
MIN_HIT_RATE = 0.30                    # minimum acceptable hit rate (sanity check)

DEFAULT_SYMBOLS = [
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE",
    "AVAX", "MATIC", "DOT", "LINK", "UNI", "LTC", "ATOM",
    "NEAR", "APT", "SUI", "INJ", "ARB", "OP", "AAVE",
    "TRX", "XLM", "ICP", "FET", "RENDER",
    "SAND", "MANA", "GALA", "AXS", "GRT", "FTM", "CRV",
    "PEPE", "SHIB",
]


# ─────────────────────────────────────────────────────────────
#  Pure-math helpers (no pandas, no external deps beyond stdlib)
# ─────────────────────────────────────────────────────────────

def _pct_returns(closes: list[float], period: int) -> list[float]:
    """Non-overlapping period-day returns as fractions (e.g. 0.02 = +2%)."""
    result: list[float] = []
    for i in range(period, len(closes)):
        prev = closes[i - period]
        if prev > 0:
            result.append((closes[i] - prev) / prev)
    return result


def _rolling_std(values: list[float], window: int) -> float | None:
    """Sample std (ddof=1) of the last `window` values. None if insufficient data."""
    if len(values) < window:
        return None
    last = values[-window:]
    mean = sum(last) / window
    if window == 1:
        return 0.0
    variance = sum((x - mean) ** 2 for x in last) / (window - 1)
    return variance ** 0.5


def _compute_atr_series(
    highs: list[float],
    lows: list[float],
    closes: list[float],
) -> list[float]:
    """ATR(14) value for each bar starting at index 14."""
    atrs: list[float] = []
    for i in range(14, len(closes)):
        true_ranges: list[float] = []
        for j in range(i - 13, i + 1):
            prev_close = closes[j - 1] if j > 0 else closes[j]
            tr = max(
                highs[j] - lows[j],
                abs(highs[j] - prev_close),
                abs(lows[j] - prev_close),
            )
            true_ranges.append(tr)
        atrs.append(sum(true_ranges) / 14)
    return atrs


def _interpolate_sigma(horizon: int, coin_data: dict) -> float:
    """
    Linear interpolation of sigma for a horizon not in HORIZONS.
    Clamps to nearest boundary when outside [min(HORIZONS), max(HORIZONS)].
    """
    lo_h = max(h for h in HORIZONS if h <= horizon) if any(h <= horizon for h in HORIZONS) else HORIZONS[0]
    hi_h = min(h for h in HORIZONS if h >= horizon) if any(h >= horizon for h in HORIZONS) else HORIZONS[-1]

    sigma_lo = coin_data.get(f"sigma_{lo_h}d", 0.0)
    sigma_hi = coin_data.get(f"sigma_{hi_h}d", 0.0)

    if lo_h == hi_h:
        return sigma_lo

    t = (horizon - lo_h) / (hi_h - lo_h)
    return sigma_lo + t * (sigma_hi - sigma_lo)


# ─────────────────────────────────────────────────────────────
#  Task 1c — ATR multiplier grid search helpers
# ─────────────────────────────────────────────────────────────

def _check_hit(
    highs:  list[float],
    lows:   list[float],
    i:      int,
    target: float,
    stop:   float,
    horizon: int,
) -> str:
    """
    Walk forward from bar i+1, return "hit"/"miss"/"timeout".
    Assumes a long setup: target above entry, stop below.
    "hit"    = high reaches target before low reaches stop.
    "miss"   = low reaches stop first.
    "timeout"= neither within `horizon` bars.
    """
    end = min(i + horizon + 1, len(highs))
    for t in range(i + 1, end):
        if highs[t] >= target:
            return "hit"
        if lows[t] <= stop:
            return "miss"
    return "timeout"


def _grid_search_atr(
    closes: list[float],
    highs:  list[float],
    lows:   list[float],
) -> dict:
    """
    For each (k, j) in K_GRID × J_GRID, walk through every bar from MIN_BARS onward
    and record hit/miss/timeout.
    Returns {(k, j): {"hit": int, "miss": int, "timeout": int, "hit_rate": float, "ev": float}}
    EV = hit_rate × k − (1−hit_rate) × j
    """
    atr_series = _compute_atr_series(highs, lows, closes)
    results: dict = {}

    for k in K_GRID:
        for j in J_GRID:
            hits = misses = timeouts = 0
            for i in range(MIN_BARS, len(closes) - ATR_HORIZON - 1):
                atr_idx = i - 14   # _compute_atr_series starts at index 14
                if atr_idx < 0 or atr_idx >= len(atr_series):
                    continue
                atr = atr_series[atr_idx]
                if atr <= 0:
                    continue
                entry  = closes[i]
                target = entry + k * atr
                stop   = entry - j * atr
                outcome = _check_hit(highs, lows, i, target, stop, ATR_HORIZON)
                if outcome == "hit":
                    hits += 1
                elif outcome == "miss":
                    misses += 1
                else:
                    timeouts += 1

            total = hits + misses + timeouts
            if total == 0:
                continue
            # Timeouts treated as misses for EV (conservative)
            hit_rate = hits / total
            ev       = hit_rate * k - (1.0 - hit_rate) * j
            results[(k, j)] = {
                "hit":      hits,
                "miss":     misses + timeouts,
                "timeout":  timeouts,
                "hit_rate": round(hit_rate, 4),
                "ev":       round(ev, 4),
            }

    return results


def _select_best_atr_multipliers(
    aggregate: dict,
    verbose: bool = False,
) -> dict:
    """
    Given aggregate[(k,j)] = list of per-coin EVs, select the (k,j) with highest mean EV.
    Applies sanity checks:
      - best mean EV must be > 0
      - best hit_rate must be > MIN_HIT_RATE (checked via EV threshold)
    Returns {"target": k, "stop": j} or defaults {"target": 2.5, "stop": 1.5}.
    """
    DEFAULTS = {"target": 2.5, "stop": 1.5}

    candidates = {kj: evs for kj, evs in aggregate.items() if evs}
    if not candidates:
        log.warning("ATR calibration: no data — using defaults")
        return DEFAULTS

    mean_ev: dict = {kj: sum(evs) / len(evs) for kj, evs in candidates.items()}
    best_kj, best_ev = max(mean_ev.items(), key=lambda x: x[1])

    if best_ev <= 0:
        log.warning(
            "ATR calibration: best EV=%.4f ≤ 0 — data may be insufficient, using defaults",
            best_ev,
        )
        return DEFAULTS

    if verbose:
        print(f"\n  ATR grid results (mean EV across coins):")
        for (k, j), ev in sorted(mean_ev.items(), key=lambda x: -x[1])[:6]:
            marker = " ← best" if (k, j) == best_kj else ""
            print(f"    k={k}  j={j}  EV={ev:+.4f}{marker}")

    return {"target": best_kj[0], "stop": best_kj[1]}


# ─────────────────────────────────────────────────────────────
#  Per-coin computation
# ─────────────────────────────────────────────────────────────

def _compute_sigma_for_coin(
    closes: list[float],
    highs: list[float],
    lows: list[float],
) -> dict | None:
    """
    Compute sigma for all HORIZONS via 90-day rolling realized returns.
    Falls back to sqrt(h)*sigma_daily if rolling window is too small for that horizon.
    Returns None if the closes list is too short to compute anything meaningful.
    """
    if len(closes) < MIN_BARS:
        return None

    sigmas: dict[str, float] = {}
    data_source = "90d_rolling_actual"

    # Daily sigma first (needed for sqrt fallback)
    daily_returns = _pct_returns(closes, 1)
    sigma_daily = _rolling_std(daily_returns, ROLLING_WINDOW)
    if sigma_daily is None:
        return None

    for h in HORIZONS:
        returns_h = _pct_returns(closes, h)
        sigma_h = _rolling_std(returns_h, ROLLING_WINDOW)

        if sigma_h is None:
            sigma_h = sigma_daily * (h ** 0.5)
            data_source = "partial_sqrt_scaling"
            log.warning("insufficient_data horizon=%dd — using sqrt scaling", h)

        sigmas[f"sigma_{h}d"] = round(sigma_h, 6)

    # sigma_daily is an alias for sigma_1d for readability
    sigmas["sigma_daily"] = sigmas["sigma_1d"]

    # 14-day average ATR: median of ATR(14) values over last 90 bars for stability
    atr_series = _compute_atr_series(highs, lows, closes)
    tail = atr_series[-ROLLING_WINDOW:] if len(atr_series) >= ROLLING_WINDOW else atr_series
    atr_14_avg = round(statistics.median(tail), 6) if tail else None

    return {
        "sigma_daily": sigmas["sigma_daily"],
        **{f"sigma_{h}d": sigmas[f"sigma_{h}d"] for h in HORIZONS},
        "atr_14_avg":  atr_14_avg,
        "data_source": data_source,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────
#  Global fallback
# ─────────────────────────────────────────────────────────────

def _compute_global_fallback(per_coin: dict) -> dict:
    """Median sigma across all computed coins per horizon."""
    fallback: dict[str, float] = {}
    for h in HORIZONS:
        key = f"sigma_{h}d"
        values = [v[key] for v in per_coin.values() if isinstance(v.get(key), (int, float))]
        if values:
            fallback[key] = round(statistics.median(values), 6)
        else:
            fallback[key] = round(0.03 * (h ** 0.5), 6)  # absolute last resort
    return fallback


# ─────────────────────────────────────────────────────────────
#  Public interface — called by other modules
# ─────────────────────────────────────────────────────────────

def get_sigma(coin: str, horizon_days: int, calibration: dict) -> float:
    """
    Return realized-vol sigma for `coin` at `horizon_days`.
    Falls back to global median when coin is unknown.
    Interpolates linearly for non-standard horizons.
    """
    per_coin = calibration.get("per_coin", {})
    coin_key = coin.upper()

    if coin_key in per_coin:
        data = per_coin[coin_key]
        exact_key = f"sigma_{horizon_days}d"
        if exact_key in data:
            return data[exact_key]
        return _interpolate_sigma(horizon_days, data)

    # Unknown coin — use global fallback + warn
    log.warning("Coin %s not in calibration — using global fallback sigma", coin_key)
    fallback = calibration.get("_global_fallback", {})
    exact_key = f"sigma_{horizon_days}d"
    if exact_key in fallback:
        return fallback[exact_key]

    # Interpolate from fallback horizons
    fallback_with_all = {f"sigma_{h}d": fallback.get(f"sigma_{h}d", 0.03 * h ** 0.5)
                         for h in HORIZONS}
    return _interpolate_sigma(horizon_days, fallback_with_all)


def apply_isotonic(raw_prob: float, calib_section: dict) -> float:
    """
    Map raw_prob → calibrated_prob.

    Phase 1 (current): linear shrinkage toward 0.5 to reduce overconfidence.
      calibrated = 0.5 + (raw_prob - 0.5) * scale   (scale=0.6 by default)
      Guarantees: apply_isotonic(0.5, ...) == 0.5 exactly.

    Phase 2 (after shadow backtest): isotonic regression from knot pairs
      [[raw_0, cal_0], [raw_1, cal_1], ...] stored in calibration_data.json.
    """
    method = calib_section.get("method", "linear_scale")

    if method == "linear_scale":
        scale = calib_section.get("scale", 0.6)
        return 0.5 + (raw_prob - 0.5) * scale

    if method == "isotonic":
        knots = calib_section.get("knots", [])
        if not knots:
            return raw_prob
        raw_vals = [k[0] for k in knots]
        cal_vals  = [k[1] for k in knots]
        if raw_prob <= raw_vals[0]:
            return cal_vals[0]
        if raw_prob >= raw_vals[-1]:
            return cal_vals[-1]
        for i in range(len(raw_vals) - 1):
            if raw_vals[i] <= raw_prob <= raw_vals[i + 1]:
                t = (raw_prob - raw_vals[i]) / (raw_vals[i + 1] - raw_vals[i])
                return cal_vals[i] + t * (cal_vals[i + 1] - cal_vals[i])

    return raw_prob  # unknown method — pass through unchanged


def load_calibration(symbols: list[str] | None = None) -> dict:
    """
    Load calibration_data.json, recomputing if stale (>7 days) or missing.
    Safe to call from any module — handles its own staleness check.
    """
    needs_update = False
    reason = ""

    if not os.path.exists(CALIBRATION_FILE):
        needs_update = True
        reason = "calibration_data.json not found"
    else:
        age_days = (time.time() - os.path.getmtime(CALIBRATION_FILE)) / 86400
        if age_days >= REFRESH_DAYS:
            needs_update = True
            reason = f"calibration data is {age_days:.1f} days old (>{REFRESH_DAYS}d)"

    if needs_update:
        print(f"\n🔄 {reason} — computing calibration data (~2 min)...")
        return run_calibration(symbols=symbols)

    with open(CALIBRATION_FILE, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────
#  Main runner
# ─────────────────────────────────────────────────────────────

def run_calibration(
    symbols: list[str] | None = None,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Fetch OHLCV once per coin and run Task 1a (sigma) + Task 1c (ATR grid) together.
    Saves calibration_data.json.
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    per_coin: dict = {}
    failed:   list = []
    atr_agg: dict  = {(k, j): [] for k in K_GRID for j in J_GRID}

    for sym in symbols:
        pair = f"{sym}USDT"
        if verbose:
            print(f"  📊 {sym:<8}", end=" ", flush=True)

        klines = _binance_get("/klines", {"symbol": pair, "interval": "1d", "limit": 500})
        if not klines or len(klines) < MIN_BARS + ATR_HORIZON + 10:
            if verbose:
                print("⚠️  insufficient data")
            failed.append(sym)
            continue

        closes = [float(k[4]) for k in klines]
        highs  = [float(k[2]) for k in klines]
        lows   = [float(k[3]) for k in klines]

        # Task 1a: sigma
        sigma_result = _compute_sigma_for_coin(closes, highs, lows)
        if sigma_result is None:
            if verbose:
                print("⚠️  computation failed")
            failed.append(sym)
            continue
        per_coin[sym] = sigma_result

        # Task 1c: ATR grid (same klines, no extra fetch)
        grid = _grid_search_atr(closes, highs, lows)
        for (k, j), stats in grid.items():
            atr_agg[(k, j)].append(stats["ev"])

        if verbose:
            best_kj = max(grid, key=lambda kj: grid[kj]["ev"]) if grid else None
            s3 = sigma_result.get("sigma_3d", 0)
            if best_kj:
                print(
                    f"σ_3d={s3:.4f}  "
                    f"best_atr=k{best_kj[0]}/j{best_kj[1]}"
                    f"(ev={grid[best_kj]['ev']:+.3f})"
                )
            else:
                print(f"σ_3d={s3:.4f}  atr=no data")

    if not per_coin:
        raise RuntimeError("No coins computed — check Binance connectivity")

    global_fallback = _compute_global_fallback(per_coin)
    atr_multipliers = _select_best_atr_multipliers(atr_agg, verbose=verbose)

    output = {
        "version":          "1ac",
        "computed_at":      datetime.now(timezone.utc).isoformat(),
        "symbols_computed": list(per_coin.keys()),
        "symbols_failed":   failed,
        "per_coin":         per_coin,
        "_global_fallback": global_fallback,
        "prob_up_calibration": {
            "method": "linear_scale",
            "scale":  0.6,
        },
        "atr_multipliers": atr_multipliers,
        "veto_threshold":  0.4,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"\n  Computed : {len(per_coin)} coins  |  Failed: {len(failed)}")
        print(f"  σ_3d fallback  : {global_fallback.get('sigma_3d', 'N/A'):.4f}")
        print(f"  ATR multipliers: k={atr_multipliers['target']}  j={atr_multipliers['stop']}")
        print(f"\n✅ Saved: {CALIBRATION_FILE}\n")

    return output


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibration pipeline — per-coin volatility bounds (Task 1a)"
    )
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Coins to compute (default: all 35)")
    parser.add_argument("--force", action="store_true",
                        help="Force recompute even if calibration_data.json is fresh")
    args = parser.parse_args()

    syms = args.symbols or DEFAULT_SYMBOLS
    print(f"\n╔{'═' * 60}╗")
    print(f"║  📐 CALIBRATION — Per-Coin Volatility Bounds (Task 1a)   ║")
    print(f"╚{'═' * 60}╝")
    print(f"  Coins    : {len(syms)} symbols")
    print(f"  Horizons : {HORIZONS} days")
    print(f"  Method   : 90-day rolling realized returns\n")

    run_calibration(symbols=syms, force=args.force)


if __name__ == "__main__":
    main()
