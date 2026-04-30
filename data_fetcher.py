"""
data_fetcher.py — Crypto Data Fetcher + Opportunity Scanner
Usage:
  python data_fetcher.py --symbol BTC
  python data_fetcher.py --overview
  python data_fetcher.py --scan              # scan all Bitkub coins
  python data_fetcher.py --scan --top 20

Data sources (ฟรี ไม่ต้อง key):
  Primary : Binance Public REST API
  Bitkub  : Bitkub Public REST API (THB prices + Thai-listed coins)
  Fallback: CoinGecko Free API
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

import requests

from config import (
    BINANCE_BASE, COINGECKO_BASE, BITKUB_BASE,
    OUTPUT_DIR, LOG_LEVEL,
    SCANNER_MIN_VOLUME_THB, SCANNER_MIN_CHANGE_PCT, SCANNER_TOP_N,
)

logging.basicConfig(level=getattr(logging, LOG_LEVEL, "INFO"),
                    format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

os.makedirs(OUTPUT_DIR, exist_ok=True)

_HEADERS = {"User-Agent": "CryptoOpportunityScanner/2.0 (Research, Free APIs)"}
_TIMEOUT = 15

_SYMBOL_TO_CG_ID = {
    "BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin",
    "SOL": "solana", "XRP": "ripple", "ADA": "cardano",
    "DOGE": "dogecoin", "AVAX": "avalanche-2", "MATIC": "matic-network",
    "DOT": "polkadot", "LINK": "chainlink", "UNI": "uniswap",
    "LTC": "litecoin", "ATOM": "cosmos", "NEAR": "near",
    "APT": "aptos", "SUI": "sui", "INJ": "injective-protocol",
    "ARB": "arbitrum", "OP": "optimism", "PEPE": "pepe",
    "SHIB": "shiba-inu", "FET": "fetch-ai", "RENDER": "render-token",
    "TRX": "tron", "XLM": "stellar", "ICP": "internet-computer",
    # Bitkub-listed coins
    "KUB": "bitkub-coin", "SIX": "six-network", "JFIN": "jfin-coin",
    "ERN": "ethernity-chain", "SAND": "the-sandbox", "MANA": "decentraland",
    "GALA": "gala", "AXS": "axie-infinity", "AAVE": "aave",
    "GRT": "the-graph", "BAND": "band-protocol", "FTM": "fantom",
    "CAKE": "pancakeswap-token", "CRV": "curve-dao-token",
    "WIF": "dogwifcoin", "BONK": "bonk",
}

_OVERVIEW_IDS = (
    "bitcoin,ethereum,binancecoin,solana,ripple,cardano,"
    "dogecoin,avalanche-2,matic-network,polkadot"
)


# ─────────────────────────────────────────────────────────────
#  HTTP helpers
# ─────────────────────────────────────────────────────────────
def _binance_get(path: str, params: dict | None = None) -> Any:
    try:
        r = requests.get(f"{BINANCE_BASE}{path}", params=params or {},
                         headers=_HEADERS, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.debug("Binance %s failed: %s", path, e)
        return None


def _cg_get(path: str, params: dict | None = None) -> Any:
    try:
        r = requests.get(f"{COINGECKO_BASE}{path}", params=params or {},
                         headers=_HEADERS, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.debug("CoinGecko %s failed: %s", path, e)
        return None


def _bitkub_get(path: str, params: dict | None = None) -> Any:
    try:
        r = requests.get(f"{BITKUB_BASE}{path}", params=params or {},
                         headers=_HEADERS, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        # Some Bitkub endpoints wrap in {"error": 0, "result": {...}}
        if isinstance(data, dict) and "result" in data and "error" in data:
            return data["result"] if data["error"] == 0 else None
        return data
    except Exception as e:
        log.debug("Bitkub %s failed: %s", path, e)
        return None


# ─────────────────────────────────────────────────────────────
#  Symbol normalization
# ─────────────────────────────────────────────────────────────
def _normalize_symbol(raw: str) -> tuple[str, str]:
    """Returns (binance_pair, base_symbol) e.g. 'BTC' → ('BTCUSDT', 'BTC')"""
    s = raw.upper().strip()
    if s.endswith("USDT"):
        base = s[:-4]
        return s, base
    return f"{s}USDT", s


def _get_cg_id(symbol: str) -> str:
    sym = symbol.upper()
    if sym in _SYMBOL_TO_CG_ID:
        return _SYMBOL_TO_CG_ID[sym]
    result = _cg_get("/search", {"query": sym})
    if result and result.get("coins"):
        for coin in result["coins"][:5]:
            if (coin.get("symbol") or "").upper() == sym:
                return coin["id"]
    return sym.lower()


# ─────────────────────────────────────────────────────────────
#  Technical calculations
# ─────────────────────────────────────────────────────────────
def _calc_rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [d for d in deltas[-period:] if d > 0]
    losses = [-d for d in deltas[-period:] if d < 0]
    avg_gain = sum(gains) / period if gains else 0
    avg_loss = sum(losses) / period if losses else 0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - 100 / (1 + rs), 2)


def _calc_ma(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    return round(sum(values[-period:]) / period, 6)


def _calc_volume_spike(current_vol: float, volumes: list[float]) -> float:
    """Ratio of current 24h volume vs avg of last 14 daily bars (excl. current)"""
    if not volumes or len(volumes) < 3:
        return 1.0
    baseline = volumes[max(0, len(volumes) - 15):-1]
    if not baseline:
        return 1.0
    avg = sum(baseline) / len(baseline)
    return round(current_vol / avg, 2) if avg > 0 else 1.0


def _analyze_ohlcv_structure(closes: list[float], volumes: list[float]) -> dict:
    n = len(closes)
    if n < 8:
        return {"phase": "INSUFFICIENT_DATA"}

    half = n // 2
    recent_closes = closes[half:]

    lows_8 = closes[-8:]
    higher_lows = all(lows_8[i] >= lows_8[i-1] * 0.97 for i in range(1, len(lows_8)))

    recent_range_pct = (
        (max(recent_closes) - min(recent_closes)) / min(recent_closes) * 100
        if min(recent_closes) > 0 else 100
    )

    recent_vol = sum(volumes[half:]) / len(volumes[half:]) if volumes[half:] else 0
    older_vol = sum(volumes[:half]) / len(volumes[:half]) if volumes[:half] else 0
    if older_vol > 0:
        vol_trend = (
            "increasing" if recent_vol > older_vol * 1.15
            else "decreasing" if recent_vol < older_vol * 0.85
            else "stable"
        )
    else:
        vol_trend = "unknown"

    if recent_range_pct < 10 and higher_lows:
        phase = "TIGHT_RANGE_HIGHER_LOWS"
    elif recent_range_pct < 15 and higher_lows:
        phase = "CONSOLIDATING_HIGHER_LOWS"
    elif recent_range_pct < 15:
        phase = "CONSOLIDATING_FLAT"
    elif higher_lows:
        phase = "UPTREND_PULLBACK"
    else:
        phase = "VOLATILE_NO_STRUCTURE"

    return {
        "phase": phase,
        "range_pct": round(recent_range_pct, 2),
        "higher_lows": higher_lows,
        "volume_trend": vol_trend,
    }


# ─────────────────────────────────────────────────────────────
#  Opportunity Scoring
# ─────────────────────────────────────────────────────────────
def _quick_opportunity_score(
    pct_change: float,
    vol_thb: float,
    price: float,
    high_24h: float,
    low_24h: float,
) -> int:
    """Quick score from Bitkub ticker data only (no OHLCV needed)"""
    score = 0

    # Momentum 24h (0-40)
    if pct_change >= 20:   score += 40
    elif pct_change >= 10: score += 30
    elif pct_change >= 5:  score += 20
    elif pct_change >= 2:  score += 10
    elif pct_change >= 0:  score += 3

    # Volume in THB (0-35)
    if vol_thb >= 100_000_000:  score += 35
    elif vol_thb >= 50_000_000: score += 28
    elif vol_thb >= 10_000_000: score += 20
    elif vol_thb >= 5_000_000:  score += 12
    elif vol_thb >= 1_000_000:  score += 5

    # Price near 24h high = momentum continuing (0-25)
    if high_24h > 0 and price > 0:
        pct_from_high = (high_24h - price) / high_24h * 100
        if pct_from_high <= 1:   score += 25
        elif pct_from_high <= 3: score += 18
        elif pct_from_high <= 6: score += 10
        elif pct_from_high <= 10: score += 5

    return min(score, 100)


def _calc_full_opportunity_score(c: dict) -> int:
    """Full opportunity score using OHLCV technicals (0-100)"""
    score = 0
    chg_24h  = c.get("change_24h_pct", 0) or 0
    chg_7d   = c.get("change_7d_pct", 0) or 0
    vol_spike = c.get("volume_spike", 1.0) or 1.0
    rsi       = c.get("rsi_14")
    phase     = c.get("phase", "")
    above_ma99 = c.get("above_ma99")
    above_ma25 = c.get("above_ma25")

    # 24h momentum (0-25)
    if chg_24h >= 20:   score += 25
    elif chg_24h >= 10: score += 20
    elif chg_24h >= 5:  score += 13
    elif chg_24h >= 2:  score += 7
    elif chg_24h >= 0:  score += 2

    # 7d momentum (0-15)
    if chg_7d >= 30:    score += 15
    elif chg_7d >= 15:  score += 11
    elif chg_7d >= 5:   score += 6
    elif chg_7d >= 0:   score += 2

    # Volume spike (0-25)
    if vol_spike >= 3.0:   score += 25
    elif vol_spike >= 2.0: score += 18
    elif vol_spike >= 1.5: score += 12
    elif vol_spike >= 1.2: score += 7
    else:                  score += 3

    # Price structure (0-20)
    phase_pts = {
        "TIGHT_RANGE_HIGHER_LOWS":    20,
        "CONSOLIDATING_HIGHER_LOWS":  16,
        "UPTREND_PULLBACK":           12,
        "CONSOLIDATING_FLAT":          6,
        "VOLATILE_NO_STRUCTURE":       0,
    }
    score += phase_pts.get(phase, 5)
    if above_ma99 is True: score += 3
    if above_ma25 is True: score += 2

    # RSI momentum zone (0-15) — sweet spot 50-68
    if rsi is not None:
        if 50 <= rsi <= 68:   score += 15
        elif 35 <= rsi < 50:  score += 8
        elif 68 < rsi <= 78:  score += 6
        elif rsi < 35:        score += 5
        # RSI > 78: 0 pts (overextended)

    return min(score, 100)


def _calc_opportunity_grade(score: int) -> str:
    if score >= 70: return "A"
    if score >= 65: return "B+"
    if score >= 60: return "B"
    if score >= 55: return "C+"
    if score >= 50: return "C"
    if score >= 45: return "D+"
    if score >= 35: return "D"
    return "F"


# ─────────────────────────────────────────────────────────────
#  Bitkub API
# ─────────────────────────────────────────────────────────────
def _fetch_bitkub_all_tickers() -> dict:
    """Fetch all Bitkub market tickers — free, no auth, one call"""
    data = _bitkub_get("/market/ticker")
    return data if isinstance(data, dict) else {}


def _get_bitkub_coin_data(symbol: str) -> dict:
    """Get THB price + stats for a coin from Bitkub"""
    key = f"THB_{symbol.upper()}"
    tickers = _fetch_bitkub_all_tickers()
    if key in tickers:
        t = tickers[key]
        try:
            return {
                "listed": True,
                "price_thb": float(t.get("last", 0)),
                "high_24h_thb": float(t.get("high24hr", 0)),
                "low_24h_thb": float(t.get("low24hr", 0)),
                "change_24h_pct": float(t.get("percentChange", 0)),
                "volume_thb": float(t.get("quoteVolume", 0)),
                "volume_base": float(t.get("baseVolume", 0)),
            }
        except (ValueError, TypeError):
            pass
    return {"listed": False}


# ─────────────────────────────────────────────────────────────
#  Opportunity Scanner
# ─────────────────────────────────────────────────────────────
def scan_opportunities(
    min_vol_thb: float = SCANNER_MIN_VOLUME_THB,
    min_change_pct: float = SCANNER_MIN_CHANGE_PCT,
    top_n: int = SCANNER_TOP_N,
    fetch_deep: bool = True,
) -> dict:
    """
    Scan Bitkub for opportunities.
    Pass 1: Quick filter from all Bitkub tickers (1 API call)
    Pass 2: Fetch full OHLCV from Binance for top candidates
    """
    log.info("🔍 Starting opportunity scan...")

    # BTC regime
    btc_raw = _binance_get("/ticker/24hr", {"symbol": "BTCUSDT"})
    btc_chg = float(btc_raw.get("priceChangePercent", 0)) if btc_raw else 0.0
    if btc_chg < -5:        btc_regime = "RISK_OFF"
    elif btc_chg < -2.5:    btc_regime = "CAUTION"
    elif abs(btc_chg) <= 3: btc_regime = "RANGING"
    else:                   btc_regime = "UPTREND"

    # Pass 1 — Bitkub tickers
    tickers = _fetch_bitkub_all_tickers()
    if not tickers:
        log.warning("Bitkub API unavailable — switching to CoinGecko fallback...")
        return _scan_fallback_coingecko(btc_regime, btc_chg, top_n)

    candidates = []
    for key, ticker in tickers.items():
        if not key.startswith("THB_"):
            continue
        symbol = key[4:]
        if not symbol or len(symbol) > 10:
            continue
        try:
            last     = float(ticker.get("last", 0))
            pct_chg  = float(ticker.get("percentChange", 0))
            vol_thb  = float(ticker.get("quoteVolume", 0))
            high_24h = float(ticker.get("high24hr", last))
            low_24h  = float(ticker.get("low24hr", last))
        except (ValueError, TypeError):
            continue

        if last <= 0 or vol_thb < min_vol_thb:
            continue
        if abs(pct_chg) < min_change_pct:
            continue

        quick = _quick_opportunity_score(pct_chg, vol_thb, last, high_24h, low_24h)
        candidates.append({
            "symbol":        symbol,
            "price_thb":     round(last, 8),
            "change_24h_pct": round(pct_chg, 2),
            "volume_thb":    round(vol_thb, 2),
            "high_24h_thb":  round(high_24h, 8),
            "low_24h_thb":   round(low_24h, 8),
            "quick_score":   quick,
            "listed_bitkub": True,
        })

    candidates.sort(key=lambda x: x["quick_score"], reverse=True)
    pool = candidates[:min(top_n * 2, 30)]

    # Pass 2 — Deep OHLCV fetch for top candidates
    if fetch_deep and pool:
        print(f"  📡 Fetching technicals for {len(pool)} candidates...")
        for c in pool:
            sym = c["symbol"]
            try:
                data = fetch_crypto(sym)
                if "error" not in data:
                    tech    = data.get("technicals", {})
                    struct  = tech.get("price_structure", {})
                    history = data.get("history_30d", [])
                    vols    = [bar["volume"] for bar in history]
                    base_vol = data.get("price", {}).get("volume_24h_base", 0) or 0

                    c["rsi_14"]       = tech.get("rsi_14")
                    c["rsi_14_4h"]    = tech.get("rsi_14_4h")
                    c["phase"]        = struct.get("phase", "N/A")
                    c["volume_trend"] = struct.get("volume_trend", "N/A")
                    c["above_ma25"]   = tech.get("above_ma25")
                    c["above_ma99"]   = tech.get("above_ma99")
                    c["atr_14"]       = tech.get("atr_14")
                    c["pct_from_high"] = tech.get("pct_from_high")
                    c["price_usd"]    = data.get("price", {}).get("current")
                    c["change_7d_pct"] = data.get("market_data", {}).get("change_7d_pct")
                    c["volume_spike"] = _calc_volume_spike(base_vol, vols)
                    c["opportunity_score"] = _calc_full_opportunity_score(c)
            except Exception as e:
                log.debug("Deep fetch failed %s: %s", sym, e)
                c["opportunity_score"] = c["quick_score"]
            c["grade"] = _calc_opportunity_grade(
                c.get("opportunity_score", c["quick_score"])
            )

    pool.sort(key=lambda x: x.get("opportunity_score", x["quick_score"]), reverse=True)

    return {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "btc_regime":         btc_regime,
        "btc_change_24h_pct": round(btc_chg, 2),
        "total_scanned":      len(candidates),
        "opportunities":      pool[:top_n],
    }


def _scan_fallback_coingecko(btc_regime: str, btc_chg: float, top_n: int) -> dict:
    """Fallback scanner using CoinGecko when Bitkub is unavailable"""
    gainers = _cg_get("/coins/markets", {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 50,
        "page": 1,
        "price_change_percentage": "24h,7d",
        "sparkline": "false",
    })
    opportunities = []
    if gainers:
        for c in sorted(gainers, key=lambda x: x.get("price_change_percentage_24h") or 0, reverse=True):
            pct_chg = c.get("price_change_percentage_24h") or 0
            if pct_chg < 2:
                continue
            # Convert USD volume to rough THB equivalent (×35) for scoring
            vol_thb_est = (c.get("total_volume") or 0) * 35
            score = _quick_opportunity_score(
                pct_chg, vol_thb_est,
                c.get("current_price", 0),
                c.get("high_24h", 0),
                c.get("low_24h", 0),
            )
            opportunities.append({
                "symbol":          (c.get("symbol") or "").upper(),
                "price_usd":       c.get("current_price"),
                "change_24h_pct":  round(pct_chg, 2),
                "change_7d_pct":   round(c.get("price_change_percentage_7d_in_currency") or 0, 2),
                "volume_usd":      c.get("total_volume"),
                "listed_bitkub":   False,
                "quick_score":     score,
                "opportunity_score": score,
                "grade":           _calc_opportunity_grade(score),
            })

    return {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "btc_regime":         btc_regime,
        "btc_change_24h_pct": round(btc_chg, 2),
        "total_scanned":      len(opportunities),
        "opportunities":      opportunities[:top_n],
        "source":             "coingecko_fallback",
    }


# ─────────────────────────────────────────────────────────────
#  Fetch single coin
# ─────────────────────────────────────────────────────────────
def fetch_crypto(symbol_or_pair: str) -> dict:
    pair, base = _normalize_symbol(symbol_or_pair)
    log.info("Fetching %s (Binance primary)...", pair)

    result: dict = {
        "symbol":      base,
        "pair":        pair,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "data_source": "unknown",
    }

    # Binance price + 24h stats
    price_data = _binance_get("/ticker/price", {"symbol": pair})
    binance_price = float(price_data["price"]) if price_data and "price" in price_data else None

    stats_24h = _binance_get("/ticker/24hr", {"symbol": pair})
    if stats_24h and "lastPrice" in stats_24h:
        result["data_source"] = "binance"
        price = float(stats_24h["lastPrice"])
        result["price"] = {
            "current":          round(price, 8),
            "open_24h":         round(float(stats_24h.get("openPrice", price)), 8),
            "high_24h":         round(float(stats_24h.get("highPrice", price)), 8),
            "low_24h":          round(float(stats_24h.get("lowPrice", price)), 8),
            "change_24h_pct":   round(float(stats_24h.get("priceChangePercent", 0)), 2),
            "volume_24h_usdt":  round(float(stats_24h.get("quoteVolume", 0)), 2),
            "volume_24h_base":  round(float(stats_24h.get("volume", 0)), 4),
            "trades_24h":       int(stats_24h.get("count", 0)),
        }
    elif binance_price:
        result["data_source"] = "binance_price_only"
        result["price"] = {"current": round(binance_price, 8)}
    else:
        log.info("Binance unavailable for %s — trying CoinGecko...", pair)

    # Daily OHLCV klines (100 bars)
    klines = _binance_get("/klines", {"symbol": pair, "interval": "1d", "limit": 100})
    technicals = {}

    if klines and len(klines) >= 14:
        closes  = [float(k[4]) for k in klines]
        volumes = [float(k[5]) for k in klines]
        highs   = [float(k[2]) for k in klines]
        lows    = [float(k[3]) for k in klines]

        rsi = _calc_rsi(closes)
        if rsi is not None:
            technicals["rsi_14"] = rsi

        technicals["ma7"]  = _calc_ma(closes, 7)
        technicals["ma25"] = _calc_ma(closes, 25)
        technicals["ma99"] = _calc_ma(closes, 99)

        current_price = result.get("price", {}).get("current", closes[-1])
        if technicals.get("ma25"):
            technicals["above_ma25"] = current_price > technicals["ma25"]
        if technicals.get("ma99"):
            technicals["above_ma99"] = current_price > technicals["ma99"]

        atrs = [highs[i] - lows[i] for i in range(-14, 0)]
        technicals["atr_14"] = round(sum(atrs) / 14, 8)

        technicals["high_period"] = round(max(highs), 8)
        technicals["low_period"]  = round(min(lows), 8)
        technicals["pct_from_high"] = round(
            (current_price - max(highs)) / max(highs) * 100, 2
        ) if max(highs) > 0 else 0

        technicals["price_structure"] = _analyze_ohlcv_structure(closes, volumes)

        # Volume spike vs 14-day avg
        base_vol = result.get("price", {}).get("volume_24h_base", 0) or 0
        if base_vol:
            technicals["volume_spike"] = _calc_volume_spike(base_vol, volumes)

        # 4h RSI for intraday momentum
        klines_4h = _binance_get("/klines", {"symbol": pair, "interval": "4h", "limit": 50})
        if klines_4h and len(klines_4h) >= 14:
            closes_4h = [float(k[4]) for k in klines_4h]
            rsi_4h = _calc_rsi(closes_4h)
            if rsi_4h is not None:
                technicals["rsi_14_4h"] = rsi_4h

        result["history_30d"] = [
            {
                "date":   datetime.fromtimestamp(int(k[0]) / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
                "open":   round(float(k[1]), 8),
                "high":   round(float(k[2]), 8),
                "low":    round(float(k[3]), 8),
                "close":  round(float(k[4]), 8),
                "volume": round(float(k[5]), 4),
            }
            for k in klines[-30:]
        ]

    result["technicals"] = technicals

    # CoinGecko: market cap + 7d change
    cg_id   = _get_cg_id(base)
    cg_data = _cg_get("/simple/price", {
        "ids": cg_id,
        "vs_currencies": "usd",
        "include_market_cap": "true",
        "include_24hr_vol":   "true",
        "include_24hr_change": "true",
        "include_7d_change":  "true",
    })
    if cg_data and cg_id in cg_data:
        cd = cg_data[cg_id]
        result["market_data"] = {
            "market_cap_usd":  cd.get("usd_market_cap"),
            "volume_24h_usd":  cd.get("usd_24h_vol"),
            "change_24h_pct":  cd.get("usd_24h_change"),
            "change_7d_pct":   cd.get("usd_7d_change"),
        }
        if "price" not in result:
            result["price"] = {
                "current":        cd.get("usd"),
                "change_24h_pct": cd.get("usd_24h_change"),
            }
            result["data_source"] = "coingecko"

    # yfinance last resort
    if "price" not in result:
        try:
            import yfinance as yf
            hist = yf.Ticker(f"{base}-USD").history(period="7d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
                prev  = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price
                result["price"] = {
                    "current":        round(price, 8),
                    "change_24h_pct": round((price - prev) / prev * 100, 2) if prev else 0,
                }
                result["data_source"] = "yfinance"
        except Exception as e:
            return {"symbol": base, "error": f"All sources failed: {e}"}

    # Bitkub THB price
    result["bitkub"] = _get_bitkub_coin_data(base)

    # Opportunity score
    opp_score = _calc_full_opportunity_score({
        "change_24h_pct": result.get("price", {}).get("change_24h_pct", 0),
        "change_7d_pct":  result.get("market_data", {}).get("change_7d_pct", 0),
        "volume_spike":   technicals.get("volume_spike", 1.0),
        "rsi_14":         technicals.get("rsi_14"),
        "phase":          technicals.get("price_structure", {}).get("phase", ""),
        "above_ma25":     technicals.get("above_ma25"),
        "above_ma99":     technicals.get("above_ma99"),
    })
    result["opportunity_score"] = opp_score
    result["opportunity_grade"] = _calc_opportunity_grade(opp_score)

    return result


# ─────────────────────────────────────────────────────────────
#  Market Overview
# ─────────────────────────────────────────────────────────────
def fetch_overview() -> dict:
    log.info("Fetching crypto market overview...")

    coins = _cg_get("/coins/markets", {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 20,
        "page": 1,
        "sparkline": "false",
        "price_change_percentage": "24h,7d",
    })

    global_data = _cg_get("/global")

    btc_24h = _binance_get("/ticker/24hr", {"symbol": "BTCUSDT"})
    btc_regime = "UNKNOWN"
    btc_chg = 0.0
    if btc_24h and "priceChangePercent" in btc_24h:
        btc_chg = float(btc_24h["priceChangePercent"])
        if btc_chg < -5:        btc_regime = "RISK_OFF"
        elif btc_chg < -2.5:    btc_regime = "CAUTION"
        elif abs(btc_chg) <= 3: btc_regime = "RANGING"
        else:                   btc_regime = "UPTREND"

    top_coins = []
    if coins:
        for c in coins[:15]:
            mc = c.get("market_cap") or 0
            top_coins.append({
                "rank":          c.get("market_cap_rank"),
                "symbol":        (c.get("symbol") or "").upper(),
                "name":          c.get("name"),
                "price_usd":     c.get("current_price"),
                "market_cap_usd": mc,
                "change_24h_pct": round(c.get("price_change_percentage_24h") or 0, 2),
                "change_7d_pct":  round(c.get("price_change_percentage_7d_in_currency") or 0, 2),
                "volume_24h_usd": c.get("total_volume"),
            })

    result = {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "btc_regime":         btc_regime,
        "btc_change_24h_pct": round(btc_chg, 2),
        "top_15_coins":       top_coins,
    }

    if global_data and "data" in global_data:
        gd = global_data["data"]
        result["global"] = {
            "btc_dominance_pct":        round((gd.get("market_cap_percentage") or {}).get("btc", 0), 1),
            "eth_dominance_pct":        round((gd.get("market_cap_percentage") or {}).get("eth", 0), 1),
            "total_market_cap_usd":     gd.get("total_market_cap", {}).get("usd"),
            "market_cap_change_24h_pct": round(gd.get("market_cap_change_percentage_24h_usd") or 0, 2),
            "active_coins":             gd.get("active_cryptocurrencies"),
        }

    return result


# ─────────────────────────────────────────────────────────────
#  Print summaries
# ─────────────────────────────────────────────────────────────
def print_crypto_summary(data: dict):
    if "error" in data:
        print(f"\n❌ {data.get('symbol', '?')}: {data['error']}\n")
        return

    sym   = data["symbol"]
    p     = data.get("price", {})
    tech  = data.get("technicals", {})
    mkt   = data.get("market_data", {})
    bk    = data.get("bitkub", {})
    price = p.get("current", 0)
    chg   = p.get("change_24h_pct", 0)
    arrow = "▲" if (chg or 0) >= 0 else "▼"

    grade = data.get("opportunity_grade", "?")
    score = data.get("opportunity_score", 0)
    grade_bar = {
        "A":  "🔥🔥🔥",
        "B+": "🔥🔥",
        "B":  "⚡⚡",
        "C+": "📈",
        "C":  "👀",
        "D+": "😐",
        "D":  "😴",
        "F":  "🚫",
    }.get(grade, "")

    print(f"\n{'='*60}")
    print(f"  {sym}/USDT  [{data['data_source']}]")
    print(f"  🎯 Opportunity: [{grade}] {score}/100  {grade_bar}")
    print(f"{'='*60}")
    print(f"  Price   : ${price:,.8f}")

    if p.get("high_24h"):
        print(f"  24h     : High ${p['high_24h']:,.8f}  Low ${p['low_24h']:,.8f}")
    if tech.get("high_period"):
        print(f"  Period  : High ${tech['high_period']:,.8f}  ({tech.get('pct_from_high', 0):+.1f}% from high)")

    if mkt.get("market_cap_usd"):
        mc = mkt["market_cap_usd"]
        mc_str = f"${mc/1e9:.2f}B" if mc >= 1e9 else f"${mc/1e6:.2f}M"
        print(f"  Mkt Cap : {mc_str}")
    if mkt.get("change_7d_pct") is not None:
        print(f"  7d      : {mkt['change_7d_pct']:+.2f}%")
    if p.get("volume_24h_usdt"):
        vol = p["volume_24h_usdt"]
        vol_str = f"${vol/1e9:.2f}B" if vol >= 1e9 else f"${vol/1e6:.2f}M"
        print(f"  Vol 24h : {vol_str}")

    # Bitkub THB price
    if bk.get("listed"):
        print(f"\n  🇹🇭 Bitkub : ฿{bk['price_thb']:,.4f} THB")
        vol_thb = bk.get("volume_thb", 0)
        if vol_thb:
            vol_thb_str = f"฿{vol_thb/1e6:.1f}M" if vol_thb >= 1e6 else f"฿{vol_thb:,.0f}"
            print(f"  Bitkub Vol: {vol_thb_str} THB")
    else:
        print(f"\n  🇹🇭 Bitkub : ไม่มีในรายการ")

    print(f"\n  --- Technical ---")
    if tech.get("rsi_14"):
        rsi = tech["rsi_14"]
        rsi_4h = tech.get("rsi_14_4h")
        flag = " ⚠️ Overbought" if rsi > 78 else " 🔥 Momentum zone!" if 50 <= rsi <= 68 else " 📉 Oversold" if rsi < 35 else ""
        rsi_str = f"{rsi}{flag}"
        if rsi_4h:
            rsi_str += f"  |  4h RSI: {rsi_4h}"
        print(f"  RSI(14) : {rsi_str}")
    if tech.get("volume_spike"):
        vs = tech["volume_spike"]
        vs_flag = "  🔥 SPIKE!" if vs >= 2 else "  ↑ elevated" if vs >= 1.3 else ""
        print(f"  Vol Spike: ×{vs:.2f}{vs_flag}")
    if tech.get("ma25"):
        print(f"  MA25    : ${tech['ma25']:,.8f}  {'✅' if tech.get('above_ma25') else '❌'}")
    if tech.get("ma99"):
        print(f"  MA99    : ${tech['ma99']:,.8f}  {'✅' if tech.get('above_ma99') else '❌'}")
    if tech.get("atr_14"):
        print(f"  ATR(14) : ${tech['atr_14']:,.8f}")

    struct = tech.get("price_structure", {})
    if struct.get("phase"):
        print(f"  Phase   : {struct['phase']}")
        print(f"  Vol Trend: {struct.get('volume_trend', 'N/A')}")

    print(f"\n  Saved to: {OUTPUT_DIR}/{sym}.json")
    print(f"{'='*60}\n")


def print_overview_summary(data: dict):
    regime_emoji = {"RISK_OFF": "🔴", "CAUTION": "🟡", "RANGING": "🟠", "UPTREND": "🟢"}.get(
        data.get("btc_regime"), "⚪"
    )
    print(f"\n{'='*60}")
    print(f"  Crypto Market Overview")
    print(f"{'='*60}")
    print(f"  BTC Regime: {regime_emoji} {data.get('btc_regime')} ({data.get('btc_change_24h_pct', 0):+.2f}% 24h)")

    g = data.get("global", {})
    if g:
        mc = g.get("total_market_cap_usd", 0)
        mc_str = f"${mc/1e12:.2f}T" if mc and mc >= 1e12 else f"${mc/1e9:.2f}B" if mc else "N/A"
        print(f"  Total Mkt Cap: {mc_str}  ({g.get('market_cap_change_24h_pct', 0):+.2f}% 24h)")
        print(f"  BTC Dom: {g.get('btc_dominance_pct', 0):.1f}%  ETH Dom: {g.get('eth_dominance_pct', 0):.1f}%")

    print(f"\n  Top 15 Coins:")
    print(f"  {'#':>3} {'Symbol':<8} {'Price':>14} {'24h':>8} {'7d':>8}")
    print(f"  {'-'*50}")
    for c in (data.get("top_15_coins") or [])[:15]:
        p = c.get("price_usd") or 0
        p_str = f"${p:,.4f}" if p < 1000 else f"${p:,.2f}"
        print(
            f"  {c.get('rank', '?'):>3} {c.get('symbol', '?'):<8} {p_str:>14} "
            f"{c.get('change_24h_pct', 0):>+7.2f}% {c.get('change_7d_pct', 0):>+7.2f}%"
        )
    print(f"{'='*60}\n")


def print_scan_results(scan: dict):
    regime_emoji = {"RISK_OFF": "🔴", "CAUTION": "🟡", "RANGING": "🟠", "UPTREND": "🟢"}.get(
        scan.get("btc_regime"), "⚪"
    )
    grade_icon = {"A": "🔥🔥🔥", "B+": "🔥🔥", "B": "⚡⚡", "C+": "📈", "C": "👀", "D+": "😐", "D": "😴", "F": "🚫"}

    print(f"\n{'╔' + '═'*64 + '╗'}")
    print(f"║  🚀 CRYPTO OPPORTUNITY SCANNER — Bitkub + Binance          ║")
    print(f"{'╚' + '═'*64 + '╝'}")

    ts = scan.get("timestamp", "")[:16].replace("T", " ")
    print(f"\n  BTC Regime  : {regime_emoji} {scan.get('btc_regime')} ({scan.get('btc_change_24h_pct', 0):+.2f}% 24h)")
    print(f"  Scanned     : {scan.get('total_scanned', 0)} coins → {len(scan.get('opportunities', []))} hot picks")
    print(f"  Time        : {ts} UTC")
    if scan.get("source") == "coingecko_fallback":
        print(f"  ⚠️  Source  : CoinGecko fallback (Bitkub API unavailable)")

    if scan.get("btc_regime") == "RISK_OFF":
        print(f"\n  ⛔ WARNING: BTC RISK_OFF — altcoin longs มีความเสี่ยงสูงมาก !")
    elif scan.get("btc_regime") == "CAUTION":
        print(f"\n  ⚠️  CAUTION: BTC อ่อนแรง — size เล็กๆ ก่อน")

    opps = scan.get("opportunities", [])
    if not opps:
        print("\n  ไม่พบ opportunity ที่น่าสนใจในตอนนี้\n")
        return

    print(f"\n  {'Grd':>5} {'Symbol':<7} {'THB Price':>12} {'7d%':>7} "
          f"{'VolSpike':>9} {'RSI':>5} {'Phase':<28} {'Score':>5}")
    print(f"  {'-'*85}")

    for c in opps:
        grade   = c.get("grade", "?")
        score   = c.get("opportunity_score", c.get("quick_score", 0))
        sym     = c.get("symbol", "?")
        p_thb   = c.get("price_thb", 0)
        chg_24h = c.get("change_24h_pct", 0)
        chg_7d  = c.get("change_7d_pct") or 0
        spike   = c.get("volume_spike")
        rsi     = c.get("rsi_14")
        phase   = (c.get("phase") or "N/A")[:27]
        icon    = grade_icon.get(grade, "")

        p_str     = f"฿{p_thb:,.4f}" if p_thb else "N/A"
        spike_str = (f"🔥×{spike:.1f}" if spike and spike >= 2
                     else f"↑×{spike:.1f}" if spike and spike >= 1.3
                     else (f"×{spike:.1f}" if spike else "N/A"))
        rsi_str   = f"{rsi:.1f}" if rsi else "N/A"
        c7_str    = f"{chg_7d:+.1f}%" if chg_7d else "  N/A"

        print(
            f"  [{grade}]{icon:<1} {sym:<7} {p_str:>12} "
            f"{c7_str:>7} "
            f"{spike_str:>9} {rsi_str:>5}  {phase:<28} {score:>5}"
        )

    print(f"\n  ─────────────────────────────────────────────────────────────")
    print(f"  💡 วิเคราะห์เชิงลึก : python analyzer.py --symbol <SYMBOL> --fetch")
    print(f"  🔄 Scan ใหม่        : python scanner.py\n")


# ─────────────────────────────────────────────────────────────
#  Save JSON
# ─────────────────────────────────────────────────────────────
def save_json(data: dict, name: str):
    path = os.path.join(OUTPUT_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    return path


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Crypto Data Fetcher + Opportunity Scanner")
    parser.add_argument("--symbol",     type=str,  help="Symbol e.g. BTC, BTCUSDT, ETH")
    parser.add_argument("--overview",   action="store_true", help="Market overview (top 15 coins)")
    parser.add_argument("--scan",       action="store_true", help="Scan Bitkub for opportunities")
    parser.add_argument("--top",        type=int,  default=SCANNER_TOP_N, help="Top N results for scan")
    parser.add_argument("--min-vol",    type=float, default=SCANNER_MIN_VOLUME_THB, help="Min THB volume for scan")
    parser.add_argument("--min-change", type=float, default=SCANNER_MIN_CHANGE_PCT, help="Min 24h %% change for scan")
    args = parser.parse_args()

    if not args.symbol and not args.overview and not args.scan:
        parser.print_help()
        sys.exit(1)

    if args.overview:
        data = fetch_overview()
        print_overview_summary(data)
        save_json(data, "market_overview")
        print(f"✅ Overview saved: {OUTPUT_DIR}/market_overview.json\n")

    if args.scan:
        scan = scan_opportunities(
            min_vol_thb=args.min_vol,
            min_change_pct=args.min_change,
            top_n=args.top,
        )
        print_scan_results(scan)
        save_json(scan, "opportunity_scan")
        print(f"✅ Scan saved: {OUTPUT_DIR}/opportunity_scan.json\n")

    if args.symbol:
        pair, base = _normalize_symbol(args.symbol)
        data = fetch_crypto(args.symbol)
        print_crypto_summary(data)
        save_json(data, base)


if __name__ == "__main__":
    main()
