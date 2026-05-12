"""
config.py — Load API keys and settings from .env
"""
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-6"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
OUTPUT_DIR = "output"

BINANCE_BASE = "https://api.binance.com/api/v3"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
BITKUB_BASE = "https://api.bitkub.com/api"          # free, no auth needed

SUPPORTED_SYMBOLS = [
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE",
    "AVAX", "MATIC", "POL", "DOT", "LINK", "UNI", "LTC", "ATOM",
    "NEAR", "APT", "SUI", "INJ", "ARB", "OP", "PEPE", "SHIB",
    # Bitkub-listed + popular altcoins
    "KUB", "SIX", "JFIN", "SAND", "MANA", "GALA", "AXS",
    "AAVE", "GRT", "BAND", "FTM", "CAKE", "CRV", "TRX", "XLM",
    "ICP", "FET", "RENDER", "WIF", "BONK", "ERN",
]

# Opportunity scanner defaults
SCANNER_MIN_VOLUME_USDT = 1_000_000   # minimum volume filter (THB units, Bitkub)
SCANNER_MIN_CHANGE_PCT = 2.0         # minimum 2% 24h move
SCANNER_TOP_N = 15                   # display top N results


# Feature flag: set to "1" to use legacy (biased) forecaster for A/B comparison
USE_LEGACY_FORECASTER = os.getenv("USE_LEGACY_FORECASTER", "0") == "1"


# ─── Tunable thresholds (used by reconciliation.py and analyzer.py) ──────
# Reconciliation gate
RECONCILE_VETO_CONFIDENCE         = 0.6   # AI must exceed this to be veto-eligible
RECONCILE_DISAGREEMENT_NEUTRAL    = 0.1   # |neutral_score| > this counts as taking a side
RECONCILE_CONFIDENCE_REDUCTION    = 0.6   # multiplier when mild disagreement detected

# Trading targets
MIN_RR_RATIO = 1.5    # min reward/risk for an actionable signal
SL_MAX_PCT   = 0.15   # absolute cap on stop-loss distance (15%)

# Backtest freshness
BACKTEST_REFRESH_DAYS = 7


def get_analysis_tier() -> str:
    if ANTHROPIC_API_KEY:
        return "tier1_claude"
    if GROQ_API_KEY:
        return "tier2_groq"
    return "tier3_technical"
