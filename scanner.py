"""
scanner.py — CRYPTO OPPORTUNITY SCANNER CLI
ค้นหาเหรียญ momentum + pre-breakout จาก Binance TH ทั้งหมด

Usage:
  python scanner.py                        # scan ทุก Binance TH coins
  python scanner.py --top 20               # top 20
  python scanner.py --min-vol 2000000      # min 2M USDT volume
  python scanner.py --min-change 5         # min 5% 24h change
  python scanner.py --analyze              # scan + deep analyze top 3
  python scanner.py --no-deep              # quick scan เฉพาะ ticker (เร็วกว่า)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from config import OUTPUT_DIR, SCANNER_MIN_VOLUME_USDT, SCANNER_MIN_CHANGE_PCT, SCANNER_TOP_N
from data_fetcher import scan_opportunities, print_scan_results, save_json


def main():
    parser = argparse.ArgumentParser(
        description="🚀 Crypto Opportunity Scanner — Bitkub + Binance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--top",        type=int,   default=SCANNER_TOP_N,
                        help=f"Top N results (default: {SCANNER_TOP_N})")
    parser.add_argument("--min-vol",    type=float, default=SCANNER_MIN_VOLUME_USDT,
                        help=f"Min volume filter (default: {SCANNER_MIN_VOLUME_USDT:,.0f})")
    parser.add_argument("--min-change", type=float, default=SCANNER_MIN_CHANGE_PCT,
                        help=f"Min 24h %% change (default: {SCANNER_MIN_CHANGE_PCT})")
    parser.add_argument("--no-deep",    action="store_true",
                        help="Quick scan only — skip OHLCV fetch (faster)")
    parser.add_argument("--analyze",    action="store_true",
                        help="Deep analyze top 3 opportunities after scan")
    args = parser.parse_args()

    print(f"\n  🔍 Scanning Binance TH... (min vol: ${args.min_vol:,.0f} USDT | min change: {args.min_change}%)")

    scan = scan_opportunities(
        min_vol_usdt=args.min_vol,
        min_change_pct=args.min_change,
        top_n=args.top,
        fetch_deep=not args.no_deep,
    )

    print_scan_results(scan)
    save_json(scan, "opportunity_scan")

    # Optional: deep analyze top 3
    if args.analyze and scan.get("opportunities"):
        from data_fetcher import fetch_crypto, save_json as sfn
        from analyzer import analyze

        top3 = scan["opportunities"][:3]
        print(f"\n{'='*62}")
        print(f"  🧠 DEEP ANALYSIS — Top {len(top3)} Opportunities")
        print(f"{'='*62}\n")

        for c in top3:
            sym = c["symbol"]
            print(f"\n{'─'*62}")
            print(f"  Analyzing {sym}...")
            try:
                data = fetch_crypto(sym)
                sfn(data, sym)
                result = analyze(sym, data)
                print(result["analysis"])
                print(f"\n  ✅ Saved: {OUTPUT_DIR}/{sym}_analysis.json")
            except Exception as e:
                print(f"  ❌ Failed to analyze {sym}: {e}")


if __name__ == "__main__":
    main()
