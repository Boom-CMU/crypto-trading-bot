"""
telegram_bot.py — ส่งผล Daily Crypto Scan ไปยัง Telegram
Usage:
  python telegram_bot.py            # ส่งผลล่าสุดจาก output/opportunity_scan.json
  python telegram_bot.py --test     # ทดสอบ connection
  python telegram_bot.py --scan     # scan แล้วส่งเลย (ไม่ต้องรัน scanner.py ก่อน)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta

import requests
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
SCAN_FILE = os.path.join("output", "opportunity_scan.json")

_TELEGRAM_API = "https://api.telegram.org"
_MAX_MSG_LEN  = 4096


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────
def _esc(text: str) -> str:
    """Escape HTML special chars for Telegram HTML parse mode."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _grade_emoji(grade: str) -> str:
    return {
        "A": "🔥",
        "B+": "📈", "B": "📈",
        "C+": "👀", "C": "👀",
        "D+": "😐", "D": "😴",
        "F": "😴",
    }.get(grade, "😴")


def _regime_emoji(regime: str) -> str:
    return {
        "UPTREND":  "✅",
        "RANGING":  "⚖️",
        "CAUTION":  "⚠️",
        "RISK_OFF": "🔴",
    }.get(regime, "❓")


def _rsi_flag(rsi) -> str:
    if rsi is None:
        return ""
    if rsi > 78:   return " ⚠️OB"
    if rsi >= 50:  return " 🔥"
    if rsi < 35:   return " 📉OS"
    return ""


def _spike_flag(spike) -> str:
    if spike is None:
        return ""
    if spike >= 2.0:  return " 🔥SPIKE"
    if spike >= 1.3:  return " ↑"
    return ""


def _phase_short(phase: str) -> str:
    return {
        "TIGHT_RANGE_HIGHER_LOWS":   "💎 TIGHT_RANGE",
        "CONSOLIDATING_HIGHER_LOWS": "🎯 CONSOL_HL",
        "UPTREND_PULLBACK":          "📈 UT_PULLBACK",
        "CONSOLIDATING_FLAT":        "😐 FLAT",
        "VOLATILE_NO_STRUCTURE":     "⚠️ VOLATILE",
    }.get(phase, phase or "N/A")


def _fmt_thb(price) -> str:
    if not price:
        return "N/A"
    p = float(price)
    if p >= 1_000_000: return f"฿{p/1_000_000:.2f}M"
    if p >= 1_000:     return f"฿{p:,.0f}"
    if p >= 1:         return f"฿{p:,.2f}"
    return f"฿{p:.6f}"


def _fmt_vol_thb(vol) -> str:
    if not vol:
        return "N/A"
    v = float(vol)
    if v >= 1_000_000_000: return f"฿{v/1e9:.1f}B"
    if v >= 1_000_000:     return f"฿{v/1e6:.1f}M"
    if v >= 1_000:         return f"฿{v/1e3:.0f}K"
    return f"฿{v:.0f}"


# ─────────────────────────────────────────────────────────────
#  Message builder
# ─────────────────────────────────────────────────────────────
def build_message(scan: dict, top_n: int = 8) -> str:
    ts_raw  = scan.get("timestamp", "")
    regime  = scan.get("btc_regime", "N/A")
    btc_chg = scan.get("btc_change_24h_pct", 0)
    total   = scan.get("total_scanned", 0)
    opps    = scan.get("opportunities", [])

    # Parse timestamp → Bangkok time (UTC+7)
    try:
        dt_utc = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        dt_bkk = dt_utc + timedelta(hours=7)
        date_str = dt_bkk.strftime("%d %b %Y %H:%M")
    except Exception:
        date_str = ts_raw[:16] if ts_raw else "N/A"

    btc_chg_str = f"{btc_chg:+.2f}%"
    regime_icon = _regime_emoji(regime)

    lines = [
        f"🔍 <b>Daily Crypto Scan</b> — {_esc(date_str)} (BKK)",
        f"🌐 BTC Regime: <b>{_esc(regime)}</b> {regime_icon}  ({_esc(btc_chg_str)})",
        f"📊 Scanned: {total} Bitkub coins",
        "",
    ]

    if regime == "RISK_OFF":
        lines += [
            "🚨 <b>RISK OFF — ตลาดอันตราย!</b>",
            "   BTC ดิ่งหนัก — พิจารณาลดสถานะหรืองดเทรดก่อน",
            "",
        ]

    # Filter only grade S/A/B and top_n
    shown = [o for o in opps if o.get("grade", "D") in ("S", "A", "B")][:top_n]
    if not shown:
        shown = opps[:top_n]

    if not shown:
        lines.append("😴 ไม่พบ opportunity ที่น่าสนใจวันนี้")
    else:
        lines.append("🏆 <b>Top Opportunities:</b>")
        lines.append("")

        for coin in shown:
            sym    = coin.get("symbol", "?")
            grade  = coin.get("grade", "?")
            score  = coin.get("opportunity_score") or coin.get("quick_score", 0)
            price  = coin.get("price_thb")
            chg24  = coin.get("change_24h_pct", 0)
            vol    = coin.get("volume_thb")
            phase  = coin.get("phase", "")
            rsi    = coin.get("rsi_14")
            spike  = coin.get("volume_spike")
            chg7d  = coin.get("change_7d_pct")

            chg24_str = f"{chg24:+.1f}%" if chg24 else "N/A"
            chg7d_str = f"7d {chg7d:+.1f}%" if chg7d is not None else ""
            rsi_str   = f"RSI {rsi:.0f}{_rsi_flag(rsi)}" if rsi is not None else ""
            spike_str = f"Vol ×{spike:.1f}{_spike_flag(spike)}" if spike else ""
            phase_str = _phase_short(phase) if phase else ""

            # Row 1: grade + symbol + score
            g_emoji = _grade_emoji(grade)
            lines.append(
                f"{g_emoji} <b>[{_esc(grade)}] {_esc(sym)}</b>"
                f"  <code>{int(score)}/100</code>"
            )
            # Row 2: price + change
            row2 = f"   {_esc(_fmt_thb(price))}  {_esc(chg24_str)}"
            if chg7d_str:
                row2 += f"  |  {_esc(chg7d_str)}"
            lines.append(row2)
            # Row 3: vol + spike + RSI
            tech_parts = [p for p in [spike_str, rsi_str] if p]
            if tech_parts:
                lines.append(f"   {_esc('  |  '.join(tech_parts))}")
            # Row 4: vol THB + phase
            row4_parts = []
            if vol:
                row4_parts.append(f"Vol {_fmt_vol_thb(vol)}")
            if phase_str:
                row4_parts.append(phase_str)
            if row4_parts:
                lines.append(f"   {_esc('  |  '.join(row4_parts))}")
            lines.append("")

    lines += [
        "─" * 30,
        "⚠️ ใช้ประกอบการตัดสินใจเท่านั้น — ไม่ใช่คำแนะนำทางการเงิน",
    ]

    msg = "\n".join(lines)

    # Truncate if over Telegram limit
    if len(msg) > _MAX_MSG_LEN:
        msg = msg[:_MAX_MSG_LEN - 50] + "\n\n... (ข้อความยาวเกิน ตัดออก)"

    return msg


# ─────────────────────────────────────────────────────────────
#  Telegram sender
# ─────────────────────────────────────────────────────────────
def send_message(text: str, token: str = BOT_TOKEN, chat_id: str = CHAT_ID) -> bool:
    if not token or not chat_id:
        print("❌ กรุณาตั้งค่า TELEGRAM_BOT_TOKEN และ TELEGRAM_CHAT_ID ใน .env")
        return False

    url = f"{_TELEGRAM_API}/bot{token}/sendMessage"
    payload = {
        "chat_id":    chat_id,
        "text":       text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get("ok"):
            print("✅ ส่ง Telegram สำเร็จ")
            return True
        print(f"❌ Telegram API error: {data.get('description', 'unknown')}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ ส่ง Telegram ไม่สำเร็จ: {e}")
        return False


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Telegram Daily Crypto Report")
    parser.add_argument("--test",  action="store_true", help="ส่ง test message")
    parser.add_argument("--scan",  action="store_true", help="รัน scanner แล้วส่งเลย")
    parser.add_argument("--top",   type=int, default=8,  help="แสดงกี่เหรียญ (default: 8)")
    args = parser.parse_args()

    if not BOT_TOKEN or not CHAT_ID:
        print("❌ ไม่พบ TELEGRAM_BOT_TOKEN หรือ TELEGRAM_CHAT_ID ใน .env")
        print("   ดูวิธีตั้งค่าใน .env.example")
        sys.exit(1)

    if args.test:
        now_bkk = datetime.now(timezone.utc) + timedelta(hours=7)
        msg = (
            "✅ <b>Telegram Bot Test</b>\n"
            f"เวลา: {now_bkk.strftime('%d %b %Y %H:%M')} (BKK)\n"
            "การเชื่อมต่อสำเร็จ! Bot พร้อมส่ง Daily Scan แล้ว 🚀"
        )
        success = send_message(msg)
        sys.exit(0 if success else 1)

    if args.scan:
        print("📡 กำลัง scan...")
        from data_fetcher import scan_opportunities, save_json
        from config import SCANNER_MIN_VOLUME_THB, SCANNER_MIN_CHANGE_PCT, SCANNER_TOP_N
        scan = scan_opportunities(
            min_vol_thb=SCANNER_MIN_VOLUME_THB,
            min_change_pct=SCANNER_MIN_CHANGE_PCT,
            top_n=SCANNER_TOP_N,
            fetch_deep=True,
        )
        save_json(scan, "opportunity_scan")
    else:
        if not os.path.exists(SCAN_FILE):
            print(f"❌ ไม่พบ {SCAN_FILE}")
            print("   รัน: python scanner.py ก่อน หรือใช้ --scan flag")
            sys.exit(1)
        with open(SCAN_FILE, encoding="utf-8") as f:
            scan = json.load(f)

    msg = build_message(scan, top_n=args.top)
    print("\n─── Preview ───")
    print(msg)
    print("───────────────\n")

    success = send_message(msg)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
