# Crypto Opportunity Scanner

> **Claude Code guidance** — อ่านไฟล์นี้ก่อนทำงานกับโปรเจกต์

## Overview

CRYPTO OPPORTUNITY SCANNER — Balanced Analysis Mode
- Style: วิเคราะห์ทั้ง bull และ bear scenarios เท่ากัน, data-driven targets
- Data sources: Binance (primary) + **Bitkub** (THB price, free) + CoinGecko (fallback)
- Analysis: 3-tier fallback (Claude → Groq → Technical Rules) + Reconciliation Gate
- Output: console summary + JSON ใน `output/`

---

## Commands

### 🚀 Opportunity Scanner

```bash
# สแกน Bitkub ทั้งหมด หาเหรียญ hot opportunity
python scanner.py

# Custom options
python scanner.py --top 20              # top 20 coins
python scanner.py --min-vol 5000000     # min 5M THB volume
python scanner.py --min-change 5        # min 5% 24h change
python scanner.py --no-deep             # quick scan (ไม่ fetch OHLCV, เร็วกว่า)
python scanner.py --analyze             # scan + deep analyze top 3

# หรือผ่าน data_fetcher โดยตรง
python data_fetcher.py --scan
python data_fetcher.py --scan --top 20 --min-change 5
```

### ดึงข้อมูล crypto

```bash
# ดึงข้อมูลเดี่ยว (รองรับทั้ง BTC หรือ BTCUSDT)
python data_fetcher.py --symbol BTCUSDT
python data_fetcher.py --symbol BTC
python data_fetcher.py --symbol ETH

# Market overview (top 15 coins + BTC regime)
python data_fetcher.py --overview
```

### วิเคราะห์ crypto

```bash
# วิเคราะห์ (ใช้ข้อมูลที่ fetch ไว้แล้ว)
python analyzer.py --symbol BTC

# Fetch + วิเคราะห์ในขั้นตอนเดียว
python analyzer.py --symbol ETH --fetch
python analyzer.py --symbol SOL --fetch
```

### Calibration (ครั้งแรก และ refresh ทุก 7 วัน)

```bash
python calibration.py
```

---

## ข้อมูลที่ได้จาก data_fetcher

| ข้อมูล | รายละเอียด | แหล่ง |
|--------|-----------|-------|
| Price | Current, 24h High/Low, % change | Binance |
| Volume 24h | USDT + base asset volume | Binance |
| RSI(14) daily | จากราคาปิดรายวัน 100 bars | คำนวณ |
| RSI(14) 4h | intraday momentum | คำนวณ |
| MA7/MA25/MA99 | 7/25/99-day moving averages | คำนวณ |
| ATR(14) | สำหรับ target/invalidation calculation | คำนวณ |
| Volume Spike | ratio เทียบกับ avg 14 วัน | คำนวณ |
| Price Structure | Phase + volume trend (จาก OHLCV) | คำนวณ |
| Neutral Score | market structure score [-1,+1] | คำนวณ |
| Market Cap | USD value | CoinGecko |
| 7d change | % เปลี่ยนแปลง 7 วัน | CoinGecko |
| Bitkub THB price | ราคา THB + volume | **Bitkub API** |
| Opportunity Score | 0-100 composite score | คำนวณ |
| Opportunity Grade | S/A/B/C/D | คำนวณ |

### Price Structure Phases
- `TIGHT_RANGE_HIGHER_LOWS` — Pre-breakout setup (สัญญาณ bullish)
- `CONSOLIDATING_HIGHER_LOWS` — Accumulation กำลังเกิด
- `CONSOLIDATING_FLAT` — Sideways ไม่มีทิศทาง
- `UPTREND_PULLBACK` — Healthy pullback ใน uptrend
- `VOLATILE_NO_STRUCTURE` — ระวัง ไม่มี structure

---

## Analysis Tiers

| Tier | Engine | ต้องการ |
|------|--------|---------|
| 1 | Claude (Anthropic) | ANTHROPIC_API_KEY + credit |
| 2 | Groq Llama3 (ฟรี) | GROQ_API_KEY — สมัครฟรีที่ groq.com |
| 3 | Technical Rules | ไม่ต้อง key ใดๆ |

ระบบบอก user ว่า "กำลังใช้ [Tier X] วิเคราะห์อยู่" เสมอ

---

## Opportunity Grading System

| Grade | Score | ความหมาย |
|-------|-------|---------|
| S | 82-100 | 🔥 FIRE — setup แน่น เข้าได้เลย |
| A | 65-81 | ⚡ Strong — น่าสนใจมาก |
| B | 48-64 | 📈 Watch — รอ volume confirm |
| C | 30-47 | 👀 Weak — setup ยังไม่ดีพอ |
| D | 0-29 | 😴 Pass — ข้ามไป |

Score คำนวณจาก: momentum 24h/7d (40pts) + volume spike (25pts) + phase structure (20pts) + RSI zone (15pts)

---

## Risk/Reward (Calibrated ATR)

Target และ Stop คำนวณจาก calibration_data.json (grid search จาก historical data):
- **Target** : Entry + k × ATR(14)  — k มาจาก calibration (default 2.5)
- **Invalidation** : Entry − j × ATR(14)  — j มาจาก calibration (default 1.5)
- R:R = k/j (ถ้า < 1.5 → signal = neutral, รอ setup ที่ดีกว่า)

รัน `python calibration.py` เพื่ออัพเดท k/j จาก historical hit rate จริง

---

## เมื่อ User ถามเรื่อง Crypto

**ขั้นตอนมาตรฐาน:**

1. ตรวจ `output/calibration_data.json` — ถ้าไม่มี รัน `python calibration.py` ก่อน (~2 นาที)
2. รัน `python scanner.py` เพื่อหา opportunity จาก Bitkub ทั้งหมด
3. ถ้า BTC regime = RISK_OFF → แจ้ง user ว่าตลาดอันตราย
4. รัน `python analyzer.py --symbol <SYMBOL> --fetch` เพื่อวิเคราะห์เชิงลึก
5. อ่าน `output/<SYMBOL>.json` และ `output/<SYMBOL>_analysis.json` แล้ว **แสดงผลใน chat โดยใช้ format ด้านล่างทุกครั้ง** (ใส่ใน code block)

> **สำคัญ:** User ใช้ผ่าน chat ไม่ได้เห็น terminal output — ต้อง print combined format ใน chat response เสมอ ห้ามแค่สรุปสั้นๆ
> ดึงข้อมูลจาก `result["analysis"]` (pre-formatted string) เป็นหลัก — แสดงตามที่ analyzer สร้างมาได้เลย

### Combined Chat Format (ใช้ทุกครั้งที่วิเคราะห์เหรียญ)

````
════════════════════════════════════════════════════════════
🚀 {SYMBOL}/USDT  [{data_source}]  |  🎯 [{grade}] {score}/100  {grade_bar}
════════════════════════════════════════════════════════════
💰 ราคา       : ${price}
   Vol 24h    : {volume_24h}  |  vs เฉลี่ย: ×{volume_spike} {spike_flag}
   Market Cap : {market_cap}
   7d          : {change_7d_pct}%

🇹🇭 Bitkub    : ฿{price_thb} THB  |  Vol: {volume_thb}
               (ถ้าไม่มีใน Bitkub ให้แสดง "ไม่มีในรายการ")

────────────────────────────────────────────────────────────
📊 Technical Indicators:
   RSI(14)    : {rsi_14} {rsi_flag}  |  4h RSI: {rsi_14_4h}
   Vol Spike  : ×{volume_spike} {spike_flag}
   MA25       : ${ma25} {ma25_check}  |  MA99: ${ma99} {ma99_check}
   ATR(14)    : ${atr_14}
   Phase      : {phase}
   Neutral Score: {neutral_score} {ns_label}

────────────────────────────────────────────────────────────
📋 คำแนะนำการเทรด:
   สัญญาณ      : {signal} {signal_emoji}
   Entry        : ${entry_price}
   Target       : ${target}  ({target_pct:+}%)  ↑/↓
   Invalidation : ${inval}   ({inval_pct}%)
   R/R Ratio    : 1:{rr}
   แนวรับ       : ${support}  —  แนวต้าน: ${resistance}
   จังหวะเข้า   : {entry_timing}

────────────────────────────────────────────────────────────
💡 ช่วงคาด 2σ (horizon: {horizon_label}):
   ช่วงที่คาด        : {lower_pct}% ถึง {upper_pct}%  ใน {horizon_days} วัน
   ความแม่นยำทิศทาง  : ~{accuracy_pct}%
🏆 คะแนนความคุ้มค่า: {composite}/100 — {label}

────────────────────────────────────────────────────────────
⏰ แนะนำเวลาซื้อขาย (ICT = เวลาไทย UTC+7):
   กรอบเวลา    : {horizon_label}
   เซสชั่นหลัก : {best_session}  ★
   เซสชั่นรอง  : {secondary_session}
   เงื่อนไขเข้า: {entry_condition}
   หลีกเลี่ยง  : {avoid_window}
   ระดับเร่งด่วน: {urgency_display}

────────────────────────────────────────────────────────────
🧠 AI Analysis ({tier_name}):
📈 Bull case ({bull_pct}%): {bull_thesis}
   • {bull_evidence_1}
📉 Bear case ({bear_pct}%): {bear_thesis}
   • {bear_evidence_1}
⚖️  Base case ({base_pct}%): {base_thesis}
🎯 Direction: {direction_display}  |  Confidence: {confidence_pct}%
   Target: ${ai_target}  |  Invalidation: ${ai_inval}
💬 {reasoning}

────────────────────────────────────────────────────────────
🔀 RECONCILIATION:
   AI Direction  : {ai_direction}  (confidence {confidence_pct}%)
   Neutral Score : {neutral_score}  {ns_label}
   Final Signal  : {final_signal_display}
════════════════════════════════════════════════════════════
⚠️ Crypto มีความผันผวนสูง ใช้ประกอบการตัดสินใจเท่านั้น
````

**กฎการแสดงผล:**
- ถ้าไม่มีข้อมูล field ใด → แสดง `N/A` ไม่ใช่ข้ามบรรทัด
- `grade_bar`: S=🔥🔥🔥 / A=⚡⚡ / B=📈 / C=👀 / D=😴
- `rsi_flag`: >78=⚠️ Overbought / 50-68=🔥 Momentum zone / <35=📉 Oversold
- `spike_flag`: ≥2=🔥 SPIKE! / ≥1.3=↑ elevated
- `signal_emoji`: BUY=🟢 / SELL=🔴 / HOLD=🟡 / VETO HOLD=🔴🔴
- `ma_check`: above=✅ / below=❌
- `ns_label`: >0.1=📈 bullish / <-0.1=📉 bearish / else=⚖️ neutral
- `final_signal_display`:
  - LONG/SHORT ปกติ = ✅ {direction}
  - HOLD (vetoed) = 🔴 HOLD — {reason}
  - Confidence ลดลง = ⚠️ {direction} (confidence ลดลง จาก structure ขัดแย้ง)
- ถ้าไม่มี `accuracy_pct` → แสดง "รัน python backtest.py ก่อน"
- ถ้า Tier 3 (ไม่มี AI) → ข้ามส่วน 🧠 AI Analysis และ 🔀 Reconciliation
- ถ้าไม่มี timing block → ข้ามส่วน ⏰ ทั้งบล็อก
- **สำคัญ**: แสดง `result["analysis"]` field ตรงๆ ก่อน แล้วเสริมข้อมูลจาก JSON fields อื่น

**ตัวอย่างคำถามที่รองรับ:**
- "หาเหรียญน่าเล่นวันนี้" → `python scanner.py --analyze`
- "BTC/ETH/SOL แนวโน้มเป็นไง" → fetch + analyze
- "ตลาดตอนนี้เป็นยังไง" → `python data_fetcher.py --overview`
- "เหรียญใน Bitkub ตัวไหนน่าสนใจ" → `python scanner.py`
- "SOL กับ AVAX อันไหนน่าสนใจกว่า" → fetch + analyze ทั้งสอง แล้ว compare grade

---

## Symbols ที่รองรับ

**Global**: BTC, ETH, BNB, SOL, XRP, ADA, DOGE, AVAX, MATIC, DOT, LINK, UNI, LTC, ATOM, NEAR, APT, SUI, INJ, ARB, OP, PEPE, SHIB, WIF, BONK

**Bitkub-listed**: KUB, SIX, JFIN, ERN, SAND, MANA, GALA, AXS, AAVE, GRT, BAND, FTM, CAKE, CRV

และ Binance-listed coins อื่นๆ ทุกตัว (ใช้ symbol + USDT เช่น WIFUSDT)

---

## Output Files

- `output/<SYMBOL>.json` — ข้อมูลดิบ + Bitkub THB price + opportunity score
- `output/<SYMBOL>_analysis.json` — ผลวิเคราะห์ + tier ที่ใช้ + forecast range + neutral_score
- `output/market_overview.json` — ภาพรวมตลาด
- `output/opportunity_scan.json` — ผลสแกน Bitkub ทั้งหมด
- `output/calibration_data.json` — ATR multipliers + sigma ต่อเหรียญ (refresh ทุก 7 วัน)
- `output/veto_log.jsonl` — log ทุก veto event และ confidence reduction

### โครงสร้าง analysis JSON ใหม่ (หลัง refactor)

```json
{
  "symbol": "BTC",
  "tier_used": "Tier 2 — Groq ...",
  "analysis": "<pre-formatted string ทั้งหมด — แสดงตรงๆ ได้เลย>",
  "forecast": {
    "expected_pct": 4.3,
    "upper_pct": 4.3,
    "lower_pct": -4.3,
    "horizon_days": 3,
    "accuracy_pct": 18.5,
    "mode": "new"
  },
  "neutral_score": 0.42,
  "data_snapshot": { ... }
}
```

---

## Data Sources

| Source | ใช้เพื่อ | ต้อง key? |
|--------|---------|---------|
| Binance API | Price, OHLCV, 4h RSI | ❌ ฟรี |
| Bitkub API | THB price, scanner (ทุก Bitkub coins) | ❌ ฟรี |
| CoinGecko | Market cap, 7d change, fallback | ❌ ฟรี |

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# แก้ไข .env ใส่ GROQ_API_KEY (ฟรี) หรือ ANTHROPIC_API_KEY

# สร้าง calibration data (ครั้งแรก — ต้องทำก่อนวิเคราะห์)
python calibration.py
```

---

## Limitations

- Binance อาจ block IP นอก region — CoinGecko เป็น fallback
- Bitkub API อาจ block IP นอกประเทศไทย
- CoinGecko free tier: 30 requests/min
- Opportunity score คำนวณจาก technical เท่านั้น ไม่รวม news/sentiment
- Tier 3 ไม่รู้เรื่อง news, listings, partnerships
- calibration_data.json ต้องมีก่อนถึงจะใช้ symmetric forecast และ ATR calibration ได้เต็มที่
