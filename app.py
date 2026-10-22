"""
Crypto Spot Desk — best-effort buy / sell decisions
===================================================
Spot only. Focus: high-quality BUY LOW / SELL HIGH calls.

Missing optional feeds (funding, OI) are ignored — not treated as failures.
Core decision uses: price action, RSI, Bollinger, HTF trend, volume,
Fear&Greed, global mcap, FRED/Yahoo yields when available.

Not financial advice. Not guaranteed profit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timezone
import requests
import time
import xml.etree.ElementTree as ET
import os

st.set_page_config(page_title="Spot Desk", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.big { font-size: 1.9rem; font-weight: 800; margin: 0.15rem 0; }
.buy { color: #00c853; } .sell { color: #ffab00; } .wait { color: #90a4ae; }
.card {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 0.6rem;
}
.alert-buy {
  background: rgba(0,200,83,0.18); border-left: 5px solid #00c853;
  padding: 12px 16px; border-radius: 8px; font-weight: 700; margin: 8px 0;
}
.alert-sell {
  background: rgba(255,171,0,0.18); border-left: 5px solid #ffab00;
  padding: 12px 16px; border-radius: 8px; font-weight: 700; margin: 8px 0;
}
.ok { color: #00c853; } .no { color: #ff5252; }
.small { color: #9e9e9e; font-size: 0.88rem; }
.live-dot { color: #ff1744; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

CRYPTO = {
    "Bitcoin": "BTCUSDT", "Ethereum": "ETHUSDT", "Solana": "SOLUSDT",
    "BNB": "BNBUSDT", "XRP": "XRPUSDT", "Cardano": "ADAUSDT",
    "Dogecoin": "DOGEUSDT", "Avalanche": "AVAXUSDT", "Chainlink": "LINKUSDT",
    "ONDO": "ONDOUSDT", "Stellar": "XLMUSDT", "Hedera": "HBARUSDT",
    "Algorand": "ALGOUSDT", "Flare": "FLRUSDT", "Sui": "SUIUSDT",
    "Near": "NEARUSDT", "Render": "RENDERUSDT", "Pepe": "PEPEUSDT",
    "Toncoin": "TONUSDT", "Tron": "TRXUSDT", "Litecoin": "LTCUSDT",
    "Polkadot": "DOTUSDT", "Aave": "AAVEUSDT",
}
INTERVALS = ["15m", "30m", "1h", "4h", "1d"]
HTF_MAP = {"15m": "1h", "30m": "4h", "1h": "4h", "4h": "1d", "1d": "1d"}
BINANCE = ["https://data-api.binance.vision", "https://api.binance.us"]


def get_fred_key():
    try:
        k = st.secrets.get("FRED_API_KEY", None)
        if k:
            return str(k).strip()
    except Exception:
        pass
    return os.environ.get("FRED_API_KEY", "").strip() or None


def fmt(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    x = float(x)
    if abs(x) < 0.01:
        return f"${x:.6f}"
    if abs(x) < 1:
        return f"${x:.4f}"
    if abs(x) < 100:
        return f"${x:.3f}"
    return f"${x:,.2f}"


@st.cache_data(ttl=12, show_spinner=False)
def binance_ticker_price(symbol):
    for host in BINANCE:
        try:
            r = requests.get(f"{host}/api/v3/ticker/price", params={"symbol": symbol}, timeout=8)
            if r.status_code == 200:
                return float(r.json()["price"])
        except Exception:
            continue
    return None


@st.cache_data(ttl=20, show_spinner=False)
def binance_klines(symbol, interval, limit=300):
    params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
    for host in BINANCE:
        try:
            r = requests.get(f"{host}/api/v3/klines", params=params, timeout=10)
            if r.status_code != 200:
                continue
            raw = r.json()
            if not isinstance(raw, list) or len(raw) < 40:
                continue
            df = pd.DataFrame(raw, columns=[
                "t", "Open", "High", "Low", "Close", "Volume",
                "T", "q", "n", "V", "Q", "x"
            ])
            df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_localize(None)
            df = df.set_index("t")
            for c in ["Open", "High", "Low", "Close", "Volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        except Exception:
            continue
    return pd.DataFrame()


@st.cache_data(ttl=90, show_spinner=False)
def fear_greed():
    try:
        d = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8).json()["data"][0]
        return int(d["value"]), d["value_classification"]
    except Exception:
        return None, None


@st.cache_data(ttl=120, show_spinner=False)
def coingecko_global():
    try:
        g = requests.get("https://api.coingecko.com/api/v3/global", timeout=10).json()["data"]
        return {
            "mcap_chg_24h": g.get("market_cap_change_percentage_24h_usd"),
            "btc_dom": g.get("market_cap_percentage", {}).get("btc"),
        }
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def fred_series(series_id, api_key):
    if not api_key:
        return None
    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 5,
            },
            timeout=12,
        )
        if r.status_code != 200:
            return None
        for o in r.json().get("observations", []):
            if o.get("value") not in (None, "."):
                return float(o["value"])
    except Exception:
        return None
    return None


@st.cache_data(ttl=600, show_spinner=False)
def yahoo_last(symbol):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=5d&interval=1d"
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            return float(r.json()["chart"]["result"][0]["meta"]["regularMarketPrice"])
    except Exception:
        pass
    return None


@st.cache_data(ttl=600, show_spinner=False)
def news_headlines(query="cryptocurrency", limit=6):
    items = []
    feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        f"https://news.google.com/rss/search?q={query.replace(' ', '+')}+crypto&hl=en-US&gl=US&ceid=US:en",
    ]
    for url in feeds:
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": "SpotDesk/1.0"})
            if r.status_code != 200:
                continue
            root = ET.fromstring(r.content)
            for item in root.iter("item"):
                title = (item.findtext("title") or "").strip()
                link = item.findtext("link") or ""
                if title:
                    items.append({"title": title, "link": link})
                if len(items) >= limit:
                    return items
        except Exception:
            continue
    return items[:limit]


def send_webhook(url, content):
    if not url or not url.startswith("http"):
        return False
    try:
        r = requests.post(url, json={"content": content}, timeout=8)
        return r.status_code < 300
    except Exception:
        return False


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, min_periods=n).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, min_periods=n).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

def atr_s(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def trend_bias(df):
    if df.empty or len(df) < 30:
        return "RANGE"
    c = df["Close"]
    e12, e26 = ema(c, 12), ema(c, 26)
    e50 = ema(c, min(50, max(20, len(df) // 2)))
    last = c.iloc[-1]
    if e12.iloc[-1] > e26.iloc[-1] > e50.iloc[-1] and last > e26.iloc[-1]:
        return "UP"
    if e12.iloc[-1] < e26.iloc[-1] < e50.iloc[-1] and last < e26.iloc[-1]:
        return "DOWN"
    return "RANGE"


def technicals(df):
    d = df.copy()
    c = d["Close"]
    d["EMA12"] = ema(c, 12)
    d["EMA26"] = ema(c, 26)
    d["EMA50"] = ema(c, 50) if len(d) >= 50 else ema(c, max(20, len(d) // 3))
    d["RSI"] = rsi(c)
    d["ATR"] = atr_s(d)
    mid, std = c.rolling(20).mean(), c.rolling(20).std()
    d["BB_u"], d["BB_l"] = mid + 2 * std, mid - 2 * std
    d["BB_pct"] = (c - d["BB_l"]) / (d["BB_u"] - d["BB_l"]).replace(0, np.nan)
    d["Swing_low"] = d["Low"].rolling(20).min()
    d["Swing_high"] = d["High"].rolling(20).max()
    d["Vol_MA"] = d["Volume"].rolling(20).mean()
    d["Vol_ratio"] = d["Volume"] / d["Vol_MA"].replace(0, np.nan)
    # momentum turning up from low (helps avoid pure falling knives)
    d["RSI_rising"] = d["RSI"] > d["RSI"].shift(2)
    d["MACD"] = d["EMA12"] - d["EMA26"]
    d["MACD_sig"] = ema(d["MACD"], 9)
    d["MACD_up"] = d["MACD"] > d["MACD_sig"]
    return d.dropna()


def load_macro(fred_key):
    dgs10 = fred_series("DGS10", fred_key) if fred_key else None
    fed = fred_series("FEDFUNDS", fred_key) if fred_key else None
    unrate = fred_series("UNRATE", fred_key) if fred_key else None
    if dgs10 is None:
        dgs10 = yahoo_last("^TNX")
    if fed is None:
        fed = yahoo_last("^IRX")
    return {"DGS10": dgs10, "FEDFUNDS": fed, "UNRATE": unrate}


def decide(last, htf, fng_val, fng_label, global_m, macro):
    """
    Stricter spot decision engine.
    - Required: value (BB or RSI) + HTF not DOWN
    - Soft blocks: extreme greed + not cheap, hard market crash
    - Weighted score; BUY only on strong score
    - Missing optional data simply omitted
    """
    price = float(last["Close"])
    rsi_v = float(last["RSI"])
    bb = float(last["BB_pct"]) if pd.notna(last["BB_pct"]) else 0.5
    atr = float(last["ATR"]) if pd.notna(last["ATR"]) else price * 0.01
    local_up = float(last["EMA12"]) >= float(last["EMA26"])
    above_50 = float(last["Close"]) >= float(last["EMA50"]) * 0.98
    vol_ratio = float(last.get("Vol_ratio") or 1)
    rsi_rising = bool(last.get("RSI_rising", False))
    macd_up = bool(last.get("MACD_up", False))

    stop = min(price - 1.5 * atr, float(last["Swing_low"]) * 0.998)
    risk = max(price - stop, atr * 0.4)
    exit_px = price + 2.0 * risk

    # Weighted factors: (name, weight, pass?, detail)
    factors = []

    # --- CORE (high weight) ---
    at_value = bb <= 0.35 or rsi_v <= 38
    factors.append(("At value (BB low or RSI low)", 3.0, at_value, f"BB {bb*100:.0f}% · RSI {rsi_v:.0f}"))

    htf_ok = htf in ("UP", "RANGE")
    factors.append(("HTF allows buys (not DOWN)", 3.0, htf_ok, f"HTF={htf}"))

    # avoid catching knives: want some turn OR range with deep oversold
    turn_ok = rsi_rising or macd_up or (rsi_v < 28 and htf == "RANGE")
    factors.append(("Momentum not still collapsing", 2.0, turn_ok,
                    f"RSI rising={rsi_rising} MACD up={macd_up}"))

    # --- IMPORTANT ---
    factors.append(("RSI not overbought", 1.5, rsi_v < 60, f"RSI {rsi_v:.0f}"))
    factors.append(("Local structure OK", 1.5, local_up or htf == "UP" or rsi_v < 30, "EMA / HTF / deep OS"))
    factors.append(("Not extended above value", 1.5, bb < 0.55 or rsi_v < 45, f"BB {bb*100:.0f}%"))

    # volume: very dead tape is a soft negative, not a hard fail
    factors.append(("Volume not dead", 1.0, vol_ratio >= 0.45, f"{vol_ratio:.1f}x MA"))

    # --- CONTEXT (only if data exists) ---
    if fng_val is not None:
        # Greed is OK only if we're truly at value; extreme greed blocks weak setups
        fng_ok = fng_val < 80 or (fng_val < 90 and at_value and rsi_v < 32)
        factors.append(("Sentiment not reckless greed", 1.5, fng_ok, f"{fng_val} {fng_label}"))
        factors.append(("Fear supports buy-low", 1.0, fng_val <= 45, f"{fng_val} {fng_label}"))

    if global_m.get("mcap_chg_24h") is not None:
        chg = float(global_m["mcap_chg_24h"])
        factors.append(("Broad market not crashing", 2.0, chg > -6.0, f"Mcap 24h {chg:+.1f}%"))

    if macro.get("DGS10") is not None:
        factors.append(("Yields not in spike panic zone", 1.0, macro["DGS10"] < 5.75, f"10Y {macro['DGS10']:.2f}%"))

    # Score
    score = sum(w for _, w, ok, _ in factors if ok)
    max_score = sum(w for _, w, _, _ in factors)
    ratio = score / max_score if max_score else 0

    # Hard blocks
    hard_block = []
    if htf == "DOWN" and rsi_v > 32:
        hard_block.append("HTF DOWN and not deeply oversold")
    if fng_val is not None and fng_val >= 85 and bb > 0.4:
        hard_block.append("Extreme greed + not cheap")
    if global_m.get("mcap_chg_24h") is not None and float(global_m["mcap_chg_24h"]) <= -8:
        hard_block.append("Severe risk-off day (mcap)")

    # SELL HIGH: take profit / exit zone for holders
    sell_high = (
        (rsi_v >= 68 and bb >= 0.7)
        or (rsi_v >= 72)
        or (bb >= 0.88)
        or (fng_val is not None and fng_val >= 80 and bb >= 0.65 and rsi_v >= 60)
    )

    # BUY: must be at value, pass HTF rule, no hard block, strong score
    buy_low = (
        not sell_high
        and not hard_block
        and at_value
        and htf_ok
        and ratio >= 0.62
        and score >= 7.0
    )

    if buy_low:
        action = "BUY"
        conf = "HIGH" if ratio >= 0.78 and turn_ok else "MED"
    elif sell_high:
        action = "SELL"
        exit_px = price
        stop = None
        conf = "HIGH" if rsi_v >= 72 or bb >= 0.85 else "MED"
    else:
        action = "WAIT"
        conf = "—"

    checklist = [(n, ok, d) for n, w, ok, d in factors]

    return {
        "action": action,
        "confidence": conf,
        "entry": price if action == "BUY" else None,
        "exit": exit_px if action != "WAIT" else float(last["BB_u"]),
        "stop": stop,
        "buy_zone": float(last["BB_l"]),
        "sell_zone": float(last["BB_u"]),
        "limit_buy": float(last["BB_l"]),
        "checks": checklist,
        "score": round(score, 1),
        "max_score": round(max_score, 1),
        "ratio": ratio,
        "hard_block": hard_block,
        "rsi": rsi_v,
        "bb": bb,
        "atr": atr,
        "price": price,
    }


def chart_price(df):
    plot = df[["Close"]].tail(120).reset_index()
    plot.columns = ["Time", "Price"]
    ymin, ymax = float(plot["Price"].min()), float(plot["Price"].max())
    pad = (ymax - ymin) * 0.08 or 1
    ch = (
        alt.Chart(plot).mark_line(color="#42a5f5", strokeWidth=2)
        .encode(
            x=alt.X("Time:T", title=None),
            y=alt.Y("Price:Q", scale=alt.Scale(domain=[ymin - pad, ymax + pad], zero=False), title=None),
            tooltip=["Time:T", alt.Tooltip("Price:Q", format=".4f")],
        )
        .properties(height=300)
        .interactive()
    )
    st.altair_chart(ch, use_container_width=True)


# ── Sidebar ───────────────────────────────────────────────────
fred_key = get_fred_key()
st.sidebar.title("Spot Desk")
name = st.sidebar.selectbox("Coin", list(CRYPTO.keys()), index=0)
symbol = CRYPTO[name]
custom = st.sidebar.text_input("Or pair", placeholder="BTCUSDT")
if custom.strip():
    symbol = custom.strip().upper()
    name = symbol

interval = st.sidebar.selectbox("Timeframe", INTERVALS, index=0)
account = st.sidebar.number_input("Account $", 100.0, value=10000.0, step=100.0)
risk_pct = st.sidebar.slider("Risk % if stopped", 0.25, 2.0, 0.5, 0.25)

st.sidebar.markdown("---")
live = st.sidebar.checkbox("🔴 Live feed ON", value=False)
secs = st.sidebar.slider("Refresh seconds", 20, 120, 45)
alert_on = st.sidebar.checkbox("Alert on BUY / SELL", value=True)
webhook = st.sidebar.text_input("Webhook (optional)", placeholder="Discord webhook URL")

if fred_key:
    st.sidebar.success("FRED key loaded")
else:
    st.sidebar.caption("Optional: secret FRED_API_KEY for macro")

if st.sidebar.button("Refresh now"):
    st.cache_data.clear()
    st.rerun()

# ── Data ──────────────────────────────────────────────────────
now = datetime.now(timezone.utc)
st.title(f"{name}")
if live:
    st.markdown(
        f"<span class='live-dot'>● LIVE</span> · {interval} · {now.strftime('%H:%M:%S')} UTC",
        unsafe_allow_html=True,
    )
else:
    st.caption(f"{interval} · {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")

with st.spinner("Analyzing…"):
    live_px = binance_ticker_price(symbol)
    raw = binance_klines(symbol, interval, 300)
    htf_raw = binance_klines(symbol, HTF_MAP.get(interval, "1h"), 150)
    htf = trend_bias(htf_raw) if not htf_raw.empty else "RANGE"
    fng_val, fng_label = fear_greed()
    global_m = coingecko_global()
    macro = load_macro(fred_key)
    news = news_headlines(name if name.replace(" ", "").isalpha() else "crypto", 6)

if raw.empty:
    st.error("No price data.")
    st.stop()

df = technicals(raw)
if df.empty:
    st.error("Not enough data.")
    st.stop()

last = df.iloc[-1]
dec = decide(last, htf, fng_val, fng_label, global_m, macro)

# Live price for entry/exit display
if live_px is not None:
    dec["price"] = live_px
    if dec["action"] == "BUY":
        dec["entry"] = live_px
        atr = dec["atr"]
        stop = min(live_px - 1.5 * atr, float(last["Swing_low"]) * 0.998)
        risk = max(live_px - stop, atr * 0.4)
        dec["stop"] = stop
        dec["exit"] = live_px + 2.0 * risk
    elif dec["action"] == "SELL":
        dec["exit"] = live_px

action = dec["action"]
prev = st.session_state.get("last_action")

if alert_on and action in ("BUY", "SELL"):
    if action == "BUY":
        st.markdown(
            f"<div class='alert-buy'>🔔 BUY — market {fmt(dec['entry'])} · sell target {fmt(dec['exit'])} · conf {dec['confidence']}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='alert-sell'>🔔 SELL — exit {fmt(dec['exit'])} · conf {dec['confidence']}</div>",
            unsafe_allow_html=True,
        )
    if prev != action:
        st.toast(f"{action} {name}", icon="🔔")
        if webhook.strip():
            send_webhook(
                webhook.strip(),
                f"{action} {name} @ {fmt(dec['price'])} entry={fmt(dec.get('entry'))} exit={fmt(dec.get('exit'))} conf={dec['confidence']}",
            )
st.session_state["last_action"] = action

size = 0.0
if action == "BUY" and dec.get("stop") and dec.get("entry"):
    risk_amt = dec["entry"] - dec["stop"]
    if risk_amt > 0:
        size = (account * risk_pct / 100.0) / risk_amt

# Decision card
if action == "BUY":
    lim = dec.get("limit_buy")
    lim_h = f"<p class='small'>Limit idea (below market): {fmt(lim)}</p>" if lim and lim < dec["entry"] else ""
    st.markdown(
        f"""
        <div class="card">
          <div class="small">BEST EFFORT SPOT CALL · confidence {dec['confidence']}</div>
          <div class="big buy">BUY</div>
          <p style="font-size:1.15rem;margin-top:0.8rem">
            <b>Entry</b> (buy at market)<br><span class="big buy">{fmt(dec['entry'])}</span>
          </p>
          <p style="font-size:1.15rem">
            <b>Exit target</b> (sell later — above entry)<br><span class="big sell">{fmt(dec['exit'])}</span>
          </p>
          {lim_h}
          <p class="small">Stop {fmt(dec['stop'])} · Size ≈ {size:.6g} · Score {dec['score']}/{dec['max_score']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif action == "SELL":
    st.markdown(
        f"""
        <div class="card">
          <div class="small">BEST EFFORT SPOT CALL · confidence {dec['confidence']}</div>
          <div class="big sell">SELL</div>
          <p style="font-size:1.15rem;margin-top:0.8rem">
            <b>Exit</b> (sell coin for cash)<br><span class="big sell">{fmt(dec['exit'])}</span>
          </p>
          <p class="small">Next buy zone ≈ {fmt(dec['buy_zone'])}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    reason = "; ".join(dec["hard_block"]) if dec["hard_block"] else "Setup not strong enough"
    st.markdown(
        f"""
        <div class="card">
          <div class="small">BEST EFFORT SPOT CALL</div>
          <div class="big wait">WAIT</div>
          <p>Price <b>{fmt(dec['price'])}</b></p>
          <p>Buy zone ≈ {fmt(dec['buy_zone'])} · Sell zone ≈ {fmt(dec['sell_zone'])}</p>
          <p class="small">Score {dec['score']}/{dec['max_score']} · {reason}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Price", fmt(dec["price"]))
c2.metric("RSI", f"{dec['rsi']:.0f}")
c3.metric("HTF", htf)
c4.metric("Fear & Greed", f"{fng_val}" if fng_val is not None else "—", fng_label)
c5.metric("Score", f"{dec['score']}/{dec['max_score']}")

st.subheader("Why (weighted checklist)")
st.caption("Only factors with data are scored. Blank futures feeds are ignored on purpose.")
for label, ok, detail in dec["checks"]:
    st.markdown(
        f"{'<span class=ok>✓</span>' if ok else '<span class=no>✗</span>'} **{label}** — {detail}",
        unsafe_allow_html=True,
    )

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Context")
    st.write(f"BB position: **{dec['bb']*100:.0f}%** · ATR: **{fmt(dec['atr'])}**")
    if global_m.get("mcap_chg_24h") is not None:
        st.write(f"Global mcap 24h: **{float(global_m['mcap_chg_24h']):+.2f}%**")
    if global_m.get("btc_dom") is not None:
        st.write(f"BTC dominance: **{float(global_m['btc_dom']):.1f}%**")
with col_b:
    st.subheader("Macro")
    st.write(f"10Y: **{macro['DGS10']:.2f}%**" if macro.get("DGS10") is not None else "10Y: —")
    st.write(f"Fed/short: **{macro['FEDFUNDS']:.2f}%**" if macro.get("FEDFUNDS") is not None else "Fed/short: —")
    st.write(f"Unemployment: **{macro['UNRATE']:.2f}%**" if macro.get("UNRATE") is not None else "Unemployment: —")

st.subheader("News")
if news:
    for n in news:
        st.markdown(f"- [{n['title']}]({n['link']})" if n.get("link") else f"- {n['title']}")
else:
    st.caption("No headlines.")

st.subheader("Price")
chart_price(df)

with st.expander("How decisions are made"):
    st.markdown("""
**Goal:** fewer, better spot calls — buy low, sell high.

**BUY only if**
- Price is in a **value** zone (low BB or low RSI)
- Higher timeframe is **not DOWN** (or very deep oversold exception is limited)
- Weighted score is strong (~62%+ of available points)
- No hard block (extreme greed + expensive, or severe market crash)

**SELL when** RSI/BB (and sometimes greed) say **expensive** — for people who already hold the coin.

**Ignored when missing:** funding, open interest (often blocked on cloud). They do not tank the score.

Prefer **HIGH** confidence. **WAIT** is a valid decision.
""")

st.caption("Not financial advice · You place all orders")

if live:
    time.sleep(secs)
    st.rerun()
