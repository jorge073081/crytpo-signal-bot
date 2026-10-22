"""
Crypto Spot Desk — Buy Low / Sell High + Live feed + Alerts
===========================================================
Spot only. No shorts. No leverage. No learning module.

Data (public):
  Binance price/volume/funding/OI/long-short
  Fear & Greed, CoinGecko global + trending
  FRED macro, RSS news, mempool (BTC)

Live feed: optional auto-refresh
Alerts: on-screen + toast + optional webhook (Discord/Telegram URL)

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
from io import StringIO
import json

st.set_page_config(page_title="Spot Desk · Live", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.big { font-size: 1.85rem; font-weight: 800; margin: 0.15rem 0; }
.buy { color: #00c853; }
.sell { color: #ffab00; }
.wait { color: #90a4ae; }
.card {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 12px;
  padding: 1.2rem 1.4rem;
  margin-bottom: 0.6rem;
}
.alert-buy {
  background: rgba(0,200,83,0.18);
  border-left: 5px solid #00c853;
  padding: 12px 16px; border-radius: 8px; font-weight: 700; margin: 8px 0;
}
.alert-sell {
  background: rgba(255,171,0,0.18);
  border-left: 5px solid #ffab00;
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


@st.cache_data(ttl=20, show_spinner=False)
def binance_klines(symbol, interval, limit=250):
    params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
    for host in BINANCE:
        try:
            r = requests.get(f"{host}/api/v3/klines", params=params, timeout=10)
            if r.status_code != 200:
                continue
            raw = r.json()
            if not isinstance(raw, list) or len(raw) < 30:
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



@st.cache_data(ttl=10, show_spinner=False)
def binance_ticker_price(symbol):
    """Live last trade price (more current than last closed candle)."""
    for host in BINANCE:
        try:
            r = requests.get(f"{host}/api/v3/ticker/price", params={"symbol": symbol}, timeout=8)
            if r.status_code == 200:
                return float(r.json()["price"])
        except Exception:
            continue
    return None

@st.cache_data(ttl=45, show_spinner=False)
def binance_funding(symbol):
    for host in ["https://fapi.binance.com", "https://data-api.binance.vision"]:
        try:
            r = requests.get(f"{host}/fapi/v1/fundingRate", params={"symbol": symbol, "limit": 2}, timeout=8)
            if r.status_code == 200 and r.json():
                return float(r.json()[-1]["fundingRate"])
        except Exception:
            continue
    return None


@st.cache_data(ttl=45, show_spinner=False)
def binance_oi(symbol):
    for host in ["https://fapi.binance.com"]:
        try:
            r = requests.get(f"{host}/fapi/v1/openInterest", params={"symbol": symbol}, timeout=8)
            if r.status_code == 200:
                return float(r.json().get("openInterest", 0))
        except Exception:
            continue
    return None


@st.cache_data(ttl=60, show_spinner=False)
def binance_long_short(symbol):
    """Global long/short account ratio (futures sentiment)."""
    try:
        r = requests.get(
            "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
            params={"symbol": symbol, "period": "1h", "limit": 1},
            timeout=8,
        )
        if r.status_code == 200 and r.json():
            row = r.json()[-1]
            return float(row.get("longShortRatio", 0)), float(row.get("longAccount", 0)), float(row.get("shortAccount", 0))
    except Exception:
        pass
    return None, None, None


@st.cache_data(ttl=120, show_spinner=False)
def fear_greed():
    try:
        d = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8).json()["data"][0]
        return int(d["value"]), d["value_classification"]
    except Exception:
        return None, None


@st.cache_data(ttl=180, show_spinner=False)
def coingecko_global():
    try:
        g = requests.get("https://api.coingecko.com/api/v3/global", timeout=10).json()["data"]
        return {
            "mcap_chg_24h": g.get("market_cap_change_percentage_24h_usd"),
            "btc_dom": g.get("market_cap_percentage", {}).get("btc"),
            "eth_dom": g.get("market_cap_percentage", {}).get("eth"),
        }
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def coingecko_trending():
    try:
        coins = requests.get("https://api.coingecko.com/api/v3/search/trending", timeout=10).json().get("coins", [])
        return [c.get("item", {}).get("name") for c in coins[:7] if c.get("item")]
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def fred_last(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent": "SpotDesk/1.0"})
        if r.status_code != 200:
            return None
        df = pd.read_csv(StringIO(r.text)).replace(".", np.nan).dropna()
        if df.empty or len(df.columns) < 2:
            return None
        df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        df = df.dropna()
        return float(df.iloc[-1, 1])
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def mempool_fees():
    try:
        r = requests.get("https://mempool.space/api/v1/fees/recommended", timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


@st.cache_data(ttl=600, show_spinner=False)
def news_headlines(query="cryptocurrency", limit=8):
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


def send_webhook(url: str, content: str):
    if not url or not url.startswith("http"):
        return False
    try:
        # Discord-style
        r = requests.post(url, json={"content": content}, timeout=8)
        if r.status_code < 300:
            return True
        # Telegram bot style: user pastes full API URL sometimes
        r2 = requests.post(url, json={"text": content}, timeout=8)
        return r2.status_code < 300
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
    e50 = ema(c, min(50, max(20, len(df)//2)))
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
    d["RSI"] = rsi(c)
    d["ATR"] = atr_s(d)
    mid, std = c.rolling(20).mean(), c.rolling(20).std()
    d["BB_u"], d["BB_l"] = mid + 2 * std, mid - 2 * std
    d["BB_pct"] = (c - d["BB_l"]) / (d["BB_u"] - d["BB_l"]).replace(0, np.nan)
    d["Swing_low"] = d["Low"].rolling(20).min()
    d["Vol_MA"] = d["Volume"].rolling(20).mean()
    d["Vol_ratio"] = d["Volume"] / d["Vol_MA"].replace(0, np.nan)
    return d.dropna()


def trader_decision(last, htf, fng_val, fng_label, global_m, funding, ls_ratio, macro):
    price = float(last["Close"])
    rsi_v = float(last["RSI"])
    bb = float(last["BB_pct"]) if pd.notna(last["BB_pct"]) else 0.5
    atr = float(last["ATR"]) if pd.notna(last["ATR"]) else price * 0.01
    local_up = float(last["EMA12"]) > float(last["EMA26"])
    vol_ok = float(last.get("Vol_ratio", 1) or 1) > 0.8

    stop = min(price - 1.5 * atr, float(last["Swing_low"]) * 0.998)
    risk = max(price - stop, atr * 0.5)
    exit_px = price + 2.0 * risk

    checks = [
        ("Near value (BB low zone)", bb < 0.35, f"BB {bb*100:.0f}%"),
        ("RSI not expensive", rsi_v < 55, f"RSI {rsi_v:.0f}"),
        ("RSI oversold", rsi_v < 35, f"RSI {rsi_v:.0f}"),
        ("Local trend OK", local_up or htf != "DOWN", "EMA12/26"),
        ("HTF not hard DOWN", htf in ("UP", "RANGE"), f"HTF={htf}"),
        ("Volume alive", vol_ok, f"{float(last.get('Vol_ratio') or 0):.1f}x MA"),
    ]
    if fng_val is not None:
        checks.append(("Not extreme greed", fng_val < 75, f"{fng_val} {fng_label}"))
        checks.append(("Fear helps buy-low", fng_val <= 40, f"{fng_val} {fng_label}"))
    if funding is not None:
        checks.append(("Funding not overheated", funding < 0.0005, f"{funding*100:.4f}%"))
    if ls_ratio is not None:
        # very high long ratio = crowded longs = caution on new buys
        checks.append(("Longs not extremely crowded", ls_ratio < 2.2, f"L/S {ls_ratio:.2f}"))
    if global_m.get("mcap_chg_24h") is not None:
        chg = float(global_m["mcap_chg_24h"])
        checks.append(("Market not crashing", chg > -5, f"Mcap 24h {chg:+.1f}%"))
    if macro.get("DGS10") is not None:
        checks.append(("Yields not extreme", macro["DGS10"] < 5.5, f"10Y {macro['DGS10']:.2f}%"))

    votes = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    sell_high = rsi_v >= 65 or bb >= 0.78 or (fng_val is not None and fng_val >= 78 and bb > 0.6)
    buy_low = (not sell_high) and htf != "DOWN" and votes >= max(4, total // 2) and (bb < 0.4 or rsi_v < 40)

    if buy_low:
        action = "BUY"
    elif sell_high:
        action = "SELL"
        exit_px = price
        stop = None
    else:
        action = "WAIT"

    return {
        "action": action,
        "entry": price if action == "BUY" else None,
        "exit": exit_px if action != "WAIT" else float(last["BB_u"]),
        "stop": stop,
        "buy_zone": float(last["BB_l"]),
        "sell_zone": float(last["BB_u"]),
        "checks": checks,
        "votes": votes,
        "total": total,
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
            y=alt.Y("Price:Q", scale=alt.Scale(domain=[ymin-pad, ymax+pad], zero=False), title=None),
            tooltip=["Time:T", alt.Tooltip("Price:Q", format=".4f")],
        )
        .properties(height=300).interactive()
    )
    st.altair_chart(ch, use_container_width=True)


# ── Sidebar ───────────────────────────────────────────────────
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
st.sidebar.subheader("🔴 Live feed")
live = st.sidebar.checkbox("Auto-refresh ON", value=False)
secs = st.sidebar.slider("Seconds between updates", 15, 120, 30)

st.sidebar.markdown("---")
st.sidebar.subheader("🔔 Alerts")
alert_on = st.sidebar.checkbox("Alert on BUY / SELL", value=True)
webhook = st.sidebar.text_input(
    "Webhook URL (optional)",
    placeholder="Discord webhook or similar",
    help="Posts a message when signal flips to BUY or SELL",
)

if st.sidebar.button("Refresh now"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption("Spot only · public data · you place orders")

# ── Load ──────────────────────────────────────────────────────
now = datetime.now(timezone.utc)
st.title(f"{name}")
if live:
    st.markdown(f"<span class='live-dot'>● LIVE</span> · {interval} · {now.strftime('%H:%M:%S')} UTC · refresh every {secs}s", unsafe_allow_html=True)
else:
    st.caption(f"{interval} · {now.strftime('%Y-%m-%d %H:%M:%S')} UTC · turn on Live feed in sidebar")

with st.spinner("Updating exchange · macro · sentiment · news…"):
    live_px = binance_ticker_price(symbol)
    raw = binance_klines(symbol, interval, 250)
    htf_raw = binance_klines(symbol, HTF_MAP.get(interval, "1h"), 120)
    htf = trend_bias(htf_raw) if not htf_raw.empty else "RANGE"
    fng_val, fng_label = fear_greed()
    global_m = coingecko_global()
    trending = coingecko_trending()
    funding = binance_funding(symbol)
    oi = binance_oi(symbol)
    ls_ratio, long_acc, short_acc = binance_long_short(symbol)
    macro = {
        "DGS10": fred_last("DGS10"),
        "FEDFUNDS": fred_last("FEDFUNDS"),
        "UNRATE": fred_last("UNRATE"),
    }
    fees = mempool_fees() if "BTC" in symbol else {}
    news = news_headlines(name if name.replace(" ", "").isalpha() else "crypto", 8)

if raw.empty:
    st.error("No price data from exchange.")
    st.stop()

df = technicals(raw)
if df.empty:
    st.error("Not enough candles.")
    st.stop()

last = df.iloc[-1]
dec = trader_decision(last, htf, fng_val, fng_label, global_m, funding, ls_ratio, macro)

# Use LIVE ticker for "now" and market entry (candle close can lag)
if live_px is not None:
    dec["price"] = live_px
    if dec["action"] == "BUY":
        # Market entry = live price; limit idea = buy zone if cheaper
        dec["entry"] = live_px
        atr = dec["atr"]
        stop = min(live_px - 1.5 * atr, float(last["Swing_low"]) * 0.998)
        risk = max(live_px - stop, atr * 0.5)
        dec["stop"] = stop
        dec["exit"] = live_px + 2.0 * risk  # sell target ABOVE entry
        dec["limit_buy"] = float(last["BB_l"])  # optional limit below market
    elif dec["action"] == "SELL":
        dec["exit"] = live_px  # sell at live market

# ── Alerts on signal change ───────────────────────────────────
prev = st.session_state.get("last_action")
action = dec["action"]
if alert_on and action in ("BUY", "SELL"):
    msg = (
        f"{'🟢 BUY LOW' if action=='BUY' else '🟠 SELL HIGH'} {name} ({symbol})\n"
        f"Price {fmt(dec['price'])}\n"
        f"Entry {fmt(dec.get('entry'))} · Exit {fmt(dec.get('exit'))}\n"
        f"RSI {dec['rsi']:.0f} · HTF {htf} · {now.strftime('%H:%M UTC')}"
    )
    if action == "BUY":
        st.markdown(f"<div class='alert-buy'>🔔 BUY ALERT — Buy at {fmt(dec['entry'])} (market) · Sell target {fmt(dec['exit'])}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='alert-sell'>🔔 SELL ALERT — Exit {fmt(dec['exit'])} (sell coin for cash)</div>", unsafe_allow_html=True)

    if prev != action:
        st.toast(f"{action} signal on {name}", icon="🔔")
        if webhook.strip():
            ok = send_webhook(webhook.strip(), msg)
            if ok:
                st.caption("Webhook alert sent.")
            else:
                st.caption("Webhook failed (check URL).")
st.session_state["last_action"] = action

# Size
size = 0.0
if action == "BUY" and dec.get("stop") and dec.get("entry"):
    risk_amt = dec["entry"] - dec["stop"]
    if risk_amt > 0:
        size = (account * risk_pct / 100.0) / risk_amt

# ── Main decision card ────────────────────────────────────────
if action == "BUY":
    limit_b = dec.get("limit_buy")
    limit_line = f"<p class=\"small\">Optional limit buy (below market): {fmt(limit_b)}</p>" if limit_b and limit_b < dec["entry"] else ""
    st.markdown(
        f"""
        <div class="card">
          <div class="small">SPOT DECISION · prices are live ticker</div>
          <div class="big buy">BUY</div>
          <p style="font-size:1.15rem;margin-top:0.8rem">
            <b>Entry</b> = buy here (current market)<br>
            <span class="big buy">{fmt(dec['entry'])}</span>
          </p>
          <p style="font-size:1.15rem">
            <b>Exit target</b> = sell later for profit (above entry)<br>
            <span class="big sell">{fmt(dec['exit'])}</span>
          </p>
          {limit_line}
          <p class="small">Stop {fmt(dec['stop'])} · Size ≈ {size:.6g} · Checklist {dec['votes']}/{dec['total']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif action == "SELL":
    st.markdown(
        f"""
        <div class="card">
          <div class="small">SPOT DECISION</div>
          <div class="big sell">SELL</div>
          <p style="font-size:1.2rem;margin-top:0.8rem"><b>Exit price</b><br><span class="big sell">{fmt(dec['exit'])}</span></p>
          <p class="small">Sell coin for cash. Next buy zone ≈ {fmt(dec['buy_zone'])}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div class="card">
          <div class="small">SPOT DECISION</div>
          <div class="big wait">WAIT</div>
          <p>Price <b>{fmt(dec['price'])}</b></p>
          <p>Buy zone ≈ {fmt(dec['buy_zone'])} · Sell zone ≈ {fmt(dec['sell_zone'])}</p>
          <p class="small">Checklist {dec['votes']}/{dec['total']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Price", fmt(dec["price"]))
m2.metric("RSI", f"{dec['rsi']:.0f}")
m3.metric("HTF", htf)
m4.metric("Fear & Greed", f"{fng_val}" if fng_val is not None else "—", fng_label)
m5.metric("BTC dom", f"{global_m['btc_dom']:.1f}%" if global_m.get("btc_dom") else "—")

st.divider()
a, b, c = st.columns(3)
with a:
    st.subheader("Exchange")
    st.write(f"BB: **{dec['bb']*100:.0f}%** · ATR: **{fmt(dec['atr'])}**")
    st.write(f"Funding: **{funding*100:.4f}%**" if funding is not None else "Funding: —")
    st.write(f"Open interest: **{oi:,.0f}**" if oi else "OI: —")
    if ls_ratio is not None:
        st.write(f"Long/Short ratio: **{ls_ratio:.2f}**")
with b:
    st.subheader("Macro (FRED)")
    st.write(f"10Y: **{macro['DGS10']:.2f}%**" if macro.get("DGS10") else "10Y: —")
    st.write(f"Fed funds: **{macro['FEDFUNDS']:.2f}%**" if macro.get("FEDFUNDS") else "Fed: —")
    st.write(f"Unemployment: **{macro['UNRATE']:.2f}%**" if macro.get("UNRATE") else "Unemp: —")
with c:
    st.subheader("Market / chain")
    if global_m.get("mcap_chg_24h") is not None:
        st.write(f"Global mcap 24h: **{float(global_m['mcap_chg_24h']):+.2f}%**")
    if fees:
        st.write(f"BTC fee (fast): **{fees.get('fastestFee', '—')}** sat/vB")
    if trending:
        st.write("Trending: " + ", ".join(trending[:5]))

st.subheader("Checklist")
for label, ok, detail in dec["checks"]:
    st.markdown(f"{'<span class=ok>✓</span>' if ok else '<span class=no>✗</span>'} **{label}** — {detail}", unsafe_allow_html=True)

st.subheader("News")
if news:
    for n in news[:6]:
        if n.get("link"):
            st.markdown(f"- [{n['title']}]({n['link']})")
        else:
            st.markdown(f"- {n['title']}")
else:
    st.caption("No RSS headlines right now.")

st.subheader("Price")
chart_price(df)

st.caption("Not financial advice · Live = polling refresh · Webhook optional for phone/Discord alerts")

# Live feed loop (end of script)
if live:
    time.sleep(secs)
    st.rerun()
