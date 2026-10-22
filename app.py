"""
Crypto Spot Desk — Buy Low / Sell High
======================================
Spot only (cash ↔ coin). No shorts. No leverage. No learning module.

Pulls public context from:
  • Binance (price, volume, funding bias)
  • Crypto Fear & Greed (alternative.me)
  • CoinGecko global (dominance, mcap)
  • FRED macro (10Y, Fed funds, unemployment via public CSV)
  • News headlines (RSS)

Outputs a simple ENTRY and EXIT price with a trader-style checklist.
Not financial advice. Not guaranteed profit. You place orders yourself.
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

st.set_page_config(page_title="Spot Desk · Buy Low Sell High", page_icon="⚡", layout="wide")

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
.ok { color: #00c853; } .no { color: #ff5252; } .mu { color: #ffc107; }
.small { color: #9e9e9e; font-size: 0.88rem; }
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


# ── Data fetchers ─────────────────────────────────────────────
@st.cache_data(ttl=30, show_spinner=False)
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


@st.cache_data(ttl=60, show_spinner=False)
def binance_funding(symbol):
    """Perp funding as sentiment (even for spot decisions)."""
    for host in ["https://fapi.binance.com", "https://data-api.binance.vision"]:
        try:
            # last funding rate
            r = requests.get(
                f"{host}/fapi/v1/fundingRate",
                params={"symbol": symbol, "limit": 3},
                timeout=8,
            )
            if r.status_code != 200:
                continue
            data = r.json()
            if data:
                return float(data[-1]["fundingRate"])
        except Exception:
            continue
    return None


@st.cache_data(ttl=120, show_spinner=False)
def fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8)
        d = r.json()["data"][0]
        return int(d["value"]), d["value_classification"]
    except Exception:
        return None, None


@st.cache_data(ttl=180, show_spinner=False)
def coingecko_global():
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        g = r.json()["data"]
        return {
            "mcap_chg_24h": g.get("market_cap_change_percentage_24h_usd"),
            "btc_dom": g.get("market_cap_percentage", {}).get("btc"),
            "eth_dom": g.get("market_cap_percentage", {}).get("eth"),
        }
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def fred_last(series_id: str):
    """Public FRED CSV — no API key required."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent": "SpotDesk/1.0"})
        if r.status_code != 200:
            return None, None
        df = pd.read_csv(StringIO(r.text))
        df = df.replace(".", np.nan).dropna()
        if df.empty or len(df.columns) < 2:
            return None, None
        df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        df = df.dropna()
        return float(df.iloc[-1, 1]), str(df.iloc[-1, 0])
    except Exception:
        return None, None


@st.cache_data(ttl=600, show_spinner=False)
def news_headlines(query: str = "cryptocurrency", limit: int = 8):
    """RSS — CoinDesk + Google News."""
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
                title = item.findtext("title") or ""
                link = item.findtext("link") or ""
                pub = item.findtext("pubDate") or ""
                if title:
                    items.append({"title": title.strip(), "link": link, "pub": pub})
                if len(items) >= limit:
                    break
        except Exception:
            continue
        if len(items) >= limit:
            break
    return items[:limit]


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n, min_periods=n).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / n, min_periods=n).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

def atr_s(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def trend_bias(df):
    if df.empty or len(df) < 30:
        return "RANGE"
    c = df["Close"]
    e12, e26, e50 = ema(c, 12), ema(c, 26), ema(c, min(50, len(df) // 2))
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
    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    d["BB_u"], d["BB_l"] = mid + 2 * std, mid - 2 * std
    d["BB_pct"] = (c - d["BB_l"]) / (d["BB_u"] - d["BB_l"]).replace(0, np.nan)
    d["Swing_low"] = d["Low"].rolling(20).min()
    d["Vol_MA"] = d["Volume"].rolling(20).mean()
    d["Vol_ratio"] = d["Volume"] / d["Vol_MA"].replace(0, np.nan)
    return d.dropna()


def trader_decision(last, htf, fng_val, fng_label, global_m, funding, macro):
    """
    Spot trader logic: combine technicals + sentiment + macro.
    Returns action BUY|SELL|WAIT, entry, exit, stop, reasons list.
    """
    price = float(last["Close"])
    rsi_v = float(last["RSI"])
    bb = float(last["BB_pct"]) if pd.notna(last["BB_pct"]) else 0.5
    atr = float(last["ATR"]) if pd.notna(last["ATR"]) else price * 0.01
    local_up = float(last["EMA12"]) > float(last["EMA26"])
    vol_ok = float(last.get("Vol_ratio", 1) or 1) > 0.8

    stop = min(price - 1.5 * atr, float(last["Swing_low"]) * 0.998)
    risk = max(price - stop, atr * 0.5)
    exit_px = price + 2.0 * risk  # 2R target

    checks = []  # (name, bullish_for_buy, detail)

    # Technical value
    checks.append(("Price near lows (BB)", bb < 0.35, f"BB position {bb*100:.0f}%"))
    checks.append(("RSI not expensive", rsi_v < 55, f"RSI {rsi_v:.0f}"))
    checks.append(("RSI oversold bonus", rsi_v < 35, f"RSI {rsi_v:.0f}"))
    checks.append(("Local trend OK", local_up or htf != "DOWN", "EMA12 vs EMA26"))
    checks.append(("HTF not hard DOWN", htf in ("UP", "RANGE"), f"HTF={htf}"))
    checks.append(("Volume alive", vol_ok, f"Vol vs MA {float(last.get('Vol_ratio') or 0):.1f}x"))

    # Sentiment
    if fng_val is not None:
        checks.append(("Fear & Greed not extreme greed", fng_val < 75, f"{fng_val} ({fng_label})"))
        checks.append(("Fear supports buy-low", fng_val <= 40, f"{fng_val} ({fng_label})"))
    if funding is not None:
        # negative funding → perps short squeeze risk / cheaper to be long-biased spot
        checks.append(("Funding not overheated long", funding < 0.0005, f"{funding*100:.4f}%"))
    if global_m.get("mcap_chg_24h") is not None:
        chg = float(global_m["mcap_chg_24h"])
        checks.append(("Market not in free-fall", chg > -5, f"Global mcap 24h {chg:+.1f}%"))
    if global_m.get("btc_dom") is not None:
        checks.append(("BTC dominance readable", True, f"BTC dom {float(global_m['btc_dom']):.1f}%"))

    # Macro: rising yields often risk-off for crypto
    dgs10 = macro.get("DGS10")
    fed = macro.get("FEDFUNDS")
    if dgs10 is not None:
        checks.append(("10Y yield context loaded", True, f"10Y {dgs10:.2f}%"))
        # very high yields = caution on new risk buys
        checks.append(("Yields not extreme spike zone", dgs10 < 5.5, f"10Y {dgs10:.2f}%"))
    if fed is not None:
        checks.append(("Fed funds loaded", True, f"FF {fed:.2f}%"))

    bullish_votes = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)

    # Sell-high: expensive technicals
    sell_high = rsi_v >= 65 or bb >= 0.78 or (fng_val is not None and fng_val >= 78 and bb > 0.6)

    # Buy-low: enough confluence + value
    buy_low = (
        not sell_high
        and htf != "DOWN"
        and bullish_votes >= max(4, total // 2)
        and (bb < 0.4 or rsi_v < 40)
    )

    if buy_low:
        action = "BUY"
    elif sell_high:
        action = "SELL"
        exit_px = price  # sell at market when signal is sell
        stop = price  # N/A for pure exit
    else:
        action = "WAIT"

    return {
        "action": action,
        "entry": price if action == "BUY" else None,
        "exit": exit_px if action == "BUY" else (price if action == "SELL" else float(last["BB_u"])),
        "stop": stop if action == "BUY" else None,
        "buy_zone": float(last["BB_l"]),
        "sell_zone": float(last["BB_u"]),
        "checks": checks,
        "votes": bullish_votes,
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
        alt.Chart(plot)
        .mark_line(color="#42a5f5", strokeWidth=2)
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
if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Sources: Binance · Fear&Greed · CoinGecko · FRED · RSS news\n"
    "Spot only. You execute trades yourself."
)

# ── Load everything ───────────────────────────────────────────
st.title(f"{name}")
st.caption(f"Spot desk · {interval} · {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

with st.spinner("Pulling exchange, macro, sentiment, news…"):
    raw = binance_klines(symbol, interval, 250)
    htf_raw = binance_klines(symbol, HTF_MAP.get(interval, "1h"), 120)
    htf = trend_bias(htf_raw) if not htf_raw.empty else "RANGE"
    fng_val, fng_label = fear_greed()
    global_m = coingecko_global()
    funding = binance_funding(symbol)
    dgs10, dgs10_d = fred_last("DGS10")
    fed, fed_d = fred_last("FEDFUNDS")
    unrate, unrate_d = fred_last("UNRATE")
    macro = {"DGS10": dgs10, "FEDFUNDS": fed, "UNRATE": unrate}
    news = news_headlines(name if name.isalpha() else "crypto", 8)

if raw.empty:
    st.error("No exchange price data. Try again or another pair.")
    st.stop()

df = technicals(raw)
if df.empty:
    st.error("Not enough bars for indicators.")
    st.stop()

last = df.iloc[-1]
dec = trader_decision(last, htf, fng_val, fng_label, global_m, funding, macro)

# Position size on buy
size = 0.0
if dec["action"] == "BUY" and dec["stop"] and dec["entry"]:
    risk_amt = dec["entry"] - dec["stop"]
    if risk_amt > 0:
        size = (account * risk_pct / 100.0) / risk_amt

# ── MAIN: Entry / Exit ────────────────────────────────────────
if dec["action"] == "BUY":
    st.markdown(
        f"""
        <div class="card">
          <div class="small">TRADER DECISION · SPOT</div>
          <div class="big buy">BUY</div>
          <p style="font-size:1.2rem;margin-top:0.8rem">
            <b>Entry price</b><br><span class="big buy">{fmt(dec['entry'])}</span>
          </p>
          <p style="font-size:1.2rem">
            <b>Exit price</b> (take profit target)<br><span class="big sell">{fmt(dec['exit'])}</span>
          </p>
          <p class="small">
            Stop: {fmt(dec['stop'])} ·
            Size ≈ {size:.6g} coins at {risk_pct}% risk on ${account:,.0f} ·
            Confluence {dec['votes']}/{dec['total']}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif dec["action"] == "SELL":
    st.markdown(
        f"""
        <div class="card">
          <div class="small">TRADER DECISION · SPOT</div>
          <div class="big sell">SELL</div>
          <p style="font-size:1.2rem;margin-top:0.8rem">
            <b>Exit price</b> (sell coin for cash)<br><span class="big sell">{fmt(dec['exit'])}</span>
          </p>
          <p class="small">If you hold this coin, consider selling. Not a short. Next buy zone ≈ {fmt(dec['buy_zone'])}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div class="card">
          <div class="small">TRADER DECISION · SPOT</div>
          <div class="big wait">WAIT</div>
          <p style="margin-top:0.8rem">Current price <b>{fmt(dec['price'])}</b></p>
          <p>
            Watch <b>entry zone</b> ≈ {fmt(dec['buy_zone'])}<br>
            Watch <b>exit zone</b> ≈ {fmt(dec['sell_zone'])}
          </p>
          <p class="small">Confluence {dec['votes']}/{dec['total']} — not enough for a clean buy-low setup.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Context strip
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Price", fmt(dec["price"]))
c2.metric("RSI", f"{dec['rsi']:.0f}")
c3.metric("HTF trend", htf)
c4.metric("Fear & Greed", f"{fng_val}" if fng_val is not None else "—", fng_label or None)
c5.metric("BTC dom", f"{global_m['btc_dom']:.1f}%" if global_m.get("btc_dom") else "—")

st.divider()

# Market context panels
left, right = st.columns(2)
with left:
    st.subheader("Exchange & technicals")
    st.write(f"BB position: **{dec['bb']*100:.0f}%** (0=low, 100=high)")
    st.write(f"ATR: **{fmt(dec['atr'])}**")
    if funding is not None:
        st.write(f"Perp funding (sentiment): **{funding*100:.4f}%**")
    else:
        st.write("Perp funding: unavailable")
    if global_m.get("mcap_chg_24h") is not None:
        st.write(f"Global crypto mcap 24h: **{float(global_m['mcap_chg_24h']):+.2f}%**")

with right:
    st.subheader("Macro (FRED)")
    st.write(f"10Y Treasury: **{dgs10:.2f}%**" if dgs10 is not None else "10Y: —")
    st.write(f"Fed funds: **{fed:.2f}%**" if fed is not None else "Fed funds: —")
    st.write(f"Unemployment: **{unrate:.2f}%**" if unrate is not None else "Unemployment: —")
    st.caption("Public FRED series · risk assets often sensitive to yields")

st.subheader("Checklist (why buy / wait)")
for label, ok, detail in dec["checks"]:
    mark = f"<span class='ok'>✓</span>" if ok else f"<span class='no'>✗</span>"
    st.markdown(f"{mark} **{label}** — {detail}", unsafe_allow_html=True)

st.subheader("News")
if news:
    for n in news[:6]:
        title = n["title"]
        link = n.get("link") or ""
        if link:
            st.markdown(f"- [{title}]({link})")
        else:
            st.markdown(f"- {title}")
else:
    st.caption("No headlines loaded (RSS may be blocked on this host).")

st.subheader("Price")
chart_price(df)

with st.expander("How this desk thinks"):
    st.markdown("""
**Spot only:** BUY = spend cash for coin · SELL = sell coin for cash · never short · no leverage.

**Stack of information**
1. **Exchange** — price, RSI, Bollinger, volume, higher-timeframe trend, funding sentiment  
2. **Sentiment** — Crypto Fear & Greed  
3. **Market** — CoinGecko global mcap change, BTC dominance  
4. **Macro** — FRED 10Y yield, Fed funds, unemployment  
5. **News** — recent headlines for awareness (not auto-trading on headlines)

**Buy low** when enough checklist items pass and price is in a value zone.  
**Sell high** when RSI/BB (and extreme greed) say expensive.

This does **not** guarantee profit. Data can fail; news is lagged; you execute on your exchange.
""")

st.caption("Not financial advice · Public data only · No auto-orders")
