"""
Crypto Signal Bot Pro — working charts
======================================
Native Streamlit charts + simple Plotly candlestick (no broken subplots).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone
import requests
import time

st.set_page_config(page_title="Crypto Signal Bot Pro", page_icon="⚡", layout="wide")

st.markdown("""
<style>
div[data-testid="stMetricValue"] { font-size: 1.4rem; }
.signal-buy  { color: #00c853; font-weight: 800; font-size: 1.6rem; }
.signal-sell { color: #ff1744; font-weight: 800; font-size: 1.6rem; }
.signal-hold { color: #ffab00; font-weight: 800; font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)

CRYPTO = {
    "Bitcoin": "BTCUSDT", "Ethereum": "ETHUSDT", "Solana": "SOLUSDT",
    "BNB": "BNBUSDT", "XRP": "XRPUSDT", "Cardano": "ADAUSDT",
    "Dogecoin": "DOGEUSDT", "Avalanche": "AVAXUSDT", "Chainlink": "LINKUSDT",
    "Polkadot": "DOTUSDT", "Litecoin": "LTCUSDT", "Shiba Inu": "SHIBUSDT",
    "Uniswap": "UNIUSDT", "Near": "NEARUSDT", "Aptos": "APTUSDT",
    "Arbitrum": "ARBUSDT", "Sui": "SUIUSDT", "Toncoin": "TONUSDT",
    "Tron": "TRXUSDT", "Pepe": "PEPEUSDT", "Aave": "AAVEUSDT",
    "Render": "RENDERUSDT",
}
INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]
HOSTS = ["https://data-api.binance.vision", "https://api.binance.us"]


@st.cache_data(ttl=20, show_spinner=False)
def fetch_klines(symbol: str, interval: str, limit: int = 300):
    params = {"symbol": symbol, "interval": interval, "limit": min(int(limit), 1000)}
    for host in HOSTS:
        try:
            r = requests.get(f"{host}/api/v3/klines", params=params, timeout=12)
            if r.status_code != 200:
                continue
            raw = r.json()
            if not isinstance(raw, list) or len(raw) < 10:
                continue
            df = pd.DataFrame(raw, columns=[
                "t", "Open", "High", "Low", "Close", "Volume",
                "T", "q", "n", "V", "Q", "x"
            ])
            df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_localize(None)
            df = df.set_index("t")
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if len(df) >= 10:
                return df, host.replace("https://", "")
        except Exception:
            continue
    return pd.DataFrame(), "FAILED"


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    gain = d.clip(lower=0).ewm(alpha=1/n, min_periods=n).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/n, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 30:
        return pd.DataFrame()
    d = df.copy()
    c = d["Close"]
    d["EMA_fast"] = ema(c, 12)
    d["EMA_slow"] = ema(c, 26)
    d["RSI"] = rsi(c)
    d["MACD"] = d["EMA_fast"] - d["EMA_slow"]
    d["MACD_sig"] = ema(d["MACD"], 9)
    d["MACD_hist"] = d["MACD"] - d["MACD_sig"]
    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    d["BB_u"] = mid + 2 * std
    d["BB_l"] = mid - 2 * std

    score = pd.Series(0.0, index=d.index)
    score += np.where(d["EMA_fast"] > d["EMA_slow"], 25, -25)
    score += np.where(c > d["EMA_slow"], 10, -10)
    score += np.where(d["RSI"] < 30, 20, 0)
    score += np.where(d["RSI"] > 70, -20, 0)
    score += np.where(d["MACD"] > d["MACD_sig"], 15, -15)
    d["Score"] = score.clip(-100, 100).round(0)
    d["Signal"] = d["Score"].map(lambda x: "BUY" if x >= 35 else ("SELL" if x <= -35 else "HOLD"))
    return d.dropna(subset=["RSI"])


def simple_candlestick(df: pd.DataFrame, title: str) -> go.Figure:
    """Single-panel candlestick — no make_subplots (those break on some Streamlit Cloud hosts)."""
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="OHLC",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",
        decreasing_fillcolor="#ef5350",
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA_fast"], name="EMA 12",
        line=dict(color="#ff6b6b", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA_slow"], name="EMA 26",
        line=dict(color="#51cf66", width=1.5),
    ))

    # BUY / SELL markers
    buys = df[df["Signal"] == "BUY"].tail(10)
    sells = df[df["Signal"] == "SELL"].tail(10)
    if len(buys):
        fig.add_trace(go.Scatter(
            x=buys.index, y=buys["Low"] * 0.998,
            mode="markers", name="BUY",
            marker=dict(symbol="triangle-up", size=10, color="#26a69a"),
        ))
    if len(sells):
        fig.add_trace(go.Scatter(
            x=sells.index, y=sells["High"] * 1.002,
            mode="markers", name="SELL",
            marker=dict(symbol="triangle-down", size=10, color="#ef5350"),
        ))

    fig.update_layout(
        title=title,
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font_color="#fafafa",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
        dragmode="zoom",
        modebar_add=["drawline", "drawopenpath", "drawclosedpath",
                     "drawcircle", "drawrect", "eraseshape"],
        xaxis=dict(gridcolor="#333"),
        yaxis=dict(gridcolor="#333"),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("🛠️ Control Room")
name = st.sidebar.selectbox("Asset", list(CRYPTO.keys()), index=0)
symbol = CRYPTO[name]
custom = st.sidebar.text_input("Custom symbol (e.g. BTCUSDT)", "")
if custom.strip():
    symbol = custom.strip().upper()
    name = symbol

interval = st.sidebar.selectbox("Candle size", INTERVALS, index=3)
limit = st.sidebar.select_slider("Candles", options=[100, 200, 300, 500], value=200)

st.sidebar.markdown("---")
st.sidebar.subheader("🔴 Live feed")
live = st.sidebar.checkbox("Auto-refresh ON", value=False)
secs = st.sidebar.slider("Seconds between updates", 8, 60, 15)
if st.sidebar.button("🔄 Refresh now", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption("Turn Auto-refresh on after the chart appears.")

# ── Main ──────────────────────────────────────────────────────
st.title("⚡ Crypto Signal Bot Pro")
st.caption(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC · {name} · {symbol} · {interval}")

with st.spinner("Loading…"):
    raw, source = fetch_klines(symbol, interval, limit)
    df = add_signals(raw)

if df.empty:
    st.error("No data. Try another interval or symbol.")
    st.stop()

st.success(f"Data OK · **{source}** · {len(df)} candles · {interval}")

latest = df.iloc[-1]
prev = df.iloc[-2]
delta = float(latest["Close"] - prev["Close"])
price = float(latest["Close"])
price_s = f"${price:,.6f}" if price < 1 else (f"${price:,.4f}" if price < 100 else f"${price:,.2f}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Price", price_s, f"{delta:+.6f}" if abs(delta) < 1 else f"{delta:+.2f}")
sig = str(latest["Signal"])
cls = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}.get(sig, "")
c2.markdown("**Signal**")
c2.markdown(f"<div class='{cls}'>{sig}</div>", unsafe_allow_html=True)
c3.metric("RSI", f"{float(latest['RSI']):.1f}")
c4.metric("Score", f"{int(latest['Score'])}")
c5.metric("Candles", len(df))

st.divider()

# ── Candlestick (simple, single panel) ────────────────────────
st.subheader("📊 Candlestick Chart")
try:
    fig = simple_candlestick(df, f"{name} · {interval}")
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToAdd": [
                "drawline", "drawopenpath", "drawclosedpath",
                "drawcircle", "drawrect", "eraseshape",
            ],
        },
    )
except Exception as e:
    st.warning(f"Candlestick unavailable ({e}). Showing line chart instead.")
    st.line_chart(df[["Close", "EMA_fast", "EMA_slow"]], height=400)

# ── RSI + Volume as native charts (always work) ───────────────
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("RSI (14)")
    st.line_chart(df[["RSI"]], height=220)
with col_b:
    st.subheader("Volume")
    st.bar_chart(df[["Volume"]], height=220)

st.subheader("Recent candles")
st.dataframe(
    df[["Open", "High", "Low", "Close", "Volume", "RSI", "Score", "Signal"]]
      .sort_index(ascending=False)
      .head(40),
    use_container_width=True,
    height=280,
)

if live:
    st.caption(f"Refreshing in {secs}s…")
    time.sleep(secs)
    st.rerun()
