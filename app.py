"""
Crypto Signal Bot Pro — REALTIME chart (working endpoints)
==========================================================
Uses Binance public data hosts that are NOT geo-blocked:
  1. data-api.binance.vision
  2. api.binance.us
Falls back to yfinance only if both fail.

Auto-refreshes every few seconds so candles update live.
Drawing tools enabled on the Plotly chart.

requirements:
  streamlit
  requests
  pandas
  numpy
  plotly
  yfinance   # optional fallback
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone
import requests
import time

st.set_page_config(
    page_title="Crypto Signal Bot Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stMetric { background: rgba(255,255,255,0.05); padding: 10px 14px; border-radius: 10px; }
.signal-buy  { color: #00c853; font-weight: 800; font-size: 1.7rem; }
.signal-sell { color: #ff1744; font-weight: 800; font-size: 1.7rem; }
.signal-hold { color: #ffab00; font-weight: 800; font-size: 1.7rem; }
.live-dot { color: #00c853; font-size: 1.2rem; }
.block-container { padding-top: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ── Tickers (Binance style) ───────────────────────────────────
CRYPTO = {
    "Bitcoin": "BTCUSDT",
    "Ethereum": "ETHUSDT",
    "Solana": "SOLUSDT",
    "BNB": "BNBUSDT",
    "XRP": "XRPUSDT",
    "Cardano": "ADAUSDT",
    "Dogecoin": "DOGEUSDT",
    "Avalanche": "AVAXUSDT",
    "Chainlink": "LINKUSDT",
    "Polkadot": "DOTUSDT",
    "Litecoin": "LTCUSDT",
    "Shiba Inu": "SHIBUSDT",
    "Uniswap": "UNIUSDT",
    "Near": "NEARUSDT",
    "Aptos": "APTUSDT",
    "Arbitrum": "ARBUSDT",
    "Sui": "SUIUSDT",
    "Toncoin": "TONUSDT",
    "Tron": "TRXUSDT",
    "Pepe": "PEPEUSDT",
    "Aave": "AAVEUSDT",
    "Render": "RENDERUSDT",
}

INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]

# Working hosts (main api.binance.com is geo-blocked from many places)
BINANCE_HOSTS = [
    "https://data-api.binance.vision",
    "https://api.binance.us",
]

# ── Fetch ─────────────────────────────────────────────────────
def fetch_klines(symbol: str, interval: str, limit: int = 300) -> tuple[pd.DataFrame, str]:
    """Try working Binance hosts first. Return (df, source)."""
    params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
    errors = []

    for host in BINANCE_HOSTS:
        url = f"{host}/api/v3/klines"
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code != 200:
                errors.append(f"{host}: HTTP {r.status_code}")
                continue
            raw = r.json()
            if not isinstance(raw, list) or len(raw) == 0:
                errors.append(f"{host}: empty list")
                continue

            df = pd.DataFrame(raw, columns=[
                "t", "Open", "High", "Low", "Close", "Volume",
                "T", "q", "n", "V", "Q", "x"
            ])
            df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
            df = df.set_index("t")
            for c in ["Open", "High", "Low", "Close", "Volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if len(df) < 5:
                errors.append(f"{host}: too few rows")
                continue
            return df, host.replace("https://", "")
        except Exception as e:
            errors.append(f"{host}: {type(e).__name__}: {e}")

    # Yahoo fallback
    try:
        import yfinance as yf
        yf_sym = symbol.replace("USDT", "-USD")
        period = "1d" if interval in ("1m", "3m", "5m") else "7d"
        yf_int = interval if interval in ("1m", "5m", "15m", "30m", "1h", "1d") else "15m"
        df = yf.download(yf_sym, period=period, interval=yf_int, progress=False, auto_adjust=True)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if len(df) >= 5:
                return df, "Yahoo Finance (fallback)"
    except Exception as e:
        errors.append(f"yfinance: {e}")

    st.session_state["_fetch_errors"] = errors
    return pd.DataFrame(), "FAILED"


# ── Indicators ────────────────────────────────────────────────
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, min_periods=n).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, min_periods=n).mean()
    rs = g / l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 25:
        return pd.DataFrame()
    d = df.copy()
    c = d["Close"]
    d["EMA_fast"] = ema(c, 12)
    d["EMA_slow"] = ema(c, 26)
    d["EMA_200"] = ema(c, 200) if len(d) >= 200 else np.nan
    d["RSI"] = rsi(c)
    d["MACD"] = d["EMA_fast"] - d["EMA_slow"]
    d["MACD_sig"] = ema(d["MACD"], 9)
    d["MACD_hist"] = d["MACD"] - d["MACD_sig"]
    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    d["BB_u"] = mid + 2 * std
    d["BB_l"] = mid - 2 * std
    d["Vol_MA"] = d["Volume"].rolling(20).mean()
    d["Vol_ratio"] = d["Volume"] / d["Vol_MA"].replace(0, np.nan)

    score = pd.Series(0.0, index=d.index)
    score += np.where(d["EMA_fast"] > d["EMA_slow"], 25, -25)
    score += np.where(c > d["EMA_slow"], 10, -10)
    score += np.where(d["RSI"] < 30, 20, 0)
    score += np.where(d["RSI"] > 70, -20, 0)
    score += np.where(d["MACD"] > d["MACD_sig"], 15, -15)
    score += np.where(d["MACD_hist"] > d["MACD_hist"].shift(), 5, -5)
    d["Score"] = score.clip(-100, 100).round(0)
    d["Signal"] = d["Score"].map(lambda x: "BUY" if x >= 35 else ("SELL" if x <= -35 else "HOLD"))
    return d.dropna(subset=["RSI"])


# ── Chart ─────────────────────────────────────────────────────
def make_chart(df: pd.DataFrame, title: str) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=(title, "RSI (14)", "Volume"),
    )
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="OHLC",
        increasing_line_color="#00c853", decreasing_line_color="#ff1744",
        increasing_fillcolor="#00c853", decreasing_fillcolor="#ff1744",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["EMA_fast"], name="EMA 12",
                             line=dict(color="#ff006e", width=1.3, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA_slow"], name="EMA 26",
                             line=dict(color="#38b000", width=1.3, dash="dot")), row=1, col=1)
    if "BB_u" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_u"], name="BB Upper",
                                 line=dict(color="rgba(100,140,255,0.4)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_l"], name="BB Lower",
                                 line=dict(color="rgba(100,140,255,0.4)", width=1),
                                 fill="tonexty", fillcolor="rgba(100,140,255,0.07)"), row=1, col=1)

    buys = df[df["Signal"] == "BUY"].tail(10)
    sells = df[df["Signal"] == "SELL"].tail(10)
    if len(buys):
        fig.add_trace(go.Scatter(x=buys.index, y=buys["Low"]*0.997, mode="markers", name="BUY",
                                 marker=dict(symbol="triangle-up", size=10, color="#00c853")), row=1, col=1)
    if len(sells):
        fig.add_trace(go.Scatter(x=sells.index, y=sells["High"]*1.003, mode="markers", name="SELL",
                                 marker=dict(symbol="triangle-down", size=10, color="#ff1744")), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                             line=dict(color="#a855f7", width=1.4)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,23,68,0.5)", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,200,83,0.5)", row=2, col=1)

    colors = ["#00c853" if c >= o else "#ff1744" for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Vol", marker_color=colors, opacity=0.65), row=3, col=1)

    fig.update_layout(
        template="plotly_dark", height=680, margin=dict(l=6, r=6, t=30, b=6),
        xaxis_rangeslider_visible=False, hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        modebar_add=["drawline", "drawopenpath", "drawclosedpath", "drawcircle", "drawrect", "eraseshape"],
    )
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    return fig


# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("🛠️ Control Room")
name = st.sidebar.selectbox("Asset", list(CRYPTO.keys()), index=0)
symbol = CRYPTO[name]
custom = st.sidebar.text_input("Or type Binance symbol (BTCUSDT)", "")
if custom.strip():
    symbol = custom.strip().upper()
    name = symbol

interval = st.sidebar.selectbox("Candle size", INTERVALS, index=0)  # 1m default for live feel
limit = st.sidebar.select_slider("Candles", options=[100, 200, 300, 500, 1000], value=300)

st.sidebar.markdown("---")
st.sidebar.subheader("🔴 Live feed")
live = st.sidebar.checkbox("Auto-refresh ON", value=True)
secs = st.sidebar.slider("Seconds between updates", 5, 30, 10)
if st.sidebar.button("🔄 Force refresh", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption("Drawing: use chart modebar (top-right) → line / rect / circle / erase")

# ── Main ──────────────────────────────────────────────────────
now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
st.title("⚡ Crypto Signal Bot Pro")
st.markdown(f"<span class='live-dot'>● LIVE</span> &nbsp; last tick: **{now}** &nbsp;·&nbsp; {name} `{symbol}` `{interval}`",
            unsafe_allow_html=True)

# Always show a status box so the page is never pure white
status = st.empty()
status.info(f"Fetching {symbol} {interval} …")

df_raw, source = fetch_klines(symbol, interval, limit)
df = add_signals(df_raw)

if df.empty:
    status.error(
        f"**No chart data for {symbol}.**\n\n"
        f"Tried hosts: {BINANCE_HOSTS}\n\n"
        f"Errors: {st.session_state.get('_fetch_errors', ['unknown'])}"
    )
    st.warning(
        "If this keeps happening on Streamlit Cloud, run the app **on your own computer**:\n"
        "```\npip install streamlit requests pandas numpy plotly\nstreamlit run app.py\n```\n"
        "Binance blocks many cloud IPs. Local runs work."
    )
    st.stop()

status.success(f"Data OK · source: **{source}** · {len(df)} candles · {interval}")

latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest
delta = float(latest["Close"] - prev["Close"])

m1, m2, m3, m4, m5 = st.columns(5)
price = float(latest["Close"])
fmt = f"${price:,.6f}" if price < 1 else (f"${price:,.4f}" if price < 50 else f"${price:,.2f}")
m1.metric("Price", fmt, f"{delta:+.6f}" if abs(delta) < 1 else f"{delta:+.2f}")
sig = latest["Signal"]
cls = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}[sig]
m2.markdown("**Signal**")
m2.markdown(f"<div class='{cls}'>{sig}</div>", unsafe_allow_html=True)
m3.metric("RSI", f"{latest['RSI']:.1f}")
m4.metric("Score", f"{int(latest['Score'])}")
m5.metric("Source", source.split("/")[0] if "/" in source else source[:20])

st.divider()
st.subheader("📊 Live Candlestick Chart")
fig = make_chart(df, f"{name} · {interval} · live")
st.plotly_chart(fig, use_container_width=True, config={
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToAdd": ["drawline", "drawopenpath", "drawclosedpath",
                            "drawcircle", "drawrect", "eraseshape"],
})

st.subheader("Recent candles")
show = df[["Open", "High", "Low", "Close", "Volume", "RSI", "Score", "Signal"]].sort_index(ascending=False).head(50)
st.dataframe(show, use_container_width=True, height=300)

# Live loop — only after everything is drawn
if live:
    st.caption(f"Refreshing in {secs}s …")
    time.sleep(secs)
    st.rerun()
