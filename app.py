"""
Crypto Signal Bot Pro — chart guaranteed to show
================================================
- Binance data (working hosts)
- Native Streamlit charts FIRST (always visible)
- Plotly candlestick second (with hard-coded dark colors)
- Auto-refresh OFF by default so the chart can finish painting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
def fetch_klines(symbol: str, interval: str, limit: int = 300) -> tuple[pd.DataFrame, str]:
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


def build_plotly_candle(df: pd.DataFrame, title: str) -> go.Figure:
    x = list(df.index)
    o = df["Open"].astype(float).values
    h = df["High"].astype(float).values
    l = df["Low"].astype(float).values
    c = df["Close"].astype(float).values
    v = df["Volume"].astype(float).values

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.55, 0.22, 0.23],
        subplot_titles=(title, "RSI (14)", "Volume"),
    )

    fig.add_trace(go.Candlestick(
        x=x, open=o, high=h, low=l, close=c, name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=df["EMA_fast"].astype(float).values, name="EMA 12",
        line=dict(color="#ff6b6b", width=1.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=df["EMA_slow"].astype(float).values, name="EMA 26",
        line=dict(color="#51cf66", width=1.5),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=df["RSI"].astype(float).values, name="RSI",
        line=dict(color="#cc5de8", width=1.5),
    ), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#26a69a", row=2, col=1)

    colors = ["#26a69a" if cc >= oo else "#ef5350" for oo, cc in zip(o, c)]
    fig.add_trace(go.Bar(x=x, y=v, name="Vol", marker_color=colors, opacity=0.75), row=3, col=1)

    # HARD dark background so it is never white-on-white
    fig.update_layout(
        height=700,
        margin=dict(l=8, r=8, t=36, b=8),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right", bgcolor="rgba(0,0,0,0)"),
        dragmode="zoom",
        modebar_add=["drawline", "drawopenpath", "drawclosedpath",
                     "drawcircle", "drawrect", "eraseshape"],
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor="#222", row=i, col=1)
        fig.update_yaxes(gridcolor="#222", row=i, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    return fig


# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("🛠️ Control Room")
name = st.sidebar.selectbox("Asset", list(CRYPTO.keys()), index=0)
symbol = CRYPTO[name]
custom = st.sidebar.text_input("Custom symbol (e.g. BTCUSDT)", "")
if custom.strip():
    symbol = custom.strip().upper()
    name = symbol

interval = st.sidebar.selectbox("Candle size", INTERVALS, index=3)  # 15m default — more stable
limit = st.sidebar.select_slider("Candles", options=[100, 200, 300, 500], value=200)

st.sidebar.markdown("---")
st.sidebar.subheader("🔴 Live feed")
# OFF by default so first paint completes
live = st.sidebar.checkbox("Auto-refresh ON", value=False)
secs = st.sidebar.slider("Seconds between updates", 8, 60, 15)
if st.sidebar.button("🔄 Refresh now", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.info("Turn Auto-refresh ON only AFTER you see the chart.")

# ── Main ──────────────────────────────────────────────────────
st.title("⚡ Crypto Signal Bot Pro")
st.caption(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC · {name} · {symbol} · {interval}")

with st.spinner("Loading market data…"):
    raw, source = fetch_klines(symbol, interval, limit)
    df = add_signals(raw)

if df.empty:
    st.error("No data returned. Try another interval or symbol.")
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

# ══════════════════════════════════════════════════════════════
# CHART SECTION — three layers so something ALWAYS shows
# ══════════════════════════════════════════════════════════════

st.subheader("1) Price (always visible — native Streamlit)")
# This NEVER fails to render
chart_df = df[["Close", "EMA_fast", "EMA_slow"]].copy()
chart_df.columns = ["Close", "EMA 12", "EMA 26"]
st.line_chart(chart_df, height=320, use_container_width=True)

st.subheader("2) Volume (always visible)")
st.bar_chart(df[["Volume"]], height=180, use_container_width=True)

st.subheader("3) Full Candlestick + RSI (Plotly)")
try:
    fig = build_plotly_candle(df, f"{name} · {interval}")
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
    st.warning(f"Plotly candlestick could not render: {e}")
    st.info("Use the line chart above — data is live.")

st.subheader("Recent data")
st.dataframe(
    df[["Open", "High", "Low", "Close", "Volume", "RSI", "Score", "Signal"]]
      .sort_index(ascending=False)
      .head(30),
    use_container_width=True,
    height=260,
)

# Auto-refresh only if user turned it on
if live:
    st.caption(f"Auto-refresh in {secs}s…")
    time.sleep(secs)
    st.rerun()
