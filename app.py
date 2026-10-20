"""
Crypto Signal Bot Pro — live chart (fixed render)
=================================================
Data from data-api.binance.vision / api.binance.us
Chart forced to render with defensive Plotly handling.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone
import requests
import time
import traceback

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
</style>
""", unsafe_allow_html=True)

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
BINANCE_HOSTS = [
    "https://data-api.binance.vision",
    "https://api.binance.us",
]


def fetch_klines(symbol: str, interval: str, limit: int = 300):
    params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
    errors = []
    for host in BINANCE_HOSTS:
        try:
            r = requests.get(f"{host}/api/v3/klines", params=params, timeout=10)
            if r.status_code != 200:
                errors.append(f"{host}: HTTP {r.status_code}")
                continue
            raw = r.json()
            if not isinstance(raw, list) or len(raw) < 5:
                errors.append(f"{host}: bad payload")
                continue
            df = pd.DataFrame(raw, columns=[
                "t", "Open", "High", "Low", "Close", "Volume",
                "T", "q", "n", "V", "Q", "x"
            ])
            # CRITICAL for Plotly: use timezone-naive timestamps
            df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_localize(None)
            df = df.set_index("t")
            for c in ["Open", "High", "Low", "Close", "Volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if len(df) >= 5:
                return df, host.replace("https://", "")
        except Exception as e:
            errors.append(f"{host}: {e}")
    st.session_state["_fetch_errors"] = errors
    return pd.DataFrame(), "FAILED"


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
    d["RSI"] = rsi(c)
    d["MACD"] = d["EMA_fast"] - d["EMA_slow"]
    d["MACD_sig"] = ema(d["MACD"], 9)
    d["MACD_hist"] = d["MACD"] - d["MACD_sig"]
    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    d["BB_u"] = mid + 2 * std
    d["BB_l"] = mid - 2 * std
    d["Vol_MA"] = d["Volume"].rolling(20).mean()

    score = pd.Series(0.0, index=d.index)
    score += np.where(d["EMA_fast"] > d["EMA_slow"], 25, -25)
    score += np.where(c > d["EMA_slow"], 10, -10)
    score += np.where(d["RSI"] < 30, 20, 0)
    score += np.where(d["RSI"] > 70, -20, 0)
    score += np.where(d["MACD"] > d["MACD_sig"], 15, -15)
    d["Score"] = score.clip(-100, 100).round(0)
    d["Signal"] = d["Score"].map(lambda x: "BUY" if x >= 35 else ("SELL" if x <= -35 else "HOLD"))
    return d.dropna(subset=["RSI"])


def make_chart(df: pd.DataFrame, title: str) -> go.Figure:
    """Build candlestick chart. Raises on failure so caller can show error."""
    # Ensure clean numeric arrays (Plotly is picky)
    x = df.index.tolist()
    o = df["Open"].astype(float).tolist()
    h = df["High"].astype(float).tolist()
    l = df["Low"].astype(float).tolist()
    c = df["Close"].astype(float).tolist()
    v = df["Volume"].astype(float).tolist()
    rsi_vals = df["RSI"].astype(float).tolist()
    ema_f = df["EMA_fast"].astype(float).tolist()
    ema_s = df["EMA_slow"].astype(float).tolist()

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=(title, "RSI (14)", "Volume"),
    )

    fig.add_trace(go.Candlestick(
        x=x, open=o, high=h, low=l, close=c,
        name="OHLC",
        increasing_line_color="#00c853",
        decreasing_line_color="#ff1744",
        increasing_fillcolor="#00c853",
        decreasing_fillcolor="#ff1744",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=ema_f, name="EMA 12",
        line=dict(color="#ff006e", width=1.4, dash="dash"),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=ema_s, name="EMA 26",
        line=dict(color="#38b000", width=1.4, dash="dot"),
    ), row=1, col=1)

    if "BB_u" in df.columns and df["BB_u"].notna().any():
        fig.add_trace(go.Scatter(
            x=x, y=df["BB_u"].astype(float).tolist(), name="BB Upper",
            line=dict(color="rgba(100,140,255,0.45)", width=1),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=df["BB_l"].astype(float).tolist(), name="BB Lower",
            line=dict(color="rgba(100,140,255,0.45)", width=1),
            fill="tonexty", fillcolor="rgba(100,140,255,0.08)",
        ), row=1, col=1)

    # Signal markers
    buys = df[df["Signal"] == "BUY"].tail(8)
    sells = df[df["Signal"] == "SELL"].tail(8)
    if len(buys) > 0:
        fig.add_trace(go.Scatter(
            x=buys.index.tolist(),
            y=(buys["Low"] * 0.997).astype(float).tolist(),
            mode="markers", name="BUY",
            marker=dict(symbol="triangle-up", size=11, color="#00c853"),
        ), row=1, col=1)
    if len(sells) > 0:
        fig.add_trace(go.Scatter(
            x=sells.index.tolist(),
            y=(sells["High"] * 1.003).astype(float).tolist(),
            mode="markers", name="SELL",
            marker=dict(symbol="triangle-down", size=11, color="#ff1744"),
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=rsi_vals, name="RSI",
        line=dict(color="#a855f7", width=1.5),
    ), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,23,68,0.55)", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,200,83,0.55)", row=2, col=1)

    colors = ["#00c853" if cc >= oo else "#ff1744" for oo, cc in zip(o, c)]
    fig.add_trace(go.Bar(
        x=x, y=v, name="Volume",
        marker_color=colors, opacity=0.7,
    ), row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=720,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        dragmode="zoom",
        # Drawing tools
        newshape=dict(line_color="cyan"),
        modebar_add=[
            "drawline", "drawopenpath", "drawclosedpath",
            "drawcircle", "drawrect", "eraseshape",
        ],
    )
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
    return fig


def make_simple_line_chart(df: pd.DataFrame, title: str) -> go.Figure:
    """Ultra-simple fallback if candlestick fails."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index.tolist(),
        y=df["Close"].astype(float).tolist(),
        mode="lines",
        name="Close",
        line=dict(color="#3a86ff", width=2),
    ))
    fig.update_layout(
        template="plotly_dark",
        title=title + " (simple fallback)",
        height=500,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False,
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

interval = st.sidebar.selectbox("Candle size", INTERVALS, index=0)
limit = st.sidebar.select_slider("Candles", options=[100, 200, 300, 500, 1000], value=300)

st.sidebar.markdown("---")
st.sidebar.subheader("🔴 Live feed")
live = st.sidebar.checkbox("Auto-refresh ON", value=True)
secs = st.sidebar.slider("Seconds between updates", 5, 30, 10)
if st.sidebar.button("🔄 Force refresh", use_container_width=True):
    st.rerun()

st.sidebar.caption("Draw on chart: modebar (top-right) → line / rect / circle / erase")

# ── Main ──────────────────────────────────────────────────────
now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
st.title("⚡ Crypto Signal Bot Pro")
st.markdown(
    f"<span class='live-dot'>● LIVE</span> &nbsp; {now} &nbsp;·&nbsp; {name} `{symbol}` `{interval}`",
    unsafe_allow_html=True,
)

status = st.empty()
status.info(f"Fetching {symbol} {interval} …")

df_raw, source = fetch_klines(symbol, interval, limit)
df = add_signals(df_raw)

if df.empty:
    status.error(f"No data. Errors: {st.session_state.get('_fetch_errors', [])}")
    st.stop()

status.success(f"Data OK · source: **{source}** · {len(df)} candles · {interval}")

latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest
delta = float(latest["Close"] - prev["Close"])
price = float(latest["Close"])
fmt = f"${price:,.6f}" if price < 1 else (f"${price:,.4f}" if price < 50 else f"${price:,.2f}")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Price", fmt, f"{delta:+.6f}" if abs(delta) < 1 else f"{delta:+.2f}")
sig = str(latest["Signal"])
cls = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}.get(sig, "signal-hold")
m2.markdown("**Signal**")
m2.markdown(f"<div class='{cls}'>{sig}</div>", unsafe_allow_html=True)
m3.metric("RSI", f"{float(latest['RSI']):.1f}")
m4.metric("Score", f"{int(latest['Score'])}")
m5.metric("Source", source[:22])

st.divider()
st.subheader("📊 Live Candlestick Chart")

# ── FORCE CHART TO RENDER ─────────────────────────────────────
chart_slot = st.empty()
try:
    fig = make_chart(df, f"{name} · {interval}")
    chart_slot.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "responsive": True,
            "modeBarButtonsToAdd": [
                "drawline", "drawopenpath", "drawclosedpath",
                "drawcircle", "drawrect", "eraseshape",
            ],
        },
        key=f"chart_{symbol}_{interval}_{len(df)}",  # force re-mount on data change
    )
except Exception as e:
    st.error("Candlestick failed — showing simple line chart instead.")
    st.code(traceback.format_exc())
    try:
        fig2 = make_simple_line_chart(df, f"{name} · {interval}")
        st.plotly_chart(fig2, use_container_width=True, key="fallback_chart")
    except Exception as e2:
        st.error(f"Even simple chart failed: {e2}")
        st.write("Raw last 5 rows:")
        st.dataframe(df.tail())

st.subheader("Recent candles")
st.dataframe(
    df[["Open", "High", "Low", "Close", "Volume", "RSI", "Score", "Signal"]]
      .sort_index(ascending=False)
      .head(40),
    use_container_width=True,
    height=280,
)

if live:
    st.caption(f"Next refresh in {secs}s …")
    time.sleep(secs)
    st.rerun()
