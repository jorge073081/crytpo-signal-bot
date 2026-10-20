"""
Crypto Signal Bot Pro — Near-Realtime Edition
=============================================
Uses Binance public REST API (no API key) for fresh candles.
Auto-refreshes so the chart updates like a live feed.

True tick-by-tick WebSocket is possible but fragile on Streamlit Cloud
(Binance often blocks cloud IPs). This version uses fast polling (5–30 s)
which is the most reliable “realtime” approach for Streamlit + GitHub deploy.

requirements.txt:
    streamlit
    requests
    pandas
    numpy
    plotly
    yfinance   # fallback only

Run locally:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone
import requests
import time

# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crypto Signal Bot Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stMetric { background: rgba(255,255,255,0.05); padding: 10px 14px; border-radius: 10px; }
    div[data-testid="stMetricValue"] { font-size: 1.45rem; }
    .signal-buy  { color: #00c853; font-weight: 800; font-size: 1.8rem; }
    .signal-sell { color: #ff1744; font-weight: 800; font-size: 1.8rem; }
    .signal-hold { color: #ffab00; font-weight: 800; font-size: 1.8rem; }
    .live-dot { color: #00c853; animation: pulse 1.2s infinite; }
    @keyframes pulse { 0%{opacity:1} 50%{opacity:0.35} 100%{opacity:1} }
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Ticker map  (display name → Binance symbol)
# ──────────────────────────────────────────────────────────────
CRYPTO_TICKERS = {
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
    "Polygon": "MATICUSDT",
    "Litecoin": "LTCUSDT",
    "Shiba Inu": "SHIBUSDT",
    "Uniswap": "UNIUSDT",
    "Cosmos": "ATOMUSDT",
    "Stellar": "XLMUSDT",
    "Bitcoin Cash": "BCHUSDT",
    "Near Protocol": "NEARUSDT",
    "Aptos": "APTUSDT",
    "Arbitrum": "ARBUSDT",
    "Optimism": "OPUSDT",
    "Sui": "SUIUSDT",
    "Toncoin": "TONUSDT",
    "Tron": "TRXUSDT",
    "Filecoin": "FILUSDT",
    "Internet Computer": "ICPUSDT",
    "Hedera": "HBARUSDT",
    "Algorand": "ALGOUSDT",
    "VeChain": "VETUSDT",
    "Injective": "INJUSDT",
    "Render": "RENDERUSDT",
    "Fetch.ai": "FETUSDT",
    "Pepe": "PEPEUSDT",
    "Worldcoin": "WLDUSDT",
    "Monero": "XMRUSDT",
    "Aave": "AAVEUSDT",
    "Maker": "MKRUSDT",
    "The Graph": "GRTUSDT",
    "Immutable": "IMXUSDT",
    "Kaspa": "KASUSDT",
}

INTERVAL_MAP = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

LOOKBACK_LIMITS = {
    "Last 100 candles": 100,
    "Last 200 candles": 200,
    "Last 500 candles": 500,
    "Last 1000 candles": 1000,
}

# ──────────────────────────────────────────────────────────────
# Data layer — Binance first, yfinance fallback
# ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=15, show_spinner=False)
def fetch_binance_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """Fetch OHLCV from Binance public API (no key)."""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
    try:
        r = requests.get(url, params=params, timeout=12)
        if r.status_code == 451:
            r = requests.get(
                "https://data-api.binance.vision/api/v3/klines",
                params=params,
                timeout=12,
            )
        r.raise_for_status()
        raw = r.json()
        if not raw or not isinstance(raw, list):
            return pd.DataFrame()

        df = pd.DataFrame(raw, columns=[
            "Open_time", "Open", "High", "Low", "Close", "Volume",
            "Close_time", "Quote_vol", "Trades", "Taker_buy_base",
            "Taker_buy_quote", "Ignore",
        ])
        df["Open_time"] = pd.to_datetime(df["Open_time"], unit="ms", utc=True)
        df.set_index("Open_time", inplace=True)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception as e:
        st.session_state["_last_fetch_error"] = str(e)
        return pd.DataFrame()


@st.cache_data(ttl=30, show_spinner=False)
def fetch_yfinance_fallback(symbol_yf: str, period: str, interval: str) -> pd.DataFrame:
    """Last-resort fallback if Binance is blocked."""
    try:
        import yfinance as yf
        df = yf.download(symbol_yf, period=period, interval=interval,
                         progress=False, auto_adjust=True, threads=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return pd.DataFrame()


def get_data(binance_symbol: str, interval: str, limit: int):
    """Return (dataframe, source_label)."""
    df = fetch_binance_klines(binance_symbol, interval, limit)
    if not df.empty:
        return df, "Binance (near-realtime)"
    yf_symbol = binance_symbol.replace("USDT", "-USD")
    period = "7d" if interval in ("1m", "3m", "5m", "15m") else "30d"
    yf_interval = interval if interval != "1m" else "5m"
    df = fetch_yfinance_fallback(yf_symbol, period, yf_interval)
    if not df.empty:
        return df, "Yahoo Finance (fallback – delayed)"
    return pd.DataFrame(), "None"


# ──────────────────────────────────────────────────────────────
# Indicators + Signal
# ──────────────────────────────────────────────────────────────
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(s: pd.Series, length: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 30:
        return pd.DataFrame()

    data = df.copy()
    close = data["Close"]

    data["EMA_fast"] = _ema(close, 12)
    data["EMA_slow"] = _ema(close, 26)
    data["EMA_200"] = _ema(close, 200) if len(data) >= 200 else np.nan
    data["RSI"] = _rsi(close, 14)

    ema_f = data["EMA_fast"]
    ema_s = data["EMA_slow"]
    macd = ema_f - ema_s
    data["MACD"] = macd
    data["MACD_signal"] = _ema(macd, 9)
    data["MACD_hist"] = data["MACD"] - data["MACD_signal"]

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    data["BB_upper"] = mid + 2 * std
    data["BB_mid"] = mid
    data["BB_lower"] = mid - 2 * std

    data["Vol_MA"] = data["Volume"].rolling(20).mean()
    data["Vol_ratio"] = data["Volume"] / data["Vol_MA"].replace(0, np.nan)

    tr = pd.concat([
        data["High"] - data["Low"],
        (data["High"] - close.shift()).abs(),
        (data["Low"] - close.shift()).abs(),
    ], axis=1).max(axis=1)
    data["ATR"] = tr.rolling(14).mean()

    score = pd.Series(0.0, index=data.index)
    score += np.where(data["EMA_fast"] > data["EMA_slow"], 25, -25)
    score += np.where(close > data["EMA_slow"], 10, -10)
    score += np.where(data["RSI"] < 30, 20, 0)
    score += np.where(data["RSI"] > 70, -20, 0)
    score += np.where(data["MACD"] > data["MACD_signal"], 15, -15)
    score += np.where(data["MACD_hist"] > data["MACD_hist"].shift(1), 5, -5)
    bb_pos = (close - data["BB_lower"]) / (data["BB_upper"] - data["BB_lower"]).replace(0, np.nan)
    score += np.where(bb_pos < 0.2, 10, 0)
    score += np.where(bb_pos > 0.8, -10, 0)
    score += np.where(data["Vol_ratio"] > 1.5, 8, 0) * np.sign(score)

    data["Score"] = score.clip(-100, 100).round(0)
    data["Signal"] = data["Score"].apply(
        lambda s: "BUY" if s >= 35 else ("SELL" if s <= -35 else "HOLD")
    )
    data["Price_Change_Pct"] = close.pct_change() * 100
    return data.dropna(subset=["RSI", "EMA_fast"])


# ──────────────────────────────────────────────────────────────
# Chart
# ──────────────────────────────────────────────────────────────
def build_chart(data: pd.DataFrame, title: str, show_volume: bool = True) -> go.Figure:
    rows = 3 if show_volume else 2
    heights = [0.55, 0.25, 0.20] if show_volume else [0.65, 0.35]

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=heights,
        subplot_titles=(f"{title}", "RSI (14)", "Volume") if show_volume else (f"{title}", "RSI (14)"),
    )

    fig.add_trace(go.Candlestick(
        x=data.index, open=data["Open"], high=data["High"],
        low=data["Low"], close=data["Close"], name="OHLC",
        increasing_line_color="#00c853", decreasing_line_color="#ff1744",
        increasing_fillcolor="#00c853", decreasing_fillcolor="#ff1744",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=data.index, y=data["EMA_fast"], name="EMA 12",
        line=dict(color="#ff006e", width=1.4, dash="dash"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=data["EMA_slow"], name="EMA 26",
        line=dict(color="#38b000", width=1.4, dash="dot"),
    ), row=1, col=1)
    if data["EMA_200"].notna().any():
        fig.add_trace(go.Scatter(
            x=data.index, y=data["EMA_200"], name="EMA 200",
            line=dict(color="#ffd60a", width=1.2),
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=data.index, y=data["BB_upper"], name="BB Upper",
        line=dict(color="rgba(100,140,255,0.45)", width=1),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=data["BB_lower"], name="BB Lower",
        line=dict(color="rgba(100,140,255,0.45)", width=1),
        fill="tonexty", fillcolor="rgba(100,140,255,0.08)",
    ), row=1, col=1)

    buys = data[data["Signal"] == "BUY"].tail(12)
    sells = data[data["Signal"] == "SELL"].tail(12)
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys.index, y=buys["Low"] * 0.997, mode="markers", name="BUY",
            marker=dict(symbol="triangle-up", size=11, color="#00c853",
                        line=dict(width=1, color="white")),
        ), row=1, col=1)
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells.index, y=sells["High"] * 1.003, mode="markers", name="SELL",
            marker=dict(symbol="triangle-down", size=11, color="#ff1744",
                        line=dict(width=1, color="white")),
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=data.index, y=data["RSI"], name="RSI",
        line=dict(color="#a855f7", width=1.5),
    ), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,23,68,0.55)", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,200,83,0.55)", row=2, col=1)

    if show_volume:
        colors = ["#00c853" if c >= o else "#ff1744"
                  for o, c in zip(data["Open"], data["Close"])]
        fig.add_trace(go.Bar(
            x=data.index, y=data["Volume"], name="Volume",
            marker_color=colors, opacity=0.7,
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=data.index, y=data["Vol_MA"], name="Vol MA(20)",
            line=dict(color="#ffd60a", width=1.2),
        ), row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=700,
        margin=dict(l=8, r=8, t=36, b=8),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        dragmode="zoom",
        modebar_add=["drawline", "drawopenpath", "drawclosedpath",
                     "drawcircle", "drawrect", "eraseshape"],
    )
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    return fig


# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
st.sidebar.header("🛠️ Control Room")

selected_name = st.sidebar.selectbox("Select Crypto Asset", list(CRYPTO_TICKERS.keys()), index=0)
binance_symbol = CRYPTO_TICKERS[selected_name]

custom = st.sidebar.text_input("Custom Binance symbol (e.g. BTCUSDT)", "")
if custom.strip():
    binance_symbol = custom.strip().upper()
    selected_name = binance_symbol

interval_label = st.sidebar.selectbox(
    "Candle Interval (realtime friendly)",
    list(INTERVAL_MAP.keys()),
    index=3,  # 15m default
)
interval = INTERVAL_MAP[interval_label]

lookback_label = st.sidebar.selectbox("How many candles?", list(LOOKBACK_LIMITS.keys()), index=2)
limit = LOOKBACK_LIMITS[lookback_label]

show_volume = st.sidebar.checkbox("Show Volume Panel", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🔴 Live Feed")
live_mode = st.sidebar.checkbox("Enable auto-refresh (near-realtime)", value=True)
refresh_sec = st.sidebar.slider("Refresh every (seconds)", 5, 60, 15, 5)
manual_refresh = st.sidebar.button("🔄 Refresh Now", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Chart supports drawing: use the modebar (top-right of chart) → "
    "line / path / rect / circle / erase."
)
st.sidebar.caption(
    "Data: Binance public API (no key). "
    "True tick WebSocket is blocked on many cloud hosts — this uses fast polling."
)

# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
st.title("⚡ Crypto Signal Bot Pro")
st.markdown(
    f"<span class='live-dot'>●</span> Near-realtime feed &nbsp;|&nbsp; "
    f"Last update: **{now_str}**",
    unsafe_allow_html=True,
)

if manual_refresh:
    st.cache_data.clear()

with st.spinner(f"Loading {selected_name} ({binance_symbol}) {interval} …"):
    raw, source = get_data(binance_symbol, interval, limit)
    processed = generate_signals(raw)

if processed.empty:
    st.error(
        f"**No data for {binance_symbol} ({interval}).**\n\n"
        "Possible causes:\n"
        "1. Binance blocked this host (common on Streamlit Cloud) → run the app **locally**\n"
        "2. Invalid symbol (must be like `BTCUSDT`, not `BTC-USD`)\n"
        "3. Temporary network issue\n\n"
        f"Last error: `{st.session_state.get('_last_fetch_error', 'none')}`"
    )
    st.info(
        "Tip: On Streamlit Cloud, Binance sometimes returns HTTP 451. "
        "Clone the repo and run `streamlit run app.py` on your machine for a true live feed."
    )
    st.stop()

latest = processed.iloc[-1]
prior = processed.iloc[-2] if len(processed) > 1 else latest

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    delta = latest["Close"] - prior["Close"]
    price_fmt = f"${latest['Close']:,.6f}" if latest["Close"] < 1 else (
        f"${latest['Close']:,.4f}" if latest["Close"] < 100 else f"${latest['Close']:,.2f}"
    )
    st.metric("Live Price", price_fmt, f"{delta:+.6f}" if abs(delta) < 1 else f"{delta:+.2f}")
with m2:
    sig = latest["Signal"]
    cls = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}[sig]
    st.markdown("**Signal**")
    st.markdown(f"<div class='{cls}'>{sig}</div>", unsafe_allow_html=True)
with m3:
    st.metric("RSI (14)", f"{latest['RSI']:.1f}")
with m4:
    st.metric("Momentum Score", f"{int(latest['Score'])}")
with m5:
    st.metric("Source", source.split("(")[0].strip())

c1, c2, c3, c4 = st.columns(4)
c1.metric("MACD Hist", f"{latest['MACD_hist']:.5f}")
c2.metric("ATR (14)", f"{latest['ATR']:.5f}" if pd.notna(latest.get("ATR")) else "—")
c3.metric("Last Candle %", f"{latest.get('Price_Change_Pct', 0):+.2f}%")
c4.metric("Vol vs MA", f"{latest.get('Vol_ratio', 0):.2f}×")

st.caption(f"Data source: **{source}** · Candles: {len(processed)} · Interval: {interval}")

st.divider()
st.subheader("📊 Live Candlestick Chart (draw tools enabled)")
fig = build_chart(processed, f"{selected_name} · {interval}", show_volume=show_volume)
st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": True,
        "scrollZoom": True,
        "displaylogo": False,
        "modeBarButtonsToAdd": [
            "drawline", "drawopenpath", "drawclosedpath",
            "drawcircle", "drawrect", "eraseshape",
        ],
    },
)

with st.expander("How the score works"):
    st.markdown("""
    Score range ≈ **−100 … +100**
    - EMA trend + price vs slow EMA → ±35
    - RSI oversold/overbought → ±20
    - MACD line & histogram → ±20
    - Bollinger position → ±10
    - Volume confirmation → ±8

    **BUY** ≥ +35 · **SELL** ≤ −35 · otherwise **HOLD**
    """)

st.subheader("📋 Recent Signal Log")
cols = ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD_hist",
        "EMA_fast", "EMA_slow", "Score", "Signal", "Vol_ratio"]
avail = [c for c in cols if c in processed.columns]
st.dataframe(
    processed[avail].sort_index(ascending=False).head(80).style.format({
        "Open": "{:.6f}", "High": "{:.6f}", "Low": "{:.6f}", "Close": "{:.6f}",
        "RSI": "{:.1f}", "MACD_hist": "{:.5f}",
        "EMA_fast": "{:.6f}", "EMA_slow": "{:.6f}",
        "Score": "{:.0f}", "Vol_ratio": "{:.2f}", "Volume": "{:,.0f}",
    }),
    use_container_width=True,
    height=340,
)

# ── Near-realtime loop ────────────────────────────────────────
if live_mode:
    st.caption(f"Auto-refreshing every {refresh_sec}s …")
    time.sleep(refresh_sec)
    st.rerun()
