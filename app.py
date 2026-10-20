"""
Crypto Signal Bot — Advanced Streamlit Dashboard
================================================
Self-contained: no external signals.py required.
Uses yfinance for data, Plotly for interactive candlestick charts
(with drawing tools via modebar), and a rich multi-indicator signal engine.

Requirements (add to requirements.txt):
    streamlit
    yfinance
    plotly
    pandas
    numpy
    ta   # optional but recommended for cleaner indicators; fallback included

Run:
    streamlit run app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

# Optional: cleaner indicators if `ta` is installed
try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

# ──────────────────────────────────────────────────────────────
# Page & Theme
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crypto Signal Bot Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a tighter, more professional look
st.markdown("""
<style>
    .stMetric { background: rgba(255,255,255,0.04); padding: 12px 16px; border-radius: 10px; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    .signal-buy  { color: #00c853; font-weight: 700; }
    .signal-sell { color: #ff1744; font-weight: 700; }
    .signal-hold { color: #ffab00; font-weight: 700; }
    .block-container { padding-top: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Large curated crypto ticker list (yfinance format)
# ──────────────────────────────────────────────────────────────
CRYPTO_TICKERS = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana": "SOL-USD",
    "BNB": "BNB-USD",
    "XRP": "XRP-USD",
    "Cardano": "ADA-USD",
    "Dogecoin": "DOGE-USD",
    "Avalanche": "AVAX-USD",
    "Chainlink": "LINK-USD",
    "Polkadot": "DOT-USD",
    "Polygon": "MATIC-USD",
    "Litecoin": "LTC-USD",
    "Shiba Inu": "SHIB-USD",
    "Uniswap": "UNI-USD",
    "Cosmos": "ATOM-USD",
    "Stellar": "XLM-USD",
    "Bitcoin Cash": "BCH-USD",
    "Near Protocol": "NEAR-USD",
    "Aptos": "APT-USD",
    "Arbitrum": "ARB-USD",
    "Optimism": "OP-USD",
    "Sui": "SUI-USD",
    "Toncoin": "TON-USD",
    "Tron": "TRX-USD",
    "Filecoin": "FIL-USD",
    "Internet Computer": "ICP-USD",
    "Hedera": "HBAR-USD",
    "Algorand": "ALGO-USD",
    "VeChain": "VET-USD",
    "Injective": "INJ-USD",
    "Render": "RENDER-USD",
    "Fetch.ai": "FET-USD",
    "Pepe": "PEPE-USD",
    "Worldcoin": "WLD-USD",
    " monero": "XMR-USD",
    "Aave": "AAVE-USD",
    "Maker": "MKR-USD",
    "The Graph": "GRT-USD",
    "Immutable": "IMX-USD",
    "Kaspa": "KAS-USD",
}

# Clean display names (remove accidental space)
CRYPTO_TICKERS = {k.strip(): v for k, v in CRYPTO_TICKERS.items()}

# ──────────────────────────────────────────────────────────────
# Data + Signal Engine
# ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def fetch_crypto_data(ticker: str, period: str = "30d", interval: str = "1h") -> pd.DataFrame:
    """Download OHLCV from Yahoo Finance."""
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        if df.empty:
            return pd.DataFrame()
        # Flatten multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(how="all")
        return df
    except Exception as e:
        st.warning(f"Data fetch error for {ticker}: {e}")
        return pd.DataFrame()


def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _bollinger(series: pd.Series, length=20, std=2.0):
    mid = series.rolling(length).mean()
    dev = series.rolling(length).std()
    upper = mid + std * dev
    lower = mid - std * dev
    return upper, mid, lower


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich OHLCV with technical indicators and a composite momentum score + discrete signal.
    Score roughly -100 … +100. Signal: BUY / SELL / HOLD.
    """
    if df.empty or len(df) < 30:
        return pd.DataFrame()

    data = df.copy()
    close = data["Close"]

    # Core indicators
    data["EMA_fast"] = _ema(close, 12)
    data["EMA_slow"] = _ema(close, 26)
    data["EMA_200"] = _ema(close, 200) if len(data) >= 200 else np.nan
    data["RSI"] = _rsi(close, 14)
    data["MACD"], data["MACD_signal"], data["MACD_hist"] = _macd(close)
    data["BB_upper"], data["BB_mid"], data["BB_lower"] = _bollinger(close)

    # Volume metrics
    data["Vol_MA"] = data["Volume"].rolling(20).mean()
    data["Vol_ratio"] = data["Volume"] / data["Vol_MA"].replace(0, np.nan)

    # ATR (simple true range average)
    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift()).abs()
    low_close = (data["Low"] - data["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data["ATR"] = tr.rolling(14).mean()

    # ── Composite Score ───────────────────────────────────────
    score = pd.Series(0.0, index=data.index)

    # Trend (EMA crossover + distance)
    score += np.where(data["EMA_fast"] > data["EMA_slow"], 25, -25)
    score += np.where(close > data["EMA_slow"], 10, -10)

    # RSI zones
    score += np.where(data["RSI"] < 30, 20, 0)          # oversold → bullish
    score += np.where(data["RSI"] > 70, -20, 0)         # overbought → bearish
    score += np.where((data["RSI"] >= 40) & (data["RSI"] <= 60), 5, 0)

    # MACD
    score += np.where(data["MACD"] > data["MACD_signal"], 15, -15)
    score += np.where(data["MACD_hist"] > data["MACD_hist"].shift(1), 5, -5)

    # Bollinger position
    bb_pos = (close - data["BB_lower"]) / (data["BB_upper"] - data["BB_lower"]).replace(0, np.nan)
    score += np.where(bb_pos < 0.2, 10, 0)
    score += np.where(bb_pos > 0.8, -10, 0)

    # Volume confirmation
    score += np.where(data["Vol_ratio"] > 1.5, 8, 0) * np.sign(score)

    data["Score"] = score.clip(-100, 100).round(0)

    # Discrete signal
    def classify(s):
        if s >= 35:
            return "BUY"
        elif s <= -35:
            return "SELL"
        return "HOLD"

    data["Signal"] = data["Score"].apply(classify)

    # Convenience columns for display
    data["Price_Change_Pct"] = close.pct_change() * 100

    return data.dropna(subset=["RSI", "EMA_fast"])


# ──────────────────────────────────────────────────────────────
# Chart Builder (Candles + Volume + Indicators + Drawing tools)
# ──────────────────────────────────────────────────────────────
def build_candlestick_chart(data: pd.DataFrame, ticker: str, show_volume: bool = True) -> go.Figure:
    rows = 3 if show_volume else 2
    row_heights = [0.55, 0.25, 0.20] if show_volume else [0.65, 0.35]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=(f"{ticker} Price", "RSI (14)", "Volume") if show_volume else (f"{ticker} Price", "RSI (14)"),
    )

    # ── Candlesticks ──────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="OHLC",
            increasing_line_color="#00c853",
            decreasing_line_color="#ff1744",
            increasing_fillcolor="#00c853",
            decreasing_fillcolor="#ff1744",
        ),
        row=1, col=1,
    )

    # EMAs
    fig.add_trace(
        go.Scatter(x=data.index, y=data["EMA_fast"], name="EMA 12",
                   line=dict(color="#ff006e", width=1.4, dash="dash")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data["EMA_slow"], name="EMA 26",
                   line=dict(color="#38b000", width=1.4, dash="dot")),
        row=1, col=1,
    )
    if "EMA_200" in data.columns and data["EMA_200"].notna().any():
        fig.add_trace(
            go.Scatter(x=data.index, y=data["EMA_200"], name="EMA 200",
                       line=dict(color="#ffd60a", width=1.2)),
            row=1, col=1,
        )

    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=data.index, y=data["BB_upper"], name="BB Upper",
                   line=dict(color="rgba(100,100,255,0.4)", width=1), showlegend=True),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data["BB_lower"], name="BB Lower",
                   line=dict(color="rgba(100,100,255,0.4)", width=1),
                   fill="tonexty", fillcolor="rgba(100,100,255,0.08)"),
        row=1, col=1,
    )

    # Signal markers (last few BUY / SELL)
    buys = data[data["Signal"] == "BUY"].tail(15)
    sells = data[data["Signal"] == "SELL"].tail(15)
    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=buys.index, y=buys["Low"] * 0.995,
                mode="markers", name="BUY",
                marker=dict(symbol="triangle-up", size=11, color="#00c853", line=dict(width=1, color="white")),
            ),
            row=1, col=1,
        )
    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells.index, y=sells["High"] * 1.005,
                mode="markers", name="SELL",
                marker=dict(symbol="triangle-down", size=11, color="#ff1744", line=dict(width=1, color="white")),
            ),
            row=1, col=1,
        )

    # ── RSI ───────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(x=data.index, y=data["RSI"], name="RSI",
                   line=dict(color="#a855f7", width=1.5)),
        row=2, col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,23,68,0.6)", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,200,83,0.6)", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=2, col=1)

    # ── Volume ────────────────────────────────────────────────
    if show_volume:
        colors = ["#00c853" if c >= o else "#ff1744" for o, c in zip(data["Open"], data["Close"])]
        fig.add_trace(
            go.Bar(x=data.index, y=data["Volume"], name="Volume",
                   marker_color=colors, opacity=0.7),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data["Vol_MA"], name="Vol MA(20)",
                       line=dict(color="#ffd60a", width=1.2)),
            row=3, col=1,
        )

    # Layout
    fig.update_layout(
        template="plotly_dark",
        height=720,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        dragmode="zoom",          # default interaction
        # Enable Plotly modebar drawing tools (annotations, shapes, etc.)
        modebar_add=[
            "drawline",
            "drawopenpath",
            "drawclosedpath",
            "drawcircle",
            "drawrect",
            "eraseshape",
        ],
    )
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    return fig


# ──────────────────────────────────────────────────────────────
# Sidebar Controls
# ──────────────────────────────────────────────────────────────
st.sidebar.header("🛠️ Control Room")

# Ticker selection – searchable + large list
ticker_labels = list(CRYPTO_TICKERS.keys())
selected_name = st.sidebar.selectbox(
    "Select Crypto Asset",
    options=ticker_labels,
    index=0,
    help="Large curated list of liquid crypto pairs (USD)",
)
target_asset = CRYPTO_TICKERS[selected_name]

# Optional custom ticker
custom = st.sidebar.text_input("Or enter custom ticker (e.g. BTC-USD)", value="")
if custom.strip():
    target_asset = custom.strip().upper()
    selected_name = target_asset

time_window = st.sidebar.selectbox(
    "Lookback Window",
    options=["7d", "14d", "30d", "60d", "90d", "6mo", "1y"],
    index=2,
)
data_interval = st.sidebar.selectbox(
    "Data Resolution",
    options=["15m", "30m", "1h", "4h", "1d"],
    index=2,
)

show_volume = st.sidebar.checkbox("Show Volume Panel", value=True)
auto_refresh = st.sidebar.checkbox("Auto-refresh every 60s", value=False)
refresh_btn = st.sidebar.button("🔄 Refresh Now", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Chart is fully interactive. Use the modebar (top-right of chart) "
    "to draw lines, rectangles, circles, freehand paths, or erase shapes."
)
st.sidebar.caption("Signals are educational only — not financial advice.")

# ──────────────────────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────────────────────
st.title("⚡ Crypto Signal Bot Pro")
st.caption(
    f"Real-time multi-indicator analysis • Candlestick + Drawing tools • "
    f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)

# Force cache clear on manual refresh
if refresh_btn:
    st.cache_data.clear()

with st.spinner(f"Fetching {selected_name} ({target_asset}) …"):
    raw = fetch_crypto_data(target_asset, period=time_window, interval=data_interval)
    processed = generate_signals(raw)

if processed.empty:
    st.error(
        "No data returned. Possible causes:\n"
        "- Invalid ticker or yfinance rate limit\n"
        "- Interval too fine for the chosen lookback (e.g. 15m on 1y)\n"
        "Try a different combination."
    )
    st.stop()

latest = processed.iloc[-1]
prior = processed.iloc[-2] if len(processed) > 1 else latest

# ── Hero Metrics ──────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    delta_price = latest["Close"] - prior["Close"]
    st.metric(
        "Spot Price",
        f"${latest['Close']:,.4f}" if latest["Close"] < 10 else f"${latest['Close']:,.2f}",
        f"{delta_price:+.4f}" if abs(delta_price) < 1 else f"{delta_price:+.2f}",
    )

with m2:
    sig = latest["Signal"]
    color_class = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}.get(sig, "")
    st.markdown(f"**Current Signal**")
    st.markdown(f"<h2 class='{color_class}'>{sig}</h2>", unsafe_allow_html=True)

with m3:
    st.metric("RSI (14)", f"{latest['RSI']:.1f}")

with m4:
    st.metric("Momentum Score", f"{int(latest['Score'])}", help="-100 (strong sell) → +100 (strong buy)")

with m5:
    vol_ratio = latest.get("Vol_ratio", np.nan)
    st.metric("Volume vs MA", f"{vol_ratio:.2f}×" if pd.notna(vol_ratio) else "—")

# Extra context row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("MACD Hist", f"{latest['MACD_hist']:.4f}")
with c2:
    st.metric("ATR (14)", f"{latest['ATR']:.4f}" if pd.notna(latest.get("ATR")) else "—")
with c3:
    change_pct = latest.get("Price_Change_Pct", 0)
    st.metric("Last Candle %", f"{change_pct:+.2f}%")
with c4:
    bb_width = (latest["BB_upper"] - latest["BB_lower"]) / latest["BB_mid"] * 100 if latest["BB_mid"] else np.nan
    st.metric("BB Width %", f"{bb_width:.2f}%" if pd.notna(bb_width) else "—")

st.divider()

# ── Interactive Candlestick Chart ─────────────────────────────
st.subheader("📊 Live Analytical Chart (Candles • Draw tools enabled)")
fig = build_candlestick_chart(processed, selected_name, show_volume=show_volume)
st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": True,
        "modeBarButtonsToAdd": [
            "drawline",
            "drawopenpath",
            "drawclosedpath",
            "drawcircle",
            "drawrect",
            "eraseshape",
        ],
        "scrollZoom": True,
        "displaylogo": False,
    },
)

# ── Signal Logic Explanation ──────────────────────────────────
with st.expander("ℹ️ How the Momentum Score & Signal are calculated"):
    st.markdown("""
    **Score components (approximate weights):**
    - **EMA trend** (fast vs slow + price vs slow): ±35
    - **RSI zones** (oversold / overbought / neutral): ±20–25
    - **MACD** (line vs signal + histogram direction): ±20
    - **Bollinger position**: ±10
    - **Volume confirmation** (when volume > 1.5× MA): ±8

    **Signal rules:**
    - **BUY** when Score ≥ +35
    - **SELL** when Score ≤ −35
    - **HOLD** otherwise

    This is a transparent, rules-based educational model — not a black-box ML predictor.
    Always combine with your own analysis and risk management.
    """)

# ── Historical Audit Log ──────────────────────────────────────
st.subheader("📋 Signal Audit Log (most recent first)")
display_cols = [
    "Open", "High", "Low", "Close", "Volume",
    "RSI", "MACD", "MACD_hist", "EMA_fast", "EMA_slow",
    "Score", "Signal", "Vol_ratio", "ATR",
]
available = [c for c in display_cols if c in processed.columns]
st.dataframe(
    processed[available].sort_index(ascending=False).style.format({
        "Open": "{:.4f}", "High": "{:.4f}", "Low": "{:.4f}", "Close": "{:.4f}",
        "RSI": "{:.1f}", "MACD": "{:.4f}", "MACD_hist": "{:.4f}",
        "EMA_fast": "{:.4f}", "EMA_slow": "{:.4f}",
        "Score": "{:.0f}", "Vol_ratio": "{:.2f}", "ATR": "{:.4f}",
        "Volume": "{:,.0f}",
    }),
    use_container_width=True,
    height=380,
)

# ── Quick Stats ───────────────────────────────────────────────
st.subheader("📈 Session Snapshot")
s1, s2, s3 = st.columns(3)
with s1:
    st.write("**Signal distribution (lookback)**")
    st.write(processed["Signal"].value_counts())
with s2:
    st.write("**Score stats**")
    st.write(processed["Score"].describe()[["mean", "std", "min", "max"]].round(1))
with s3:
    st.write("**RSI extremes**")
    st.write(f"Min RSI: {processed['RSI'].min():.1f}")
    st.write(f"Max RSI: {processed['RSI'].max():.1f}")
    st.write(f"Current: {latest['RSI']:.1f}")

# Auto-refresh loop
if auto_refresh:
    time.sleep(60)
    st.rerun()
