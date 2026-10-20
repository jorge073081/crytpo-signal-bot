"""
Crypto Signal Bot Pro — tight, readable charts
==============================================
Native Streamlit charts only (no black Plotly boxes).
Main price chart = Close + EMAs only so the scale stays tight.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import requests
import time

st.set_page_config(page_title="Crypto Signal Bot Pro", page_icon="⚡", layout="wide")

st.markdown("""
<style>
div[data-testid="stMetricValue"] { font-size: 1.45rem; }
.signal-buy  { color: #00c853; font-weight: 800; font-size: 1.7rem; }
.signal-sell { color: #ff1744; font-weight: 800; font-size: 1.7rem; }
.signal-hold { color: #ffab00; font-weight: 800; font-size: 1.7rem; }
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
    d["BB_mid"] = mid
    d["BB_l"] = mid - 2 * std
    d["BB_width"] = (d["BB_u"] - d["BB_l"]) / mid * 100

    score = pd.Series(0.0, index=d.index)
    score += np.where(d["EMA_fast"] > d["EMA_slow"], 25, -25)
    score += np.where(c > d["EMA_slow"], 10, -10)
    score += np.where(d["RSI"] < 30, 20, 0)
    score += np.where(d["RSI"] > 70, -20, 0)
    score += np.where(d["MACD"] > d["MACD_sig"], 15, -15)
    d["Score"] = score.clip(-100, 100).round(0)
    d["Signal"] = d["Score"].map(lambda x: "BUY" if x >= 35 else ("SELL" if x <= -35 else "HOLD"))
    return d.dropna(subset=["RSI"])


# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("🛠️ Control Room")
name = st.sidebar.selectbox("Asset", list(CRYPTO.keys()), index=0)
symbol = CRYPTO[name]
custom = st.sidebar.text_input("Custom symbol (e.g. BTCUSDT)", "")
if custom.strip():
    symbol = custom.strip().upper()
    name = symbol

interval = st.sidebar.selectbox("Candle size", INTERVALS, index=3)
limit = st.sidebar.select_slider("Candles shown", options=[50, 100, 150, 200, 300], value=100)

st.sidebar.markdown("---")
st.sidebar.subheader("🔴 Live feed")
live = st.sidebar.checkbox("Auto-refresh ON", value=False)
secs = st.sidebar.slider("Seconds between updates", 8, 60, 15)
if st.sidebar.button("🔄 Refresh now", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# ── Main ──────────────────────────────────────────────────────
st.title("⚡ Crypto Signal Bot Pro")
st.caption(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC · {name} · {symbol} · {interval}")

with st.spinner("Loading…"):
    raw, source = fetch_klines(symbol, interval, max(limit, 100))
    df_full = add_signals(raw)

if df_full.empty:
    st.error("No data. Try another interval or symbol.")
    st.stop()

# Only show the last N candles so the chart is tight and readable
df = df_full.tail(limit).copy()

st.success(f"Data OK · **{source}** · showing last **{len(df)}** candles · {interval}")

latest = df.iloc[-1]
prev = df.iloc[-2]
delta = float(latest["Close"] - prev["Close"])
pct = delta / float(prev["Close"]) * 100
price = float(latest["Close"])
price_s = f"${price:,.6f}" if price < 1 else (f"${price:,.4f}" if price < 100 else f"${price:,.2f}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Price", price_s, f"{delta:+.4f} ({pct:+.2f}%)")
sig = str(latest["Signal"])
cls = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}.get(sig, "")
c2.markdown("**Signal**")
c2.markdown(f"<div class='{cls}'>{sig}</div>", unsafe_allow_html=True)
c3.metric("RSI", f"{float(latest['RSI']):.1f}")
c4.metric("Score", f"{int(latest['Score'])}")
c5.metric("High / Low", f"{float(latest['High']):.4g} / {float(latest['Low']):.4g}")

st.divider()

# ── MAIN PRICE (Close + EMAs only → tight y-scale) ────────────
st.subheader(f"📈 {name} · Close + EMAs · {interval}")
main = df[["Close", "EMA_fast", "EMA_slow"]].copy()
main.columns = ["Close", "EMA 12", "EMA 26"]
st.line_chart(main, height=400, use_container_width=True)

# ── High / Low / Close (wick context, same tight window) ──────
st.subheader("High · Low · Close")
st.line_chart(df[["High", "Low", "Close"]], height=280, use_container_width=True)

# ── RSI + Volume ──────────────────────────────────────────────
a, b = st.columns(2)
with a:
    st.subheader("RSI (14)")
    st.line_chart(df[["RSI"]], height=240, use_container_width=True)
    st.caption("Overbought > 70 · Oversold < 30")
with b:
    st.subheader("Volume")
    st.bar_chart(df[["Volume"]], height=240, use_container_width=True)

# ── MACD ──────────────────────────────────────────────────────
st.subheader("MACD histogram")
st.bar_chart(df[["MACD_hist"]], height=200, use_container_width=True)

# ── Bollinger width (separate so it doesn't stretch price) ────
st.subheader("Bollinger band width %")
st.line_chart(df[["BB_width"]], height=180, use_container_width=True)

# ── Table ─────────────────────────────────────────────────────
st.subheader("Recent candles + signals")
st.dataframe(
    df[["Open", "High", "Low", "Close", "Volume", "RSI", "MACD_hist", "Score", "Signal"]]
      .sort_index(ascending=False)
      .head(40),
    use_container_width=True,
    height=300,
)

if live:
    st.caption(f"Refreshing in {secs}s…")
    time.sleep(secs)
    st.rerun()
