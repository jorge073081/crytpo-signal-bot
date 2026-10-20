"""
Crypto Signal Bot Pro — scanner + alerts + more tickers
=======================================================
Binance data · Altair tight-scale charts · multi-ticker scanner · alerts
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timezone
import requests
import time

st.set_page_config(page_title="Crypto Signal Bot Pro", page_icon="⚡", layout="wide")

st.markdown("""
<style>
div[data-testid="stMetricValue"] { font-size: 1.35rem; }
.signal-buy  { color: #00c853; font-weight: 800; font-size: 1.6rem; }
.signal-sell { color: #ff1744; font-weight: 800; font-size: 1.6rem; }
.signal-hold { color: #ffab00; font-weight: 800; font-size: 1.6rem; }
.alert-box { padding: 10px 14px; border-radius: 8px; margin: 6px 0; font-weight: 600; }
.alert-buy { background: rgba(0,200,83,0.15); border-left: 4px solid #00c853; }
.alert-sell { background: rgba(255,23,68,0.15); border-left: 4px solid #ff1744; }
</style>
""", unsafe_allow_html=True)

# ── Expanded ticker list ──────────────────────────────────────
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
    "ONDO": "ONDOUSDT",
    "Stellar": "XLMUSDT",
    "Hedera": "HBARUSDT",
    "Algorand": "ALGOUSDT",
    "Flare": "FLRUSDT",
    "Cosmos": "ATOMUSDT",
    "Filecoin": "FILUSDT",
    "Injective": "INJUSDT",
    "Fetch.ai": "FETUSDT",
    "Worldcoin": "WLDUSDT",
    "Immutable": "IMXUSDT",
    "The Graph": "GRTUSDT",
    "VeChain": "VETUSDT",
    "Internet Computer": "ICPUSDT",
}

INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]
HOSTS = ["https://data-api.binance.vision", "https://api.binance.us"]

# Default scan list (major + requested)
DEFAULT_SCAN = [
    "Bitcoin", "Ethereum", "Solana", "XRP", "ONDO", "Stellar",
    "Hedera", "Algorand", "Flare", "Dogecoin", "Cardano", "Chainlink",
]


@st.cache_data(ttl=25, show_spinner=False)
def fetch_klines(symbol: str, interval: str, limit: int = 300):
    params = {"symbol": symbol, "interval": interval, "limit": min(int(limit), 1000)}
    for host in HOSTS:
        try:
            r = requests.get(f"{host}/api/v3/klines", params=params, timeout=10)
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
                return df
        except Exception:
            continue
    return pd.DataFrame()


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
    d["EMA_50"] = ema(c, 50) if len(d) >= 50 else np.nan
    d["RSI"] = rsi(c)
    d["MACD"] = d["EMA_fast"] - d["EMA_slow"]
    d["MACD_sig"] = ema(d["MACD"], 9)
    d["MACD_hist"] = d["MACD"] - d["MACD_sig"]
    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    d["BB_u"] = mid + 2 * std
    d["BB_l"] = mid - 2 * std
    d["BB_width"] = (d["BB_u"] - d["BB_l"]) / mid.replace(0, np.nan) * 100
    d["Vol_MA"] = d["Volume"].rolling(20).mean()
    d["Vol_ratio"] = d["Volume"] / d["Vol_MA"].replace(0, np.nan)
    d["Pct_chg"] = c.pct_change() * 100

    # Extra signal flags
    d["MACD_cross_up"] = (d["MACD"] > d["MACD_sig"]) & (d["MACD"].shift() <= d["MACD_sig"].shift())
    d["MACD_cross_dn"] = (d["MACD"] < d["MACD_sig"]) & (d["MACD"].shift() >= d["MACD_sig"].shift())
    d["EMA_cross_up"] = (d["EMA_fast"] > d["EMA_slow"]) & (d["EMA_fast"].shift() <= d["EMA_slow"].shift())
    d["EMA_cross_dn"] = (d["EMA_fast"] < d["EMA_slow"]) & (d["EMA_fast"].shift() >= d["EMA_slow"].shift())
    d["RSI_oversold"] = d["RSI"] < 30
    d["RSI_overbought"] = d["RSI"] > 70
    d["Vol_spike"] = d["Vol_ratio"] > 2.0
    d["BB_squeeze"] = d["BB_width"] < d["BB_width"].rolling(50).mean() * 0.75

    score = pd.Series(0.0, index=d.index)
    score += np.where(d["EMA_fast"] > d["EMA_slow"], 20, -20)
    score += np.where(c > d["EMA_slow"], 10, -10)
    score += np.where(d["RSI"] < 30, 20, 0)
    score += np.where(d["RSI"] > 70, -20, 0)
    score += np.where(d["MACD"] > d["MACD_sig"], 15, -15)
    score += np.where(d["MACD_hist"] > d["MACD_hist"].shift(), 5, -5)
    score += np.where(d["Vol_spike"] & (d["Pct_chg"] > 0), 8, 0)
    score += np.where(d["Vol_spike"] & (d["Pct_chg"] < 0), -8, 0)
    score += np.where(d["EMA_cross_up"], 12, 0)
    score += np.where(d["EMA_cross_dn"], -12, 0)
    d["Score"] = score.clip(-100, 100).round(0)
    d["Signal"] = d["Score"].map(lambda x: "BUY" if x >= 35 else ("SELL" if x <= -35 else "HOLD"))
    return d.dropna(subset=["RSI"])


def snapshot(name: str, symbol: str, interval: str) -> dict:
    """One-row summary for scanner."""
    raw = fetch_klines(symbol, interval, 120)
    d = add_signals(raw)
    if d.empty:
        return {"Asset": name, "Symbol": symbol, "Status": "no data"}
    last = d.iloc[-1]
    prev = d.iloc[-2] if len(d) > 1 else last
    chg = (float(last["Close"]) - float(prev["Close"])) / float(prev["Close"]) * 100
    flags = []
    if last.get("MACD_cross_up"):
        flags.append("MACD↑")
    if last.get("MACD_cross_dn"):
        flags.append("MACD↓")
    if last.get("EMA_cross_up"):
        flags.append("EMA↑")
    if last.get("EMA_cross_dn"):
        flags.append("EMA↓")
    if last.get("RSI_oversold"):
        flags.append("OS")
    if last.get("RSI_overbought"):
        flags.append("OB")
    if last.get("Vol_spike"):
        flags.append("VOL")
    if last.get("BB_squeeze"):
        flags.append("SQZ")
    return {
        "Asset": name,
        "Symbol": symbol,
        "Price": float(last["Close"]),
        "Chg%": round(chg, 2),
        "RSI": round(float(last["RSI"]), 1),
        "Score": int(last["Score"]),
        "Signal": str(last["Signal"]),
        "Flags": " ".join(flags) if flags else "—",
        "Status": "ok",
    }


def tight_line_chart(df, cols, height=400, title=""):
    plot = df[cols].copy().reset_index()
    time_col = plot.columns[0]
    plot = plot.rename(columns={time_col: "Time"})
    # Friendly names
    rename = {"EMA_fast": "EMA 12", "EMA_slow": "EMA 26", "EMA_50": "EMA 50"}
    plot = plot.rename(columns=rename)
    cols_disp = [rename.get(c, c) for c in cols]
    long = plot.melt("Time", value_vars=cols_disp, var_name="Series", value_name="Value")
    ymin, ymax = float(long["Value"].min()), float(long["Value"].max())
    pad = (ymax - ymin) * 0.08 if ymax > ymin else max(abs(ymin) * 0.01, 1e-8)
    chart = (
        alt.Chart(long)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("Time:T", title=None),
            y=alt.Y("Value:Q", title=None, scale=alt.Scale(domain=[ymin - pad, ymax + pad], zero=False)),
            color=alt.Color("Series:N", legend=alt.Legend(orient="top")),
            tooltip=["Time:T", "Series:N", alt.Tooltip("Value:Q", format=".6f")],
        )
        .properties(height=height, title=title)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def tight_bar_chart(df, col, height=220, title=""):
    plot = df[[col]].copy().reset_index()
    plot = plot.rename(columns={plot.columns[0]: "Time", col: "Value"})
    ymin, ymax = float(plot["Value"].min()), float(plot["Value"].max())
    if ymin > 0 and ymin > (ymax - ymin) * 0.25:
        pad = (ymax - ymin) * 0.1
        domain, zero = [ymin - pad, ymax + pad], False
    else:
        domain, zero = [min(0, ymin), ymax * 1.08 if ymax else 1], True
    chart = (
        alt.Chart(plot)
        .mark_bar(opacity=0.75)
        .encode(
            x=alt.X("Time:T", title=None),
            y=alt.Y("Value:Q", title=None, scale=alt.Scale(domain=domain, zero=zero)),
            tooltip=["Time:T", alt.Tooltip("Value:Q", format=".4f")],
        )
        .properties(height=height, title=title)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("🛠️ Control Room")
page = st.sidebar.radio("View", ["Single asset", "Multi-ticker scanner"], index=0)

interval = st.sidebar.selectbox("Candle size", INTERVALS, index=3)
limit = st.sidebar.select_slider("Candles shown", options=[50, 100, 150, 200, 300], value=100)

st.sidebar.markdown("---")
st.sidebar.subheader("🔴 Live feed")
live = st.sidebar.checkbox("Auto-refresh ON", value=False)
secs = st.sidebar.slider("Seconds between updates", 10, 60, 20)
if st.sidebar.button("🔄 Refresh now", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# ══════════════════════════════════════════════════════════════
# PAGE: Multi-ticker scanner
# ══════════════════════════════════════════════════════════════
if page == "Multi-ticker scanner":
    st.title("📡 Multi-Ticker Scanner")
    st.caption(f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC · interval {interval}")

    scan_names = st.multiselect(
        "Assets to scan",
        options=list(CRYPTO.keys()),
        default=[n for n in DEFAULT_SCAN if n in CRYPTO],
    )
    if not scan_names:
        st.warning("Select at least one asset.")
        st.stop()

    rows = []
    prog = st.progress(0, text="Scanning…")
    for i, n in enumerate(scan_names):
        rows.append(snapshot(n, CRYPTO[n], interval))
        prog.progress((i + 1) / len(scan_names), text=f"Scanned {n}")
    prog.empty()

    scan_df = pd.DataFrame(rows)
    ok = scan_df[scan_df["Status"] == "ok"].copy() if "Status" in scan_df.columns else scan_df

    # Alerts from scanner
    st.subheader("🚨 Active alerts")
    alerts = []
    if not ok.empty:
        for _, r in ok.iterrows():
            if r["Signal"] == "BUY":
                alerts.append(("buy", f"**{r['Asset']}** BUY · Score {r['Score']} · RSI {r['RSI']} · {r['Flags']}"))
            elif r["Signal"] == "SELL":
                alerts.append(("sell", f"**{r['Asset']}** SELL · Score {r['Score']} · RSI {r['RSI']} · {r['Flags']}"))
            elif "OS" in str(r["Flags"]):
                alerts.append(("buy", f"**{r['Asset']}** RSI oversold ({r['RSI']})"))
            elif "OB" in str(r["Flags"]):
                alerts.append(("sell", f"**{r['Asset']}** RSI overbought ({r['RSI']})"))
            elif "MACD↑" in str(r["Flags"]) or "EMA↑" in str(r["Flags"]):
                alerts.append(("buy", f"**{r['Asset']}** bullish cross · {r['Flags']}"))
            elif "MACD↓" in str(r["Flags"]) or "EMA↓" in str(r["Flags"]):
                alerts.append(("sell", f"**{r['Asset']}** bearish cross · {r['Flags']}"))

    if not alerts:
        st.info("No strong alerts right now (no BUY/SELL or key flags).")
    else:
        for kind, msg in alerts:
            cls = "alert-buy" if kind == "buy" else "alert-sell"
            st.markdown(f"<div class='alert-box {cls}'>{msg}</div>", unsafe_allow_html=True)

    st.subheader("Scanner table")
    if ok.empty:
        st.warning("No data returned for selected assets.")
    else:
        show = ok.drop(columns=["Status"], errors="ignore")
        # Sort: BUY first, then by score desc
        order = {"BUY": 0, "HOLD": 1, "SELL": 2}
        show["_ord"] = show["Signal"].map(order)
        show = show.sort_values(["_ord", "Score"], ascending=[True, False]).drop(columns=["_ord"])

        def fmt_price(x):
            if pd.isna(x):
                return "—"
            if x < 0.01:
                return f"${x:.6f}"
            if x < 1:
                return f"${x:.4f}"
            if x < 100:
                return f"${x:.3f}"
            return f"${x:,.2f}"

        show["Price"] = show["Price"].map(fmt_price)
        st.dataframe(show, use_container_width=True, height=420)

        st.caption("Flags: MACD↑/↓ · EMA↑/↓ · OS=oversold · OB=overbought · VOL=volume spike · SQZ=BB squeeze")

    if live:
        time.sleep(secs)
        st.rerun()
    st.stop()

# ══════════════════════════════════════════════════════════════
# PAGE: Single asset
# ══════════════════════════════════════════════════════════════
st.sidebar.markdown("---")
name = st.sidebar.selectbox("Asset", list(CRYPTO.keys()), index=0)
symbol = CRYPTO[name]
custom = st.sidebar.text_input("Custom symbol (e.g. ONDOUSDT)", "")
if custom.strip():
    symbol = custom.strip().upper()
    name = symbol

st.title("⚡ Crypto Signal Bot Pro")
st.caption(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC · {name} · {symbol} · {interval}")

with st.spinner("Loading…"):
    raw = fetch_klines(symbol, interval, max(limit + 50, 120))
    df_full = add_signals(raw)

if df_full.empty:
    st.error("No data. Check symbol (must be like BTCUSDT) or try another interval.")
    st.stop()

df = df_full.tail(limit).copy()
st.success(f"Data OK · last **{len(df)}** candles · {interval}")

latest = df.iloc[-1]
prev = df.iloc[-2]
delta = float(latest["Close"] - prev["Close"])
pct = delta / float(prev["Close"]) * 100 if prev["Close"] else 0
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
c5.metric("Vol vs MA", f"{float(latest.get('Vol_ratio', 0)):.2f}×")

# Single-asset alerts
flags = []
if latest.get("MACD_cross_up"):
    flags.append(("buy", "MACD bullish cross"))
if latest.get("MACD_cross_dn"):
    flags.append(("sell", "MACD bearish cross"))
if latest.get("EMA_cross_up"):
    flags.append(("buy", "EMA 12/26 bullish cross"))
if latest.get("EMA_cross_dn"):
    flags.append(("sell", "EMA 12/26 bearish cross"))
if latest.get("RSI_oversold"):
    flags.append(("buy", f"RSI oversold ({latest['RSI']:.0f})"))
if latest.get("RSI_overbought"):
    flags.append(("sell", f"RSI overbought ({latest['RSI']:.0f})"))
if latest.get("Vol_spike"):
    flags.append(("buy" if latest["Pct_chg"] > 0 else "sell", f"Volume spike {latest['Vol_ratio']:.1f}×"))
if latest.get("BB_squeeze"):
    flags.append(("buy", "Bollinger squeeze (volatility contraction)"))

if flags:
    st.subheader("🚨 Alerts")
    for kind, msg in flags:
        box = "alert-buy" if kind == "buy" else "alert-sell"
        st.markdown(f"<div class='alert-box {box}'>{msg}</div>", unsafe_allow_html=True)

st.divider()
st.subheader(f"📈 {name} · Close + EMAs · {interval}")
tight_line_chart(df, ["Close", "EMA_fast", "EMA_slow"], height=400)

st.subheader("High · Low · Close")
tight_line_chart(df, ["High", "Low", "Close"], height=280)

a, b = st.columns(2)
with a:
    st.subheader("RSI (14)")
    tight_line_chart(df, ["RSI"], height=230)
with b:
    st.subheader("Volume")
    tight_bar_chart(df, "Volume", height=230)

st.subheader("MACD histogram")
tight_bar_chart(df, "MACD_hist", height=200)

st.subheader("Recent candles + signals")
st.dataframe(
    df[["Open", "High", "Low", "Close", "Volume", "RSI", "MACD_hist", "Score", "Signal"]]
      .sort_index(ascending=False)
      .head(40),
    use_container_width=True,
    height=280,
)

with st.expander("Signal rules"):
    st.markdown("""
    **Score (−100 … +100)**  
    EMA trend ±20 · price vs slow EMA ±10 · RSI zones ±20 · MACD ±15 · hist direction ±5 ·  
    volume spike ±8 · EMA cross ±12  

    **BUY** ≥ +35 · **SELL** ≤ −35 · else **HOLD**

    **Flags:** MACD/EMA crosses · RSI OS/OB · volume spike · BB squeeze
    """)

if live:
    st.caption(f"Refreshing in {secs}s…")
    time.sleep(secs)
    st.rerun()
