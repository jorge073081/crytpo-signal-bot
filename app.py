"""
Crypto Signal Bot Pro — Buy Low / Sell High
===========================================
Mean-reversion + momentum signals aimed at buying weakness and selling strength.
Paper-trade view shows hypothetical results. Not live exchange execution.
Educational only — not financial advice.
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
div[data-testid="stMetricValue"] { font-size: 1.3rem; }
.signal-buy  { color: #00c853; font-weight: 800; font-size: 1.55rem; }
.signal-sell { color: #ff1744; font-weight: 800; font-size: 1.55rem; }
.signal-hold { color: #ffab00; font-weight: 800; font-size: 1.55rem; }
.alert-box { padding: 10px 14px; border-radius: 8px; margin: 6px 0; font-weight: 600; }
.alert-buy { background: rgba(0,200,83,0.15); border-left: 4px solid #00c853; }
.alert-sell { background: rgba(255,23,68,0.15); border-left: 4px solid #ff1744; }
.level-box { background: rgba(255,255,255,0.04); padding: 12px 16px; border-radius: 10px; }
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
    "Render": "RENDERUSDT", "ONDO": "ONDOUSDT", "Stellar": "XLMUSDT",
    "Hedera": "HBARUSDT", "Algorand": "ALGOUSDT", "Flare": "FLRUSDT",
    "Cosmos": "ATOMUSDT", "Filecoin": "FILUSDT", "Injective": "INJUSDT",
    "Fetch.ai": "FETUSDT", "Worldcoin": "WLDUSDT", "Immutable": "IMXUSDT",
    "The Graph": "GRTUSDT", "VeChain": "VETUSDT", "Internet Computer": "ICPUSDT",
}

INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]
HOSTS = ["https://data-api.binance.vision", "https://api.binance.us"]
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
    """Buy-low / sell-high oriented scoring (mean reversion + trend filter)."""
    if df.empty or len(df) < 35:
        return pd.DataFrame()
    d = df.copy()
    c = d["Close"]
    d["EMA_fast"] = ema(c, 12)
    d["EMA_slow"] = ema(c, 26)
    d["EMA_50"] = ema(c, 50) if len(d) >= 50 else ema(c, min(50, len(d) // 2))
    d["RSI"] = rsi(c)
    d["MACD"] = d["EMA_fast"] - d["EMA_slow"]
    d["MACD_sig"] = ema(d["MACD"], 9)
    d["MACD_hist"] = d["MACD"] - d["MACD_sig"]

    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    d["BB_u"] = mid + 2 * std
    d["BB_mid"] = mid
    d["BB_l"] = mid - 2 * std
    d["BB_pct"] = (c - d["BB_l"]) / (d["BB_u"] - d["BB_l"]).replace(0, np.nan)  # 0=low, 1=high
    d["BB_width"] = (d["BB_u"] - d["BB_l"]) / mid.replace(0, np.nan) * 100

    d["Vol_MA"] = d["Volume"].rolling(20).mean()
    d["Vol_ratio"] = d["Volume"] / d["Vol_MA"].replace(0, np.nan)
    d["Pct_chg"] = c.pct_change() * 100

    # Swing levels (recent range for buy-low / sell-high)
    d["Swing_low"] = d["Low"].rolling(20).min()
    d["Swing_high"] = d["High"].rolling(20).max()
    d["Dist_to_low"] = (c - d["Swing_low"]) / c.replace(0, np.nan) * 100
    d["Dist_to_high"] = (d["Swing_high"] - c) / c.replace(0, np.nan) * 100

    # Event flags
    d["MACD_cross_up"] = (d["MACD"] > d["MACD_sig"]) & (d["MACD"].shift() <= d["MACD_sig"].shift())
    d["MACD_cross_dn"] = (d["MACD"] < d["MACD_sig"]) & (d["MACD"].shift() >= d["MACD_sig"].shift())
    d["EMA_cross_up"] = (d["EMA_fast"] > d["EMA_slow"]) & (d["EMA_fast"].shift() <= d["EMA_slow"].shift())
    d["EMA_cross_dn"] = (d["EMA_fast"] < d["EMA_slow"]) & (d["EMA_fast"].shift() >= d["EMA_slow"].shift())
    d["RSI_oversold"] = d["RSI"] < 32
    d["RSI_overbought"] = d["RSI"] > 68
    d["Near_low"] = d["BB_pct"] < 0.2
    d["Near_high"] = d["BB_pct"] > 0.8
    d["Vol_spike"] = d["Vol_ratio"] > 1.8

    # ── BUY LOW / SELL HIGH SCORE ─────────────────────────────
    # Positive = buy bias (price is "low"), Negative = sell bias (price is "high")
    score = pd.Series(0.0, index=d.index)

    # Mean reversion (core buy-low / sell-high)
    score += np.where(d["RSI"] < 25, 30, 0)
    score += np.where((d["RSI"] >= 25) & (d["RSI"] < 35), 18, 0)
    score += np.where(d["RSI"] > 75, -30, 0)
    score += np.where((d["RSI"] <= 75) & (d["RSI"] > 65), -18, 0)

    score += np.where(d["BB_pct"] < 0.15, 22, 0)   # near lower band → buy low
    score += np.where(d["BB_pct"] > 0.85, -22, 0)  # near upper band → sell high
    score += np.where(d["Dist_to_low"] < 1.5, 12, 0)
    score += np.where(d["Dist_to_high"] < 1.5, -12, 0)

    # Trend filter (don't catch falling knives blindly)
    uptrend = d["EMA_fast"] > d["EMA_slow"]
    score += np.where(uptrend, 8, -8)
    score += np.where(c > d["EMA_50"], 6, -6)

    # Momentum confirmation
    score += np.where(d["MACD"] > d["MACD_sig"], 10, -10)
    score += np.where(d["MACD_cross_up"], 14, 0)
    score += np.where(d["MACD_cross_dn"], -14, 0)
    score += np.where(d["EMA_cross_up"], 10, 0)
    score += np.where(d["EMA_cross_dn"], -10, 0)

    # Volume: spike + green candle near low = stronger buy
    score += np.where(d["Vol_spike"] & (d["Pct_chg"] > 0) & (d["BB_pct"] < 0.4), 10, 0)
    score += np.where(d["Vol_spike"] & (d["Pct_chg"] < 0) & (d["BB_pct"] > 0.6), -10, 0)

    d["Score"] = score.clip(-100, 100).round(0)

    def classify(row):
        s = row["Score"]
        # Strong buy low
        if s >= 40 or (s >= 28 and row["RSI"] < 35 and row["BB_pct"] < 0.35):
            return "BUY"
        # Strong sell high
        if s <= -40 or (s <= -28 and row["RSI"] > 65 and row["BB_pct"] > 0.65):
            return "SELL"
        return "HOLD"

    d["Signal"] = d.apply(classify, axis=1)

    # Suggested levels for the latest bar (filled for all rows for simplicity)
    d["Buy_zone"] = d[["BB_l", "Swing_low"]].max(axis=1)  # prefer higher of the two lows as zone
    d["Sell_zone"] = d[["BB_u", "Swing_high"]].min(axis=1)
    d["Stop_idea"] = d["Swing_low"] * 0.985  # soft stop under swing low
    d["Target_idea"] = d["Swing_high"] * 0.995

    return d.dropna(subset=["RSI", "BB_pct"])


def paper_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Simulate: enter on BUY, exit on next SELL (or end). Buy-low / sell-high style."""
    if df.empty:
        return pd.DataFrame()
    trades = []
    in_pos = False
    entry_price = entry_time = None
    for ts, row in df.iterrows():
        if not in_pos and row["Signal"] == "BUY":
            in_pos = True
            entry_price = float(row["Close"])
            entry_time = ts
        elif in_pos and row["Signal"] == "SELL":
            exit_price = float(row["Close"])
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            trades.append({
                "Entry time": entry_time,
                "Entry $": entry_price,
                "Exit time": ts,
                "Exit $": exit_price,
                "PnL %": round(pnl_pct, 2),
                "Result": "WIN" if pnl_pct > 0 else "LOSS",
            })
            in_pos = False
    if in_pos:
        exit_price = float(df.iloc[-1]["Close"])
        pnl_pct = (exit_price - entry_price) / entry_price * 100
        trades.append({
            "Entry time": entry_time,
            "Entry $": entry_price,
            "Exit time": df.index[-1],
            "Exit $": exit_price,
            "PnL %": round(pnl_pct, 2),
            "Result": "OPEN",
        })
    return pd.DataFrame(trades)


def snapshot(name, symbol, interval):
    raw = fetch_klines(symbol, interval, 150)
    d = add_signals(raw)
    if d.empty:
        return {"Asset": name, "Symbol": symbol, "Status": "no data"}
    last = d.iloc[-1]
    prev = d.iloc[-2] if len(d) > 1 else last
    chg = (float(last["Close"]) - float(prev["Close"])) / float(prev["Close"]) * 100
    flags = []
    if last.get("Near_low") or last.get("RSI_oversold"):
        flags.append("LOW")
    if last.get("Near_high") or last.get("RSI_overbought"):
        flags.append("HIGH")
    if last.get("MACD_cross_up"):
        flags.append("MACD↑")
    if last.get("MACD_cross_dn"):
        flags.append("MACD↓")
    if last.get("Vol_spike"):
        flags.append("VOL")
    return {
        "Asset": name,
        "Symbol": symbol,
        "Price": float(last["Close"]),
        "Chg%": round(chg, 2),
        "RSI": round(float(last["RSI"]), 1),
        "BB%": round(float(last["BB_pct"]) * 100, 0) if pd.notna(last["BB_pct"]) else None,
        "Score": int(last["Score"]),
        "Signal": str(last["Signal"]),
        "Buy zone": float(last["Buy_zone"]) if pd.notna(last["Buy_zone"]) else None,
        "Sell zone": float(last["Sell_zone"]) if pd.notna(last["Sell_zone"]) else None,
        "Flags": " ".join(flags) if flags else "—",
        "Status": "ok",
    }


def tight_line_chart(df, cols, height=400, title=""):
    plot = df[cols].copy().reset_index()
    plot = plot.rename(columns={plot.columns[0]: "Time"})
    rename = {"EMA_fast": "EMA 12", "EMA_slow": "EMA 26", "EMA_50": "EMA 50",
              "BB_l": "BB Low", "BB_u": "BB High", "Buy_zone": "Buy zone", "Sell_zone": "Sell zone"}
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


def tight_bar_chart(df, col, height=200, title=""):
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


def fmt_px(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if x < 0.01:
        return f"${x:.6f}"
    if x < 1:
        return f"${x:.4f}"
    if x < 100:
        return f"${x:.3f}"
    return f"${x:,.2f}"


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
st.sidebar.caption("Strategy: buy low (oversold / BB low) · sell high (overbought / BB high). Signals only — not live orders.")

# ══════════════════════════════════════════════════════════════
# SCANNER
# ══════════════════════════════════════════════════════════════
if page == "Multi-ticker scanner":
    st.title("📡 Scanner · Buy Low / Sell High")
    st.caption(f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC · {interval}")

    scan_names = st.multiselect(
        "Assets",
        list(CRYPTO.keys()),
        default=[n for n in DEFAULT_SCAN if n in CRYPTO],
    )
    if not scan_names:
        st.warning("Select assets.")
        st.stop()

    rows = []
    prog = st.progress(0, text="Scanning…")
    for i, n in enumerate(scan_names):
        rows.append(snapshot(n, CRYPTO[n], interval))
        prog.progress((i + 1) / len(scan_names), text=f"{n}")
    prog.empty()

    scan_df = pd.DataFrame(rows)
    ok = scan_df[scan_df.get("Status", "ok") == "ok"].copy() if "Status" in scan_df.columns else scan_df

    st.subheader("🚨 Buy-low / sell-high alerts")
    alerts = []
    if not ok.empty:
        for _, r in ok.iterrows():
            if r["Signal"] == "BUY":
                alerts.append(("buy", f"**BUY LOW** {r['Asset']} · Score {r['Score']} · RSI {r['RSI']} · zone {fmt_px(r.get('Buy zone'))}"))
            elif r["Signal"] == "SELL":
                alerts.append(("sell", f"**SELL HIGH** {r['Asset']} · Score {r['Score']} · RSI {r['RSI']} · zone {fmt_px(r.get('Sell zone'))}"))
            elif "LOW" in str(r["Flags"]):
                alerts.append(("buy", f"Near low: **{r['Asset']}** RSI {r['RSI']} · {r['Flags']}"))
            elif "HIGH" in str(r["Flags"]):
                alerts.append(("sell", f"Near high: **{r['Asset']}** RSI {r['RSI']} · {r['Flags']}"))
    if not alerts:
        st.info("No strong buy-low / sell-high alerts right now.")
    else:
        for kind, msg in alerts:
            st.markdown(f"<div class='alert-box {'alert-buy' if kind=='buy' else 'alert-sell'}'>{msg}</div>", unsafe_allow_html=True)

    st.subheader("Scanner")
    if ok.empty:
        st.warning("No data.")
    else:
        show = ok.drop(columns=["Status"], errors="ignore")
        order = {"BUY": 0, "HOLD": 1, "SELL": 2}
        show["_o"] = show["Signal"].map(order)
        show = show.sort_values(["_o", "Score"], ascending=[True, False]).drop(columns=["_o"])
        show["Price"] = show["Price"].map(fmt_px)
        if "Buy zone" in show.columns:
            show["Buy zone"] = show["Buy zone"].map(fmt_px)
        if "Sell zone" in show.columns:
            show["Sell zone"] = show["Sell zone"].map(fmt_px)
        st.dataframe(show, use_container_width=True, height=440)
        st.caption("LOW = near BB/swing low · HIGH = near BB/swing high · Prefer BUY when LOW + rising momentum.")

    if live:
        time.sleep(secs)
        st.rerun()
    st.stop()

# ══════════════════════════════════════════════════════════════
# SINGLE ASSET
# ══════════════════════════════════════════════════════════════
st.sidebar.markdown("---")
name = st.sidebar.selectbox("Asset", list(CRYPTO.keys()), index=0)
symbol = CRYPTO[name]
custom = st.sidebar.text_input("Custom (e.g. ONDOUSDT)", "")
if custom.strip():
    symbol = custom.strip().upper()
    name = symbol

st.title("⚡ Buy Low · Sell High")
st.caption(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC · {name} · {symbol} · {interval}")

with st.spinner("Loading…"):
    raw = fetch_klines(symbol, interval, max(limit + 80, 150))
    df_full = add_signals(raw)

if df_full.empty:
    st.error("No data. Use symbols like BTCUSDT / ONDOUSDT.")
    st.stop()

df = df_full.tail(limit).copy()
st.success(f"Data OK · **{len(df)}** candles · {interval}")

latest = df.iloc[-1]
prev = df.iloc[-2]
delta = float(latest["Close"] - prev["Close"])
pct = delta / float(prev["Close"]) * 100 if prev["Close"] else 0
price = float(latest["Close"])

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Price", fmt_px(price), f"{delta:+.4f} ({pct:+.2f}%)")
sig = str(latest["Signal"])
cls = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}.get(sig, "")
c2.markdown("**Action**")
c2.markdown(f"<div class='{cls}'>{'BUY LOW' if sig=='BUY' else ('SELL HIGH' if sig=='SELL' else 'HOLD')}</div>", unsafe_allow_html=True)
c3.metric("RSI", f"{float(latest['RSI']):.1f}")
c4.metric("Score", f"{int(latest['Score'])}")
bbp = float(latest["BB_pct"]) * 100 if pd.notna(latest["BB_pct"]) else 50
c5.metric("BB position", f"{bbp:.0f}%", help="0% = lower band (low) · 100% = upper band (high)")

# Levels
st.markdown("### 🎯 Levels (buy low / sell high)")
l1, l2, l3, l4 = st.columns(4)
with l1:
    st.markdown("<div class='level-box'><b>Buy zone</b><br>" + fmt_px(float(latest["Buy_zone"])) + "</div>", unsafe_allow_html=True)
with l2:
    st.markdown("<div class='level-box'><b>Sell zone</b><br>" + fmt_px(float(latest["Sell_zone"])) + "</div>", unsafe_allow_html=True)
with l3:
    st.markdown("<div class='level-box'><b>Stop idea</b><br>" + fmt_px(float(latest["Stop_idea"])) + "</div>", unsafe_allow_html=True)
with l4:
    st.markdown("<div class='level-box'><b>Target idea</b><br>" + fmt_px(float(latest["Target_idea"])) + "</div>", unsafe_allow_html=True)

# Alerts
flags = []
if latest.get("RSI_oversold") or latest.get("Near_low"):
    flags.append(("buy", f"Price is **low** — RSI {latest['RSI']:.0f}, BB {bbp:.0f}% → favor BUY LOW"))
if latest.get("RSI_overbought") or latest.get("Near_high"):
    flags.append(("sell", f"Price is **high** — RSI {latest['RSI']:.0f}, BB {bbp:.0f}% → favor SELL HIGH"))
if latest.get("MACD_cross_up"):
    flags.append(("buy", "MACD bullish cross (momentum turning up from low)"))
if latest.get("MACD_cross_dn"):
    flags.append(("sell", "MACD bearish cross (momentum turning down from high)"))
if latest.get("EMA_cross_up"):
    flags.append(("buy", "EMA cross up"))
if latest.get("EMA_cross_dn"):
    flags.append(("sell", "EMA cross down"))
if flags:
    st.subheader("🚨 Alerts")
    for kind, msg in flags:
        st.markdown(f"<div class='alert-box {'alert-buy' if kind=='buy' else 'alert-sell'}'>{msg}</div>", unsafe_allow_html=True)

st.divider()
st.subheader(f"📈 {name} · Price + EMAs")
tight_line_chart(df, ["Close", "EMA_fast", "EMA_slow"], height=380)

st.subheader("Bands & zones (low ↔ high)")
tight_line_chart(df, ["Close", "BB_l", "BB_u"], height=300)

a, b = st.columns(2)
with a:
    st.subheader("RSI")
    tight_line_chart(df, ["RSI"], height=220)
with b:
    st.subheader("Volume")
    tight_bar_chart(df, "Volume", height=220)

st.subheader("MACD histogram")
tight_bar_chart(df, "MACD_hist", height=180)

# Paper trades
st.subheader("📄 Paper trades (BUY → next SELL)")
pt = paper_trades(df_full.tail(min(len(df_full), 400)))
if pt.empty:
    st.info("No completed BUY→SELL pairs in this window.")
else:
    wins = (pt["Result"] == "WIN").sum()
    closed = pt[pt["Result"] != "OPEN"]
    total_pnl = closed["PnL %"].sum() if len(closed) else 0
    st.caption(f"Closed trades: {len(closed)} · Wins: {wins} · Sum PnL%: {total_pnl:+.2f}% (hypothetical)")
    st.dataframe(pt.sort_values("Entry time", ascending=False), use_container_width=True, height=240)

st.subheader("Recent bars")
st.dataframe(
    df[["Open", "High", "Low", "Close", "RSI", "BB_pct", "Score", "Signal"]]
      .assign(BB_pct=lambda x: (x["BB_pct"] * 100).round(0))
      .sort_index(ascending=False)
      .head(35),
    use_container_width=True,
    height=260,
)

with st.expander("How buy-low / sell-high works here"):
    st.markdown("""
    1. **Buy low** when RSI is washed out, price is near Bollinger/swing **low**, and momentum is not in free-fall (or is turning up).  
    2. **Sell high** when RSI is extended, price is near Bollinger/swing **high**, and momentum rolls over.  
    3. Trend filter (EMAs) reduces buying pure knives and selling pure rockets.  
    4. **Levels**: Buy zone ≈ max(BB low, swing low) · Sell zone ≈ min(BB high, swing high).  
    5. Paper table = enter every BUY, exit every next SELL on this timeframe (no fees/slippage).  

    **Not financial advice.** No live orders are placed. Use your own risk rules.
    """)

if live:
    time.sleep(secs)
    st.rerun()
