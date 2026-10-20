"""
Crypto Signal Bot Pro — Trader Decision Pack
============================================
Buy-low / sell-high with:
  • Multi-factor confluence score
  • ATR-based stops & targets (1.5R / 2.5R)
  • Hard trend filter (trade with higher-TF bias)
  • Risk limits: max risk %, min R:R, position size helper
  • Trade plan card + checklist for every decision

Data: Binance public (no key). Signals only — not live orders.
Not financial advice.
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
div[data-testid="stMetricValue"] { font-size: 1.25rem; }
.signal-buy  { color: #00c853; font-weight: 800; font-size: 1.5rem; }
.signal-sell { color: #ff1744; font-weight: 800; font-size: 1.5rem; }
.signal-hold { color: #ffab00; font-weight: 800; font-size: 1.5rem; }
.alert-box { padding: 10px 14px; border-radius: 8px; margin: 6px 0; font-weight: 600; }
.alert-buy { background: rgba(0,200,83,0.15); border-left: 4px solid #00c853; }
.alert-sell { background: rgba(255,23,68,0.15); border-left: 4px solid #ff1744; }
.plan-box { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
            padding: 14px 16px; border-radius: 10px; margin-bottom: 8px; }
.check-ok { color: #00c853; }
.check-no { color: #ff1744; }
.check-na { color: #888; }
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
# Higher TF for trend filter
HTF_MAP = {"1m": "15m", "3m": "15m", "5m": "1h", "15m": "1h", "30m": "4h", "1h": "4h", "4h": "1d", "1d": "1d"}
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

def atr(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def trend_bias(df: pd.DataFrame) -> str:
    """UP / DOWN / RANGE from EMA structure."""
    if df.empty or len(df) < 30:
        return "RANGE"
    c = df["Close"]
    e12, e26 = ema(c, 12), ema(c, 26)
    e50 = ema(c, 50) if len(df) >= 50 else e26
    last = c.iloc[-1]
    if e12.iloc[-1] > e26.iloc[-1] > e50.iloc[-1] and last > e26.iloc[-1]:
        return "UP"
    if e12.iloc[-1] < e26.iloc[-1] < e50.iloc[-1] and last < e26.iloc[-1]:
        return "DOWN"
    return "RANGE"


def add_signals(df: pd.DataFrame, htf_bias: str = "RANGE") -> pd.DataFrame:
    if df.empty or len(df) < 40:
        return pd.DataFrame()
    d = df.copy()
    c = d["Close"]
    d["EMA_fast"] = ema(c, 12)
    d["EMA_slow"] = ema(c, 26)
    d["EMA_50"] = ema(c, 50) if len(d) >= 50 else ema(c, max(20, len(d) // 3))
    d["RSI"] = rsi(c)
    d["MACD"] = d["EMA_fast"] - d["EMA_slow"]
    d["MACD_sig"] = ema(d["MACD"], 9)
    d["MACD_hist"] = d["MACD"] - d["MACD_sig"]
    d["ATR"] = atr(d, 14)

    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    d["BB_u"] = mid + 2 * std
    d["BB_mid"] = mid
    d["BB_l"] = mid - 2 * std
    d["BB_pct"] = (c - d["BB_l"]) / (d["BB_u"] - d["BB_l"]).replace(0, np.nan)
    d["BB_width"] = (d["BB_u"] - d["BB_l"]) / mid.replace(0, np.nan) * 100

    d["Vol_MA"] = d["Volume"].rolling(20).mean()
    d["Vol_ratio"] = d["Volume"] / d["Vol_MA"].replace(0, np.nan)
    d["Pct_chg"] = c.pct_change() * 100
    d["Swing_low"] = d["Low"].rolling(20).min()
    d["Swing_high"] = d["High"].rolling(20).max()

    d["MACD_cross_up"] = (d["MACD"] > d["MACD_sig"]) & (d["MACD"].shift() <= d["MACD_sig"].shift())
    d["MACD_cross_dn"] = (d["MACD"] < d["MACD_sig"]) & (d["MACD"].shift() >= d["MACD_sig"].shift())
    d["EMA_cross_up"] = (d["EMA_fast"] > d["EMA_slow"]) & (d["EMA_fast"].shift() <= d["EMA_slow"].shift())
    d["EMA_cross_dn"] = (d["EMA_fast"] < d["EMA_slow"]) & (d["EMA_fast"].shift() >= d["EMA_slow"].shift())
    d["RSI_oversold"] = d["RSI"] < 32
    d["RSI_overbought"] = d["RSI"] > 68
    d["Near_low"] = d["BB_pct"] < 0.22
    d["Near_high"] = d["BB_pct"] > 0.78
    d["Vol_spike"] = d["Vol_ratio"] > 1.8
    d["Local_up"] = d["EMA_fast"] > d["EMA_slow"]
    d["Local_dn"] = d["EMA_fast"] < d["EMA_slow"]

    # Raw edge score (before trend filter)
    score = pd.Series(0.0, index=d.index)
    score += np.where(d["RSI"] < 25, 28, 0)
    score += np.where((d["RSI"] >= 25) & (d["RSI"] < 35), 16, 0)
    score += np.where(d["RSI"] > 75, -28, 0)
    score += np.where((d["RSI"] <= 75) & (d["RSI"] > 65), -16, 0)
    score += np.where(d["BB_pct"] < 0.15, 20, 0)
    score += np.where(d["BB_pct"] > 0.85, -20, 0)
    score += np.where(d["MACD"] > d["MACD_sig"], 10, -10)
    score += np.where(d["MACD_cross_up"], 12, 0)
    score += np.where(d["MACD_cross_dn"], -12, 0)
    score += np.where(d["EMA_cross_up"], 10, 0)
    score += np.where(d["EMA_cross_dn"], -10, 0)
    score += np.where(d["Local_up"], 8, -8)
    score += np.where(d["Vol_spike"] & (d["Pct_chg"] > 0) & (d["BB_pct"] < 0.45), 8, 0)
    score += np.where(d["Vol_spike"] & (d["Pct_chg"] < 0) & (d["BB_pct"] > 0.55), -8, 0)
    d["Score_raw"] = score.clip(-100, 100)

    # ── TREND FILTER (hard) ───────────────────────────────────
    # Only allow BUY in UP or RANGE; SELL in DOWN or RANGE
    # In strong opposing trend, force HOLD and penalize score
    allowed_buy = htf_bias in ("UP", "RANGE")
    allowed_sell = htf_bias in ("DOWN", "RANGE")
    d["HTF_bias"] = htf_bias
    d["Score"] = d["Score_raw"].copy()
    if htf_bias == "DOWN":
        d["Score"] = d["Score"].clip(upper=25)  # cap long bias
    if htf_bias == "UP":
        d["Score"] = d["Score"].clip(lower=-25)

    def classify(row):
        s = row["Score"]
        buy_setup = s >= 36 or (s >= 28 and row["RSI"] < 36 and row["BB_pct"] < 0.35)
        sell_setup = s <= -36 or (s <= -28 and row["RSI"] > 64 and row["BB_pct"] > 0.65)
        if buy_setup and allowed_buy:
            return "BUY"
        if sell_setup and allowed_sell:
            return "SELL"
        return "HOLD"

    d["Signal"] = d.apply(classify, axis=1)

    # ── ATR stops & targets ───────────────────────────────────
    # Long: stop = entry - 1.5*ATR (or under swing low), T1 = +2R, T2 = +3R
    # Short: mirror
    atr_v = d["ATR"].replace(0, np.nan)
    d["Stop_long"] = np.minimum(d["Close"] - 1.5 * atr_v, d["Swing_low"] * 0.998)
    d["T1_long"] = d["Close"] + 2.0 * (d["Close"] - d["Stop_long"])
    d["T2_long"] = d["Close"] + 3.0 * (d["Close"] - d["Stop_long"])
    d["Stop_short"] = np.maximum(d["Close"] + 1.5 * atr_v, d["Swing_high"] * 1.002)
    d["T1_short"] = d["Close"] - 2.0 * (d["Stop_short"] - d["Close"])
    d["T2_short"] = d["Close"] - 3.0 * (d["Stop_short"] - d["Close"])

    # R:R at T1
    risk_l = (d["Close"] - d["Stop_long"]).replace(0, np.nan)
    risk_s = (d["Stop_short"] - d["Close"]).replace(0, np.nan)
    d["RR_long"] = ((d["T1_long"] - d["Close"]) / risk_l).round(2)
    d["RR_short"] = ((d["Close"] - d["T1_short"]) / risk_s).round(2)

    d["Buy_zone"] = d[["BB_l", "Swing_low"]].max(axis=1)
    d["Sell_zone"] = d[["BB_u", "Swing_high"]].min(axis=1)
    return d.dropna(subset=["RSI", "ATR"])


def build_trade_plan(row, side: str, account_usd: float, risk_pct: float, min_rr: float) -> dict:
    """Concrete plan a trader can use."""
    px = float(row["Close"])
    atr_v = float(row["ATR"]) if pd.notna(row["ATR"]) else px * 0.01
    if side == "BUY":
        stop = float(row["Stop_long"])
        t1, t2 = float(row["T1_long"]), float(row["T2_long"])
        rr = float(row["RR_long"]) if pd.notna(row["RR_long"]) else 0
        risk_per_unit = max(px - stop, atr_v * 0.5)
    else:
        stop = float(row["Stop_short"])
        t1, t2 = float(row["T1_short"]), float(row["T2_short"])
        rr = float(row["RR_short"]) if pd.notna(row["RR_short"]) else 0
        risk_per_unit = max(stop - px, atr_v * 0.5)

    risk_usd = account_usd * (risk_pct / 100.0)
    size = risk_usd / risk_per_unit if risk_per_unit > 0 else 0
    notional = size * px
    passes_rr = rr >= min_rr

    return {
        "side": side,
        "entry": px,
        "stop": stop,
        "t1": t1,
        "t2": t2,
        "rr": rr,
        "atr": atr_v,
        "risk_usd": risk_usd,
        "size": size,
        "notional": notional,
        "passes_rr": passes_rr,
        "risk_pct_of_entry": (risk_per_unit / px * 100) if px else 0,
    }


def confluence_checklist(row, htf_bias: str, side: str) -> list:
    """Trader checklist — each item True/False."""
    items = []
    if side == "BUY":
        items.append(("HTF bias allows longs", htf_bias in ("UP", "RANGE")))
        items.append(("Local trend not strongly down", bool(row.get("Local_up") or row["EMA_fast"] >= row["EMA_slow"] * 0.998)))
        items.append(("RSI not overbought (< 65)", float(row["RSI"]) < 65))
        items.append(("Near value / low (BB% < 40 or RSI < 40)", float(row["BB_pct"]) < 0.4 or float(row["RSI"]) < 40))
        items.append(("MACD supportive or crossing up", float(row["MACD"]) >= float(row["MACD_sig"]) or bool(row.get("MACD_cross_up"))))
        items.append(("R:R to T1 ≥ 1.5", float(row.get("RR_long") or 0) >= 1.5))
        items.append(("ATR defined (volatility readable)", pd.notna(row["ATR"]) and float(row["ATR"]) > 0))
    else:
        items.append(("HTF bias allows shorts", htf_bias in ("DOWN", "RANGE")))
        items.append(("Local trend not strongly up", bool(row.get("Local_dn") or row["EMA_fast"] <= row["EMA_slow"] * 1.002)))
        items.append(("RSI not oversold (> 35)", float(row["RSI"]) > 35))
        items.append(("Near premium / high (BB% > 60 or RSI > 60)", float(row["BB_pct"]) > 0.6 or float(row["RSI"]) > 60))
        items.append(("MACD supportive or crossing down", float(row["MACD"]) <= float(row["MACD_sig"]) or bool(row.get("MACD_cross_dn"))))
        items.append(("R:R to T1 ≥ 1.5", float(row.get("RR_short") or 0) >= 1.5))
        items.append(("ATR defined (volatility readable)", pd.notna(row["ATR"]) and float(row["ATR"]) > 0))
    return items


def paper_trades(df: pd.DataFrame) -> pd.DataFrame:
    trades = []
    in_pos = False
    entry = entry_t = stop = None
    side = None
    for ts, row in df.iterrows():
        if not in_pos and row["Signal"] in ("BUY", "SELL"):
            in_pos = True
            side = row["Signal"]
            entry = float(row["Close"])
            entry_t = ts
            stop = float(row["Stop_long"] if side == "BUY" else row["Stop_short"])
        elif in_pos:
            px = float(row["Close"])
            hit_stop = (side == "BUY" and px <= stop) or (side == "SELL" and px >= stop)
            opposite = row["Signal"] == ("SELL" if side == "BUY" else "BUY")
            if hit_stop or opposite:
                pnl = (px - entry) / entry * 100 if side == "BUY" else (entry - px) / entry * 100
                trades.append({
                    "Side": side,
                    "Entry time": entry_t,
                    "Entry": entry,
                    "Exit time": ts,
                    "Exit": px,
                    "Stop": stop,
                    "PnL %": round(pnl, 2),
                    "Exit reason": "STOP" if hit_stop else "SIGNAL",
                    "Result": "WIN" if pnl > 0 else "LOSS",
                })
                in_pos = False
    return pd.DataFrame(trades)


def snapshot(name, symbol, interval):
    raw = fetch_klines(symbol, interval, 180)
    htf_i = HTF_MAP.get(interval, "1h")
    htf_raw = fetch_klines(symbol, htf_i, 120)
    bias = trend_bias(htf_raw) if not htf_raw.empty else "RANGE"
    d = add_signals(raw, bias)
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
    rr = float(last["RR_long"]) if last["Signal"] == "BUY" else float(last["RR_short"]) if last["Signal"] == "SELL" else None
    return {
        "Asset": name,
        "HTF": bias,
        "Price": float(last["Close"]),
        "Chg%": round(chg, 2),
        "RSI": round(float(last["RSI"]), 1),
        "ATR%": round(float(last["ATR"]) / float(last["Close"]) * 100, 2) if last["Close"] else None,
        "Score": int(last["Score"]),
        "Signal": str(last["Signal"]),
        "R:R": rr,
        "Flags": " ".join(flags) if flags else "—",
        "Status": "ok",
    }


def fmt_px(x):
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


def tight_line_chart(df, cols, height=360, title=""):
    plot = df[cols].copy().reset_index()
    plot = plot.rename(columns={plot.columns[0]: "Time"})
    rename = {"EMA_fast": "EMA 12", "EMA_slow": "EMA 26", "EMA_50": "EMA 50",
              "BB_l": "BB Low", "BB_u": "BB High"}
    plot = plot.rename(columns=rename)
    cols_disp = [rename.get(c, c) for c in cols]
    long = plot.melt("Time", value_vars=cols_disp, var_name="Series", value_name="Value")
    ymin, ymax = float(long["Value"].min()), float(long["Value"].max())
    pad = (ymax - ymin) * 0.08 if ymax > ymin else max(abs(ymin) * 0.01, 1e-8)
    chart = (
        alt.Chart(long).mark_line(strokeWidth=2)
        .encode(
            x=alt.X("Time:T", title=None),
            y=alt.Y("Value:Q", title=None, scale=alt.Scale(domain=[ymin - pad, ymax + pad], zero=False)),
            color=alt.Color("Series:N", legend=alt.Legend(orient="top")),
            tooltip=["Time:T", "Series:N", alt.Tooltip("Value:Q", format=".6f")],
        )
        .properties(height=height, title=title).interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def tight_bar_chart(df, col, height=200):
    plot = df[[col]].copy().reset_index()
    plot = plot.rename(columns={plot.columns[0]: "Time", col: "Value"})
    ymin, ymax = float(plot["Value"].min()), float(plot["Value"].max())
    if ymin > 0 and ymin > (ymax - ymin) * 0.25:
        pad = (ymax - ymin) * 0.1
        domain, zero = [ymin - pad, ymax + pad], False
    else:
        domain, zero = [min(0, ymin), ymax * 1.08 if ymax else 1], True
    chart = (
        alt.Chart(plot).mark_bar(opacity=0.75)
        .encode(
            x=alt.X("Time:T", title=None),
            y=alt.Y("Value:Q", title=None, scale=alt.Scale(domain=domain, zero=zero)),
            tooltip=["Time:T", alt.Tooltip("Value:Q", format=".4f")],
        )
        .properties(height=height).interactive()
    )
    st.altair_chart(chart, use_container_width=True)


# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("🛠️ Control Room")
page = st.sidebar.radio("View", ["Single asset", "Multi-ticker scanner"], index=0)
interval = st.sidebar.selectbox("Candle size", INTERVALS, index=3)
limit = st.sidebar.select_slider("Candles shown", options=[50, 100, 150, 200, 300], value=100)

st.sidebar.markdown("---")
st.sidebar.subheader("Risk limits")
account_usd = st.sidebar.number_input("Account size (USD)", min_value=100.0, value=10000.0, step=100.0)
risk_pct = st.sidebar.slider("Risk per trade %", 0.25, 2.0, 0.5, 0.25)
min_rr = st.sidebar.slider("Min R:R to take trade", 1.0, 3.0, 1.5, 0.25)
st.sidebar.caption("Position size = (account × risk%) ÷ distance to stop")

st.sidebar.markdown("---")
st.sidebar.subheader("🔴 Live feed")
live = st.sidebar.checkbox("Auto-refresh ON", value=False)
secs = st.sidebar.slider("Refresh seconds", 10, 60, 20)
if st.sidebar.button("🔄 Refresh now", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# ══════════════════════════════════════════════════════════════
# SCANNER
# ══════════════════════════════════════════════════════════════
if page == "Multi-ticker scanner":
    st.title("📡 Scanner · Risk-filtered")
    st.caption(f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC · {interval} · HTF trend filter on")

    scan_names = st.multiselect("Assets", list(CRYPTO.keys()),
                                default=[n for n in DEFAULT_SCAN if n in CRYPTO])
    if not scan_names:
        st.warning("Select assets.")
        st.stop()

    rows = []
    prog = st.progress(0, text="Scanning…")
    for i, n in enumerate(scan_names):
        rows.append(snapshot(n, CRYPTO[n], interval))
        prog.progress((i + 1) / len(scan_names), text=n)
    prog.empty()

    ok = pd.DataFrame(rows)
    ok = ok[ok.get("Status", "ok") == "ok"].copy() if "Status" in ok.columns else ok

    st.subheader("🚨 Actionable (passes trend filter)")
    alerts = []
    if not ok.empty:
        for _, r in ok.iterrows():
            if r["Signal"] == "BUY" and (r.get("R:R") or 0) >= min_rr:
                alerts.append(("buy", f"**BUY** {r['Asset']} · HTF {r['HTF']} · R:R {r['R:R']} · RSI {r['RSI']} · {r['Flags']}"))
            elif r["Signal"] == "SELL" and (r.get("R:R") or 0) >= min_rr:
                alerts.append(("sell", f"**SELL** {r['Asset']} · HTF {r['HTF']} · R:R {r['R:R']} · RSI {r['RSI']} · {r['Flags']}"))
            elif r["Signal"] in ("BUY", "SELL"):
                alerts.append(("sell" if r["Signal"] == "SELL" else "buy",
                               f"{r['Signal']} {r['Asset']} but R:R {r.get('R:R')} < min {min_rr} — skip"))
    if not alerts:
        st.info("No trades pass trend filter + min R:R right now.")
    else:
        for kind, msg in alerts:
            st.markdown(f"<div class='alert-box {'alert-buy' if kind=='buy' else 'alert-sell'}'>{msg}</div>",
                        unsafe_allow_html=True)

    if not ok.empty:
        show = ok.drop(columns=["Status"], errors="ignore")
        order = {"BUY": 0, "HOLD": 1, "SELL": 2}
        show["_o"] = show["Signal"].map(order)
        show = show.sort_values(["_o", "Score"], ascending=[True, False]).drop(columns=["_o"])
        show["Price"] = show["Price"].map(fmt_px)
        st.dataframe(show, use_container_width=True, height=420)

    if live:
        time.sleep(secs)
        st.rerun()
    st.stop()

# ══════════════════════════════════════════════════════════════
# SINGLE ASSET — full trade decision pack
# ══════════════════════════════════════════════════════════════
st.sidebar.markdown("---")
name = st.sidebar.selectbox("Asset", list(CRYPTO.keys()), index=0)
symbol = CRYPTO[name]
custom = st.sidebar.text_input("Custom (e.g. ONDOUSDT)", "")
if custom.strip():
    symbol = custom.strip().upper()
    name = symbol

st.title("⚡ Trade Decision Pack")
st.caption(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC · {name} · {symbol} · {interval}")

htf_i = HTF_MAP.get(interval, "1h")
with st.spinner("Loading chart + higher-timeframe trend…"):
    raw = fetch_klines(symbol, interval, max(limit + 100, 180))
    htf_raw = fetch_klines(symbol, htf_i, 150)
    htf_bias = trend_bias(htf_raw) if not htf_raw.empty else "RANGE"
    df_full = add_signals(raw, htf_bias)

if df_full.empty:
    st.error("No data. Use symbols like BTCUSDT.")
    st.stop()

df = df_full.tail(limit).copy()
latest = df.iloc[-1]
prev = df.iloc[-2]
delta = float(latest["Close"] - prev["Close"])
pct = delta / float(prev["Close"]) * 100 if prev["Close"] else 0
price = float(latest["Close"])
sig = str(latest["Signal"])

st.success(f"Data OK · **{len(df)}** candles · HTF ({htf_i}) bias = **{htf_bias}**")

# Metrics
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Price", fmt_px(price), f"{pct:+.2f}%")
label = "BUY LOW" if sig == "BUY" else ("SELL HIGH" if sig == "SELL" else "HOLD")
cls = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}[sig]
c2.markdown("**Signal**")
c2.markdown(f"<div class='{cls}'>{label}</div>", unsafe_allow_html=True)
c3.metric("RSI", f"{float(latest['RSI']):.1f}")
c4.metric("Score", f"{int(latest['Score'])}")
c5.metric("ATR", fmt_px(float(latest["ATR"])))
c6.metric("HTF bias", htf_bias)

# ── TRADE PLAN ────────────────────────────────────────────────
st.markdown("### 📋 Trade plan")
if sig in ("BUY", "SELL"):
    plan = build_trade_plan(latest, sig, account_usd, risk_pct, min_rr)
    checks = confluence_checklist(latest, htf_bias, sig)
    n_ok = sum(1 for _, v in checks if v)
    conf = "HIGH" if n_ok >= 6 and plan["passes_rr"] else ("MED" if n_ok >= 4 else "LOW")

    p1, p2 = st.columns([1.2, 1])
    with p1:
        st.markdown(f"""
        <div class="plan-box">
        <b>Side:</b> {plan['side']}<br>
        <b>Entry:</b> {fmt_px(plan['entry'])}<br>
        <b>Stop:</b> {fmt_px(plan['stop'])} &nbsp; ({plan['risk_pct_of_entry']:.2f}% from entry)<br>
        <b>Target 1 (2R):</b> {fmt_px(plan['t1'])}<br>
        <b>Target 2 (3R):</b> {fmt_px(plan['t2'])}<br>
        <b>R:R to T1:</b> {plan['rr']:.2f} &nbsp;
        {"✅ passes min" if plan['passes_rr'] else "❌ below min R:R — skip"}<br>
        <b>Confidence:</b> {conf} ({n_ok}/7 checks)
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown(f"""
        <div class="plan-box">
        <b>Risk limits</b><br>
        Account: ${account_usd:,.0f}<br>
        Risk/trade: {risk_pct}% = <b>${plan['risk_usd']:,.2f}</b><br>
        Position size ≈ <b>{plan['size']:.6g}</b> units<br>
        Notional ≈ <b>${plan['notional']:,.2f}</b><br>
        ATR: {fmt_px(plan['atr'])}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Confluence checklist")
    for text, ok in checks:
        mark = "<span class='check-ok'>✓</span>" if ok else "<span class='check-no'>✗</span>"
        st.markdown(f"{mark} {text}", unsafe_allow_html=True)

    if not plan["passes_rr"]:
        st.warning(f"R:R {plan['rr']:.2f} < your minimum {min_rr}. Rules say **do not take** this trade.")
    elif conf == "LOW":
        st.warning("Low confluence — prefer skip or reduce size.")
    elif sig == "BUY" and htf_bias == "DOWN":
        st.error("Trend filter blocked aggressive longs — should be HOLD. Re-check data.")
    elif sig == "SELL" and htf_bias == "UP":
        st.error("Trend filter blocked aggressive shorts — should be HOLD.")
else:
    st.info(
        f"**HOLD** — no buy-low / sell-high setup that passes the trend filter "
        f"(HTF={htf_bias}). Wait for price at extremes + alignment."
    )
    st.markdown(
        f"Closest levels → Buy zone {fmt_px(float(latest['Buy_zone']))} · "
        f"Sell zone {fmt_px(float(latest['Sell_zone']))} · "
        f"ATR {fmt_px(float(latest['ATR']))}"
    )

# Alerts strip
flags = []
if latest.get("RSI_oversold") or latest.get("Near_low"):
    flags.append(("buy", f"At lows — RSI {latest['RSI']:.0f}, BB% {float(latest['BB_pct'])*100:.0f}"))
if latest.get("RSI_overbought") or latest.get("Near_high"):
    flags.append(("sell", f"At highs — RSI {latest['RSI']:.0f}, BB% {float(latest['BB_pct'])*100:.0f}"))
if latest.get("MACD_cross_up"):
    flags.append(("buy", "MACD cross up"))
if latest.get("MACD_cross_dn"):
    flags.append(("sell", "MACD cross down"))
if flags:
    st.subheader("Alerts")
    for k, m in flags:
        st.markdown(f"<div class='alert-box {'alert-buy' if k=='buy' else 'alert-sell'}'>{m}</div>",
                    unsafe_allow_html=True)

st.divider()
st.subheader(f"📈 {name} · Close + EMAs")
tight_line_chart(df, ["Close", "EMA_fast", "EMA_slow", "EMA_50"], height=360)

st.subheader("Bollinger (value vs premium)")
tight_line_chart(df, ["Close", "BB_l", "BB_u"], height=280)

a, b = st.columns(2)
with a:
    st.subheader("RSI")
    tight_line_chart(df, ["RSI"], height=220)
with b:
    st.subheader("Volume")
    tight_bar_chart(df, "Volume", height=220)

st.subheader("MACD histogram")
tight_bar_chart(df, "MACD_hist", height=180)

# Paper with stops
st.subheader("📄 Paper trades (signal entry · stop or reverse exit)")
pt = paper_trades(df_full.tail(min(len(df_full), 500)))
if pt.empty:
    st.info("No closed paper trades in window.")
else:
    closed = pt[pt["Result"].isin(["WIN", "LOSS"])]
    wins = (closed["Result"] == "WIN").sum()
    st.caption(
        f"Closed: {len(closed)} · Wins: {wins} · "
        f"Win rate: {wins/len(closed)*100:.0f}%" if len(closed) else "—"
        + f" · Sum PnL%: {closed['PnL %'].sum():+.2f}%" if len(closed) else ""
    )
    st.dataframe(pt.sort_values("Entry time", ascending=False), use_container_width=True, height=240)

st.subheader("Recent bars")
st.dataframe(
    df[["Open", "High", "Low", "Close", "RSI", "ATR", "Score", "Signal", "RR_long", "RR_short"]]
      .sort_index(ascending=False).head(30),
    use_container_width=True, height=260,
)

with st.expander("Rules summary (what a trader uses)"):
    st.markdown(f"""
### Trend filter
- Higher timeframe used: **{htf_i}** → bias **{htf_bias}**
- **BUY** only if HTF is UP or RANGE  
- **SELL** only if HTF is DOWN or RANGE  
- Against-trend signals are capped / forced to HOLD  

### Entry (buy low / sell high)
- RSI extremes + Bollinger position + MACD/EMA alignment  
- Score threshold with confluence  

### Stops & targets (ATR)
- Stop ≈ 1.5 × ATR from entry (and outside swing)  
- **T1 = 2R**, **T2 = 3R**  
- Skip if R:R to T1 **< {min_rr}** (your setting)  

### Risk
- Risk **{risk_pct}%** of ${account_usd:,.0f} = **${account_usd * risk_pct / 100:,.2f}** per trade  
- Size = risk $ ÷ stop distance  

### Discipline
- No trade if checklist confidence LOW or R:R fails  
- Paper table includes **stop-outs**, not only signal flips  

*Educational tool only. Not financial advice. No live orders.*
""")

if live:
    time.sleep(secs)
    st.rerun()
