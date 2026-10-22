"""
Crypto Signal Bot — Simple Spot
===============================
Buy low / sell high · spot only · no shorts · no leverage
Shows a clear ENTRY price and EXIT price.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timezone
from pathlib import Path
import requests
import time
import json

st.set_page_config(page_title="Buy Low Sell High", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.big-price { font-size: 2rem; font-weight: 800; margin: 0.2rem 0; }
.entry { color: #00c853; }
.exit { color: #ffab00; }
.hold { color: #90a4ae; }
.card {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 12px;
  padding: 1.25rem 1.5rem;
  margin-bottom: 0.75rem;
}
.hint { color: #9e9e9e; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
TRADES_PATH = DATA_DIR / "trades.csv"
PARAMS_PATH = DATA_DIR / "learned_params.json"

DEFAULT_PARAMS = {
    "buy_score_min": 36,
    "rsi_buy_max": 38,
    "bb_buy_max": 0.38,
    "min_rr": 1.5,
    "atr_stop_mult": 1.5,
    "sell_rsi_min": 62,
    "sell_bb_min": 0.72,
    "optimized_at": None,
    "n_trades_used": 0,
}

CRYPTO = {
    "Bitcoin": "BTCUSDT", "Ethereum": "ETHUSDT", "Solana": "SOLUSDT",
    "BNB": "BNBUSDT", "XRP": "XRPUSDT", "Cardano": "ADAUSDT",
    "Dogecoin": "DOGEUSDT", "Avalanche": "AVAXUSDT", "Chainlink": "LINKUSDT",
    "Polkadot": "DOTUSDT", "Litecoin": "LTCUSDT", "Shiba Inu": "SHIBUSDT",
    "ONDO": "ONDOUSDT", "Stellar": "XLMUSDT", "Hedera": "HBARUSDT",
    "Algorand": "ALGOUSDT", "Flare": "FLRUSDT", "Sui": "SUIUSDT",
    "Near": "NEARUSDT", "Render": "RENDERUSDT", "Pepe": "PEPEUSDT",
    "Toncoin": "TONUSDT", "Tron": "TRXUSDT", "Aave": "AAVEUSDT",
}
INTERVALS = ["5m", "15m", "30m", "1h", "4h", "1d"]
HTF_MAP = {"5m": "1h", "15m": "1h", "30m": "4h", "1h": "4h", "4h": "1d", "1d": "1d"}
HOSTS = ["https://data-api.binance.vision", "https://api.binance.us"]


def load_params():
    p = DEFAULT_PARAMS.copy()
    if PARAMS_PATH.exists():
        try:
            p.update(json.loads(PARAMS_PATH.read_text()))
        except Exception:
            pass
    if "learned_params" in st.session_state:
        p.update(st.session_state["learned_params"])
    return p


def save_params(params):
    st.session_state["learned_params"] = params
    try:
        PARAMS_PATH.write_text(json.dumps(params, indent=2))
    except Exception:
        pass


def load_trades():
    frames = []
    if TRADES_PATH.exists():
        try:
            frames.append(pd.read_csv(TRADES_PATH))
        except Exception:
            pass
    if st.session_state.get("trade_log"):
        frames.append(pd.DataFrame(st.session_state["trade_log"]))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "entry_time" in df.columns:
        df = df.drop_duplicates(subset=["symbol", "entry_time", "exit_time"], keep="last")
    return df


def persist_trades(df):
    st.session_state["trade_log"] = df.to_dict("records")
    try:
        df.to_csv(TRADES_PATH, index=False)
    except Exception:
        pass


def append_trades(rows):
    if not rows:
        return
    old = load_trades()
    combined = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    if "entry_time" in combined.columns:
        combined = combined.drop_duplicates(subset=["symbol", "entry_time", "exit_time"], keep="last")
    persist_trades(combined)


@st.cache_data(ttl=20, show_spinner=False)
def fetch_klines(symbol, interval, limit=300):
    params = {"symbol": symbol, "interval": interval, "limit": min(int(limit), 1000)}
    for host in HOSTS:
        try:
            r = requests.get(f"{host}/api/v3/klines", params=params, timeout=10)
            if r.status_code != 200:
                continue
            raw = r.json()
            if not isinstance(raw, list) or len(raw) < 20:
                continue
            df = pd.DataFrame(raw, columns=[
                "t", "Open", "High", "Low", "Close", "Volume",
                "T", "q", "n", "V", "Q", "x"
            ])
            df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_localize(None)
            df = df.set_index("t")
            for c in ["Open", "High", "Low", "Close", "Volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if len(df) >= 20:
                return df
        except Exception:
            continue
    return pd.DataFrame()


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, min_periods=n).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, min_periods=n).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

def atr_series(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def trend_bias(df):
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


def analyze(df, htf_bias, params):
    if df.empty or len(df) < 40:
        return pd.DataFrame()
    d = df.copy()
    c = d["Close"]
    d["EMA_fast"] = ema(c, 12)
    d["EMA_slow"] = ema(c, 26)
    d["RSI"] = rsi(c)
    d["ATR"] = atr_series(d, 14)
    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    d["BB_u"] = mid + 2 * std
    d["BB_l"] = mid - 2 * std
    d["BB_pct"] = (c - d["BB_l"]) / (d["BB_u"] - d["BB_l"]).replace(0, np.nan)
    d["Swing_low"] = d["Low"].rolling(20).min()
    d["Swing_high"] = d["High"].rolling(20).max()
    d["MACD"] = d["EMA_fast"] - d["EMA_slow"]
    d["MACD_sig"] = ema(d["MACD"], 9)
    d["Local_up"] = d["EMA_fast"] > d["EMA_slow"]

    score = pd.Series(0.0, index=d.index)
    score += np.where(d["RSI"] < 25, 30, 0)
    score += np.where((d["RSI"] >= 25) & (d["RSI"] < 38), 18, 0)
    score += np.where(d["BB_pct"] < 0.2, 22, 0)
    score += np.where(d["MACD"] > d["MACD_sig"], 10, -6)
    score += np.where(d["Local_up"], 10, -8)
    if htf_bias == "DOWN":
        score -= 20
    elif htf_bias == "UP":
        score += 10
    d["Score"] = score.clip(-100, 100)

    buy_min = float(params.get("buy_score_min", 36))
    rsi_buy = float(params.get("rsi_buy_max", 38))
    bb_buy = float(params.get("bb_buy_max", 0.38))
    sell_rsi = float(params.get("sell_rsi_min", 62))
    sell_bb = float(params.get("sell_bb_min", 0.72))
    allow_buy = htf_bias in ("UP", "RANGE")
    mult = float(params.get("atr_stop_mult", 1.5))

    # Levels for every bar
    d["Entry"] = d["Close"]  # buy at market when signal fires
    d["Stop"] = np.minimum(d["Close"] - mult * d["ATR"], d["Swing_low"] * 0.998)
    risk = (d["Close"] - d["Stop"]).replace(0, np.nan)
    d["Exit"] = d["Close"] + 2.0 * risk  # main take-profit (sell high target)
    d["Exit_stretch"] = d["Close"] + 3.0 * risk

    def classify(row):
        sell_high = (
            float(row["RSI"]) >= sell_rsi
            or float(row["BB_pct"]) >= sell_bb
        )
        buy_low = allow_buy and (
            float(row["Score"]) >= buy_min
            or (float(row["RSI"]) <= rsi_buy and float(row["BB_pct"]) <= bb_buy and float(row["Score"]) >= buy_min - 12)
        )
        if buy_low and not sell_high:
            return "BUY"
        if sell_high:
            return "SELL"
        return "HOLD"

    d["Signal"] = d.apply(classify, axis=1)
    return d.dropna(subset=["RSI", "ATR"])


def fmt(x):
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


def paper_spot(df, asset, symbol, interval, htf):
    rows = []
    in_pos = False
    entry = entry_t = stop = target = None
    for ts, row in df.iterrows():
        if not in_pos and row["Signal"] == "BUY":
            in_pos = True
            entry = float(row["Close"])
            entry_t = ts
            stop = float(row["Stop"])
            target = float(row["Exit"])
        elif in_pos:
            px = float(row["Close"])
            if px <= stop or row["Signal"] == "SELL" or px >= target:
                reason = "STOP" if px <= stop else ("TARGET" if px >= target else "SELL")
                pnl = (px - entry) / entry * 100
                rows.append({
                    "logged_at": datetime.now(timezone.utc).isoformat(),
                    "asset": asset, "symbol": symbol, "interval": interval,
                    "side": "SPOT_BUY",
                    "entry_time": str(entry_t), "exit_time": str(ts),
                    "entry": entry, "exit": px, "stop": stop,
                    "pnl_pct": round(pnl, 3),
                    "exit_reason": reason,
                    "result": "WIN" if pnl > 0 else "LOSS",
                    "htf_bias": htf,
                })
                in_pos = False
    return rows


def optimize(trades, current):
    t = trades.copy()
    if t.empty or "pnl_pct" not in t.columns:
        return current, "No trades yet."
    t = t[t["result"].isin(["WIN", "LOSS"])] if "result" in t.columns else t
    if len(t) < 12:
        return current, f"Need 12+ closed trades (have {len(t)})."
    new_p = current.copy()
    notes = [f"n={len(t)} avg PnL%={t['pnl_pct'].mean():+.2f}%"]
    new_p["optimized_at"] = datetime.now(timezone.utc).isoformat()
    new_p["n_trades_used"] = len(t)
    # Mild auto-tune: if avg negative, be pickier on buys
    if t["pnl_pct"].mean() < 0:
        new_p["buy_score_min"] = min(50, current.get("buy_score_min", 36) + 2)
        new_p["rsi_buy_max"] = max(30, current.get("rsi_buy_max", 38) - 2)
        notes.append("Results weak → stricter buy filters")
    else:
        notes.append("Results OK → keep thresholds")
    return new_p, " | ".join(notes)


def price_chart(df):
    plot = df[["Close"]].copy().reset_index()
    plot.columns = ["Time", "Price"]
    ymin, ymax = float(plot["Price"].min()), float(plot["Price"].max())
    pad = (ymax - ymin) * 0.08 or 1
    chart = (
        alt.Chart(plot)
        .mark_line(color="#42a5f5", strokeWidth=2)
        .encode(
            x=alt.X("Time:T", title=None),
            y=alt.Y("Price:Q", title=None, scale=alt.Scale(domain=[ymin - pad, ymax + pad], zero=False)),
            tooltip=["Time:T", alt.Tooltip("Price:Q", format=".4f")],
        )
        .properties(height=320)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


# ── UI ────────────────────────────────────────────────────────
params = load_params()

st.sidebar.title("Setup")
name = st.sidebar.selectbox("Coin", list(CRYPTO.keys()), index=0)
symbol = CRYPTO[name]
custom = st.sidebar.text_input("Or type pair", placeholder="BTCUSDT")
if custom.strip():
    symbol = custom.strip().upper()
    name = symbol

interval = st.sidebar.selectbox("Timeframe", INTERVALS, index=1)
account = st.sidebar.number_input("Account $", min_value=100.0, value=10000.0, step=100.0)
risk_pct = st.sidebar.slider("Risk % per trade", 0.25, 2.0, 0.5, 0.25)

page = st.sidebar.radio("Page", ["Trade", "History & learn"], index=0)
if st.sidebar.button("Refresh"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption("Spot only: buy coin low, sell coin high. No shorts. No leverage.")

# ── History page ──────────────────────────────────────────────
if page == "History & learn":
    st.title("History & learn")
    log = load_trades()
    up = st.file_uploader("Import CSV", type=["csv"])
    if up is not None:
        try:
            incoming = pd.read_csv(up)
            merged = pd.concat([log, incoming], ignore_index=True)
            persist_trades(merged)
            st.success("Imported")
            st.rerun()
        except Exception as e:
            st.error(e)

    if log.empty:
        st.info("No history yet. Use the Trade page — paper trades save automatically.")
    else:
        closed = log[log["result"].isin(["WIN", "LOSS"])] if "result" in log.columns else log
        if len(closed):
            c1, c2, c3 = st.columns(3)
            c1.metric("Trades", len(closed))
            c2.metric("Win rate", f"{(closed['result']=='WIN').mean()*100:.0f}%")
            c3.metric("Avg PnL %", f"{closed['pnl_pct'].mean():+.2f}%")
        st.dataframe(log.tail(100), use_container_width=True, height=320)
        st.download_button("Download CSV", log.to_csv(index=False), "spot_trades.csv", "text/csv")

    if st.button("Improve rules from history"):
        new_p, msg = optimize(log, params)
        save_params(new_p)
        st.success(msg)
    st.stop()

# ── Trade page (simple) ───────────────────────────────────────
st.title(f"{name}")
st.caption(f"Spot · {interval} · {datetime.now(timezone.utc).strftime('%H:%M UTC')}")

htf_i = HTF_MAP.get(interval, "1h")
with st.spinner("Loading prices…"):
    raw = fetch_klines(symbol, interval, 250)
    htf_raw = fetch_klines(symbol, htf_i, 120)
    htf = trend_bias(htf_raw) if not htf_raw.empty else "RANGE"
    df = analyze(raw, htf, params)

if df.empty:
    st.error("Could not load data. Try another coin or timeframe.")
    st.stop()

last = df.iloc[-1]
price = float(last["Close"])
sig = str(last["Signal"])
entry = float(last["Entry"])
exit_px = float(last["Exit"])
stop = float(last["Stop"])
risk_amt = price - stop
size = (account * risk_pct / 100.0) / risk_amt if risk_amt > 0 else 0

# Paper log
rows = paper_spot(df.tail(300), name, symbol, interval, htf)
append_trades(rows)

# ── BIG SIMPLE ANSWER ─────────────────────────────────────────
if sig == "BUY":
    st.markdown(
        f"""
        <div class="card">
          <div class="hint">RIGHT NOW</div>
          <div class="big-price entry">BUY</div>
          <p style="margin-top:1rem;font-size:1.15rem">
            <b>Entry price</b> (buy here)<br>
            <span class="big-price entry">{fmt(entry)}</span>
          </p>
          <p style="font-size:1.15rem">
            <b>Exit price</b> (sell target)<br>
            <span class="big-price exit">{fmt(exit_px)}</span>
          </p>
          <p class="hint">
            Stop if wrong: {fmt(stop)} · About {size:.6g} coins if you risk {risk_pct}% of ${account:,.0f}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.success("Buy the coin with cash at entry. Sell later at exit (or if price hits stop).")

elif sig == "SELL":
    st.markdown(
        f"""
        <div class="card">
          <div class="hint">RIGHT NOW</div>
          <div class="big-price exit">SELL</div>
          <p style="margin-top:1rem;font-size:1.15rem">
            <b>Exit price</b> (sell here if you hold the coin)<br>
            <span class="big-price exit">{fmt(price)}</span>
          </p>
          <p class="hint">
            Next buy zone if you want back in later ≈ {fmt(float(last['BB_l']))}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning("Sell the coin for cash. Do not short.")

else:
    st.markdown(
        f"""
        <div class="card">
          <div class="hint">RIGHT NOW</div>
          <div class="big-price hold">WAIT</div>
          <p style="margin-top:1rem;font-size:1.1rem">
            Current price: <b>{fmt(price)}</b>
          </p>
          <p>
            <b>When it becomes a buy, entry ≈</b> near {fmt(float(last['BB_l']))}<br>
            <b>When it becomes a sell, exit ≈</b> near {fmt(float(last['BB_u']))}
          </p>
          <p class="hint">No clear buy-low or sell-high setup yet.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Small context
c1, c2, c3 = st.columns(3)
c1.metric("Now", fmt(price))
c2.metric("RSI", f"{float(last['RSI']):.0f}")
c3.metric("Trend", htf)

st.subheader("Price")
price_chart(df.tail(120))

with st.expander("How to use (3 steps)"):
    st.markdown("""
1. Pick a **coin** and **timeframe** in the sidebar (15m or 1h is fine).  
2. Read the big box:
   - **BUY** → entry price = where to buy · exit price = where to sell  
   - **SELL** → sell the coin you already bought at that price  
   - **WAIT** → do nothing  
3. On your exchange, place a **spot** buy/sell yourself. This app does not trade for you.

Optional: open **History & learn** after many trades to tighten rules. Export CSV on Streamlit Cloud so history isn’t lost.
""")

st.caption("Spot only · Not financial advice · No automatic orders")
