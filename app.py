"""
Crypto Signal Bot Pro — SPOT ONLY: Buy Low, Sell High
=====================================================
• Spot only: cash → buy coin when low → sell coin when high → cash
• No short selling
• No leveraged positions — only owning the asset or cash (USDT)
• Trade log + stats + optimize (Cloud: export/import CSV)

Not financial advice. No live orders.
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

st.set_page_config(page_title="Buy Low Sell High · Spot Only", page_icon="⚡", layout="wide")

st.markdown("""
<style>
div[data-testid="stMetricValue"] { font-size: 1.2rem; }
.signal-buy  { color: #00c853; font-weight: 800; font-size: 1.5rem; }
.signal-sell { color: #ffab00; font-weight: 800; font-size: 1.45rem; }
.signal-hold { color: #90a4ae; font-weight: 800; font-size: 1.45rem; }
.alert-box { padding: 10px 14px; border-radius: 8px; margin: 6px 0; font-weight: 600; }
.alert-buy { background: rgba(0,200,83,0.15); border-left: 4px solid #00c853; }
.alert-sell { background: rgba(255,171,0,0.15); border-left: 4px solid #ffab00; }
.plan-box { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
            padding: 14px 16px; border-radius: 10px; margin-bottom: 8px; }
.check-ok { color: #00c853; } .check-no { color: #ff1744; }
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
    "sell_rsi_min": 62,      # take-profit / sell-high zone
    "sell_bb_min": 0.72,
    "optimized_at": None,
    "n_trades_used": 0,
}

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
HTF_MAP = {"1m": "15m", "3m": "15m", "5m": "1h", "15m": "1h", "30m": "4h", "1h": "4h", "4h": "1d", "1d": "1d"}
HOSTS = ["https://data-api.binance.vision", "https://api.binance.us"]
DEFAULT_SCAN = [
    "Bitcoin", "Ethereum", "Solana", "XRP", "ONDO", "Stellar",
    "Hedera", "Algorand", "Flare", "Dogecoin", "Cardano", "Chainlink",
]


def load_params() -> dict:
    p = DEFAULT_PARAMS.copy()
    if PARAMS_PATH.exists():
        try:
            p.update(json.loads(PARAMS_PATH.read_text()))
        except Exception:
            pass
    if "learned_params" in st.session_state:
        p.update(st.session_state["learned_params"])
    # strip any legacy short keys
    p.pop("sell_score_max", None)
    return p


def save_params(params: dict):
    st.session_state["learned_params"] = params
    try:
        PARAMS_PATH.write_text(json.dumps(params, indent=2))
    except Exception:
        pass


def load_trades() -> pd.DataFrame:
    cols = [
        "logged_at", "asset", "symbol", "interval", "side", "entry_time", "exit_time",
        "entry", "exit", "stop", "pnl_pct", "exit_reason", "result",
        "rsi", "score", "bb_pct", "htf_bias", "rr", "confidence", "atr",
    ]
    frames = []
    if TRADES_PATH.exists():
        try:
            frames.append(pd.read_csv(TRADES_PATH))
        except Exception:
            pass
    if "trade_log" in st.session_state and st.session_state["trade_log"]:
        frames.append(pd.DataFrame(st.session_state["trade_log"]))
    if not frames:
        return pd.DataFrame(columns=cols)
    df = pd.concat(frames, ignore_index=True)
    # spot only history
    if "side" in df.columns:
        df = df[df["side"].astype(str).str.upper().isin(["BUY", "LONG", "nan"]) | df["side"].isna()]
        df["side"] = "BUY"
    if "entry_time" in df.columns and "symbol" in df.columns:
        df = df.drop_duplicates(subset=["symbol", "entry_time", "exit_time"], keep="last")
    return df


def persist_trades(df: pd.DataFrame):
    st.session_state["trade_log"] = df.to_dict("records")
    try:
        DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(TRADES_PATH, index=False)
    except Exception:
        pass


def append_trades(new_rows: list):
    if not new_rows:
        return
    old = load_trades()
    add = pd.DataFrame(new_rows)
    combined = pd.concat([old, add], ignore_index=True)
    if "entry_time" in combined.columns:
        combined = combined.drop_duplicates(subset=["symbol", "entry_time", "exit_time"], keep="last")
    persist_trades(combined)


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
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def trend_bias(df: pd.DataFrame) -> str:
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


def add_signals(df: pd.DataFrame, htf_bias: str, params: dict) -> pd.DataFrame:
    """Spot only: BUY_LOW | SELL_HIGH (sell coin) | HOLD. No shorts. No leverage."""
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
    d["Vol_MA"] = d["Volume"].rolling(20).mean()
    d["Vol_ratio"] = d["Volume"] / d["Vol_MA"].replace(0, np.nan)
    d["Pct_chg"] = c.pct_change() * 100
    d["Swing_low"] = d["Low"].rolling(20).min()
    d["Swing_high"] = d["High"].rolling(20).max()

    d["MACD_cross_up"] = (d["MACD"] > d["MACD_sig"]) & (d["MACD"].shift() <= d["MACD_sig"].shift())
    d["EMA_cross_up"] = (d["EMA_fast"] > d["EMA_slow"]) & (d["EMA_fast"].shift() <= d["EMA_slow"].shift())
    d["Local_up"] = d["EMA_fast"] > d["EMA_slow"]
    d["Near_low"] = d["BB_pct"] < 0.25
    d["Near_high"] = d["BB_pct"] > 0.75
    d["RSI_oversold"] = d["RSI"] < 35
    d["RSI_overbought"] = d["RSI"] > 65
    d["Vol_spike"] = d["Vol_ratio"] > 1.8

    # Score: higher = better spot buy (buy low)
    score = pd.Series(0.0, index=d.index)
    score += np.where(d["RSI"] < 25, 30, 0)
    score += np.where((d["RSI"] >= 25) & (d["RSI"] < 38), 18, 0)
    score += np.where(d["RSI"] > 70, -15, 0)  # not a buy zone
    score += np.where(d["BB_pct"] < 0.18, 22, 0)
    score += np.where(d["BB_pct"] > 0.85, -12, 0)
    score += np.where(d["MACD"] > d["MACD_sig"], 10, -6)
    score += np.where(d["MACD_cross_up"], 14, 0)
    score += np.where(d["EMA_cross_up"], 12, 0)
    score += np.where(d["Local_up"], 10, -8)
    score += np.where(d["Vol_spike"] & (d["Pct_chg"] > 0) & (d["BB_pct"] < 0.45), 8, 0)
    # HTF: penalize buys in downtrend
    if htf_bias == "DOWN":
        score -= 20
    elif htf_bias == "UP":
        score += 10
    d["Score"] = score.clip(-100, 100)
    d["HTF_bias"] = htf_bias

    buy_min = float(params.get("buy_score_min", 36))
    rsi_buy = float(params.get("rsi_buy_max", 38))
    bb_buy = float(params.get("bb_buy_max", 0.38))
    sell_rsi = float(params.get("sell_rsi_min", 62))
    sell_bb = float(params.get("sell_bb_min", 0.72))

    # Block new buys in hard DOWN (don't catch knives without HTF support)
    allow_buy = htf_bias in ("UP", "RANGE")

    def classify(row):
        # SELL_HIGH = sell coin for cash (spot exit), never open a short
        sell_high = (
            float(row["RSI"]) >= sell_rsi
            or float(row["BB_pct"]) >= sell_bb
            or (float(row["RSI"]) > 58 and float(row["BB_pct"]) > 0.65 and float(row["MACD"]) < float(row["MACD_sig"]))
        )
        buy_low = allow_buy and (
            float(row["Score"]) >= buy_min
            or (
                float(row["Score"]) >= buy_min - 10
                and float(row["RSI"]) <= rsi_buy
                and float(row["BB_pct"]) <= bb_buy
            )
        )
        if buy_low and not sell_high:
            return "BUY"
        if sell_high:
            return "SELL_HIGH"  # sell spot inventory — NOT a short
        return "HOLD"

    d["Signal"] = d.apply(classify, axis=1)

    mult = float(params.get("atr_stop_mult", 1.5))
    atr_v = d["ATR"].replace(0, np.nan)
    d["Stop"] = np.minimum(d["Close"] - mult * atr_v, d["Swing_low"] * 0.998)
    risk = (d["Close"] - d["Stop"]).replace(0, np.nan)
    d["T1"] = d["Close"] + 2.0 * risk
    d["T2"] = d["Close"] + 3.0 * risk
    d["RR"] = (2.0 * risk / risk).round(2)  # always ~2R to T1 by construction
    d["RR"] = ((d["T1"] - d["Close"]) / risk).round(2)
    d["Buy_zone"] = d[["BB_l", "Swing_low"]].max(axis=1)
    d["Sell_zone"] = d[["BB_u", "Swing_high"]].min(axis=1)
    return d.dropna(subset=["RSI", "ATR"])


def confluence_checklist(row, htf_bias, min_rr):
    return [
        ("HTF allows buys (UP or RANGE)", htf_bias in ("UP", "RANGE")),
        ("Not strong local downtrend", bool(row.get("Local_up") or row["EMA_fast"] >= row["EMA_slow"] * 0.997)),
        ("RSI in buy-low area (< 45)", float(row["RSI"]) < 45),
        ("Near value (BB% < 45)", float(row["BB_pct"]) < 0.45),
        ("MACD not strongly against", float(row["MACD"]) >= float(row["MACD_sig"]) * 0.5 or bool(row.get("MACD_cross_up"))),
        (f"R:R to T1 ≥ {min_rr}", float(row.get("RR") or 0) >= min_rr),
        ("ATR readable", pd.notna(row["ATR"]) and float(row["ATR"]) > 0),
    ]


def build_plan(row, account_usd, risk_pct, min_rr):
    px = float(row["Close"])
    stop = float(row["Stop"])
    t1, t2 = float(row["T1"]), float(row["T2"])
    rr = float(row["RR"]) if pd.notna(row["RR"]) else 0
    atr_v = float(row["ATR"]) if pd.notna(row["ATR"]) else px * 0.01
    risk_per = max(px - stop, atr_v * 0.5)
    risk_usd = account_usd * (risk_pct / 100.0)
    size = risk_usd / risk_per if risk_per > 0 else 0
    return {
        "entry": px, "stop": stop, "t1": t1, "t2": t2, "rr": rr, "atr": atr_v,
        "risk_usd": risk_usd, "size": size, "notional": size * px,
        "passes_rr": rr >= min_rr, "risk_pct": (risk_per / px * 100) if px else 0,
    }


def simulate_longs(df, asset, symbol, interval, htf_bias, min_rr):
    """Enter on BUY, exit on SELL_HIGH or stop. Spot only."""
    rows = []
    in_pos = False
    entry = entry_t = stop = None
    meta = {}
    for ts, row in df.iterrows():
        sig = row["Signal"]
        if not in_pos and sig == "BUY":
            rr = float(row["RR"]) if pd.notna(row["RR"]) else 0
            if rr < min_rr:
                continue
            checks = confluence_checklist(row, htf_bias, min_rr)
            n_ok = sum(1 for _, v in checks if v)
            conf = "HIGH" if n_ok >= 6 else ("MED" if n_ok >= 4 else "LOW")
            in_pos = True
            entry = float(row["Close"])
            entry_t = ts
            stop = float(row["Stop"])
            meta = {
                "rsi": float(row["RSI"]), "score": float(row["Score"]),
                "bb_pct": float(row["BB_pct"]) if pd.notna(row["BB_pct"]) else None,
                "rr": rr, "confidence": conf,
                "atr": float(row["ATR"]) if pd.notna(row["ATR"]) else None,
            }
        elif in_pos:
            px = float(row["Close"])
            hit_stop = px <= stop
            sell_high = sig == "SELL_HIGH"
            if hit_stop or sell_high:
                pnl = (px - entry) / entry * 100
                rows.append({
                    "logged_at": datetime.now(timezone.utc).isoformat(),
                    "asset": asset, "symbol": symbol, "interval": interval,
                    "side": "BUY",
                    "entry_time": str(entry_t), "exit_time": str(ts),
                    "entry": entry, "exit": px, "stop": stop,
                    "pnl_pct": round(pnl, 3),
                    "exit_reason": "STOP" if hit_stop else "SELL_HIGH",
                    "result": "WIN" if pnl > 0 else "LOSS",
                    "rsi": meta.get("rsi"), "score": meta.get("score"),
                    "bb_pct": meta.get("bb_pct"), "htf_bias": htf_bias,
                    "rr": meta.get("rr"), "confidence": meta.get("confidence"),
                    "atr": meta.get("atr"),
                })
                in_pos = False
    return rows


def optimize_longs(trades: pd.DataFrame, current: dict) -> tuple[dict, str]:
    t = trades.copy()
    if t.empty or "pnl_pct" not in t.columns:
        return current, "No trades yet."
    t = t[t["result"].isin(["WIN", "LOSS"])] if "result" in t.columns else t
    if len(t) < 12:
        return current, f"Need ≥12 closed spot trades (have {len(t)}). Keep using Single asset."

    new_p = current.copy()
    notes = [f"n={len(t)} avg PnL%/trade={t['pnl_pct'].mean():+.2f}%"]

    def avg(sub):
        return float(sub["pnl_pct"].mean()) if len(sub) else None

    if "score" in t.columns:
        low = t[t["score"] < 40]
        high = t[t["score"] >= 40]
        if len(low) >= 5 and len(high) >= 5 and avg(high) is not None and avg(low) is not None:
            if avg(high) > avg(low) + 0.1:
                new_p["buy_score_min"] = min(52, max(36, current.get("buy_score_min", 36) + 2))
                notes.append("Higher entry scores did better → stricter buy_score_min")
            elif avg(low) > avg(high) + 0.15:
                new_p["buy_score_min"] = max(28, current.get("buy_score_min", 36) - 2)
                notes.append("Slightly looser buy threshold")

    if "rsi" in t.columns:
        deep = t[t["rsi"] < 30]
        mild = t[(t["rsi"] >= 30) & (t["rsi"] < 42)]
        if len(deep) >= 4 and len(mild) >= 4 and avg(deep) is not None and avg(mild) is not None:
            if avg(deep) > avg(mild):
                new_p["rsi_buy_max"] = min(36, current.get("rsi_buy_max", 38))
                notes.append("Deeper RSI buys did better → tighter rsi_buy_max")

    if "exit_reason" in t.columns:
        sh = t[t["exit_reason"] == "SELL_HIGH"]
        st_ = t[t["exit_reason"] == "STOP"]
        if len(sh) >= 3:
            notes.append(f"SELL_HIGH exits: n={len(sh)} avg={avg(sh):+.2f}%")
        if len(st_) >= 3:
            notes.append(f"STOP exits: n={len(st_)} avg={avg(st_):+.2f}%")

    if "htf_bias" in t.columns:
        for bias in ["UP", "RANGE", "DOWN"]:
            sub = t[t["htf_bias"] == bias]
            if len(sub) >= 4:
                notes.append(f"HTF {bias}: n={len(sub)} avg={avg(sub):+.2f}%")

    if "rr" in t.columns:
        low_rr = t[t["rr"] < 1.8]
        high_rr = t[t["rr"] >= 1.8]
        if len(low_rr) >= 5 and len(high_rr) >= 5 and avg(high_rr) and avg(low_rr):
            if avg(high_rr) > avg(low_rr):
                new_p["min_rr"] = min(2.5, max(float(current.get("min_rr", 1.5)), 1.8))
                notes.append("Better results at higher R:R → min_rr up")

    # If sell-high exits are weak, require higher RSI/BB to exit (let winners run)
    if "exit_reason" in t.columns and "pnl_pct" in t.columns:
        sh = t[t["exit_reason"] == "SELL_HIGH"]
        if len(sh) >= 6 and avg(sh) is not None and avg(sh) < 0.2:
            new_p["sell_rsi_min"] = min(72, current.get("sell_rsi_min", 62) + 2)
            new_p["sell_bb_min"] = min(0.85, current.get("sell_bb_min", 0.72) + 0.03)
            notes.append("Weak SELL_HIGH exits → require higher RSI/BB to take profit")

    new_p["optimized_at"] = datetime.now(timezone.utc).isoformat()
    new_p["n_trades_used"] = int(len(t))
    return new_p, " | ".join(notes)


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


def tight_line_chart(df, cols, height=340):
    plot = df[cols].copy().reset_index()
    plot = plot.rename(columns={plot.columns[0]: "Time"})
    rename = {"EMA_fast": "EMA 12", "EMA_slow": "EMA 26", "EMA_50": "EMA 50", "BB_l": "BB Low", "BB_u": "BB High"}
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
        .properties(height=height).interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def tight_bar_chart(df, col, height=190):
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


def snapshot(name, symbol, interval, params):
    raw = fetch_klines(symbol, interval, 180)
    htf_i = HTF_MAP.get(interval, "1h")
    htf_raw = fetch_klines(symbol, htf_i, 120)
    bias = trend_bias(htf_raw) if not htf_raw.empty else "RANGE"
    d = add_signals(raw, bias, params)
    if d.empty:
        return {"Asset": name, "Status": "no data"}
    last = d.iloc[-1]
    prev = d.iloc[-2] if len(d) > 1 else last
    chg = (float(last["Close"]) - float(prev["Close"])) / float(prev["Close"]) * 100
    return {
        "Asset": name, "HTF": bias, "Price": float(last["Close"]), "Chg%": round(chg, 2),
        "RSI": round(float(last["RSI"]), 1), "BB%": round(float(last["BB_pct"]) * 100, 0) if pd.notna(last["BB_pct"]) else None,
        "Score": int(last["Score"]), "Signal": str(last["Signal"]),
        "R:R": float(last["RR"]) if pd.notna(last["RR"]) else None,
        "Status": "ok",
    }


# ── Sidebar ───────────────────────────────────────────────────
params = load_params()
st.sidebar.header("🛠️ Control Room")
st.sidebar.success("Mode: **SPOT ONLY** · Cash → buy coin low → sell high → cash")
page = st.sidebar.radio("View", ["Single asset", "Multi-ticker scanner", "Learning lab"], index=0)
interval = st.sidebar.selectbox("Candle size", INTERVALS, index=3)
limit = st.sidebar.select_slider("Candles shown", options=[50, 100, 150, 200, 300], value=100)

st.sidebar.markdown("---")
st.sidebar.subheader("Risk")
account_usd = st.sidebar.number_input("Account (USD)", min_value=100.0, value=10000.0, step=100.0)
risk_pct = st.sidebar.slider("Risk per trade %", 0.25, 2.0, 0.5, 0.25)
min_rr = st.sidebar.slider("Min R:R", 1.0, 3.0, float(params.get("min_rr", 1.5)), 0.25)
params["min_rr"] = min_rr

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Buy score ≥ {params.get('buy_score_min')} · RSI buy ≤ {params.get('rsi_buy_max')} · "
    f"Sell-high RSI ≥ {params.get('sell_rsi_min')}"
)
if params.get("optimized_at"):
    st.sidebar.caption(f"Optimized: {str(params['optimized_at'])[:19]} · n={params.get('n_trades_used')}")

live = st.sidebar.checkbox("Auto-refresh", value=False)
secs = st.sidebar.slider("Refresh sec", 10, 60, 25)
if st.sidebar.button("🔄 Refresh", use_container_width=True):
    st.cache_data.clear()
    st.rerun()
st.sidebar.info("Cloud: Export trades in Learning lab after each session.")

# ══════════════════════════════════════════════════════════════
# LEARNING LAB
# ══════════════════════════════════════════════════════════════
if page == "Learning lab":
    st.title("🧠 Learning lab · Spot only")
    st.caption("Learns better **buy low** entries and **sell high** exits from your paper spot trades.")

    log = load_trades()
    closed = log[log["result"].isin(["WIN", "LOSS"])] if not log.empty and "result" in log.columns else pd.DataFrame()

    m1, m2, m3 = st.columns(3)
    m1.metric("Log rows", len(log))
    m2.metric("Closed spot trades", len(closed))
    m3.metric("Win rate", f"{(closed['result']=='WIN').mean()*100:.0f}%" if len(closed) else "—")

    up = st.file_uploader("Import trades CSV", type=["csv"])
    if up is not None:
        try:
            incoming = pd.read_csv(up)
            merged = pd.concat([log, incoming], ignore_index=True)
            if "entry_time" in merged.columns:
                merged = merged.drop_duplicates(subset=["symbol", "entry_time", "exit_time"], keep="last")
            persist_trades(merged)
            st.success(f"Log size {len(merged)}")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    if not log.empty:
        st.dataframe(log.sort_values("logged_at", ascending=False) if "logged_at" in log.columns else log,
                     use_container_width=True, height=260)
        st.download_button("⬇️ Export trades CSV", log.to_csv(index=False).encode(),
                           f"spot_trades_{datetime.now(timezone.utc).strftime('%Y%m%d')}.csv", "text/csv")
    else:
        st.info("No trades yet — open Single asset to paper-trade spot round-trips into the log.")

    st.subheader("Stats")
    if closed.empty:
        st.warning("Need closed trades.")
    else:
        exp = closed["pnl_pct"].mean()
        wins = closed.loc[closed["pnl_pct"] > 0, "pnl_pct"]
        losses = closed.loc[closed["pnl_pct"] <= 0, "pnl_pct"]
        pf = abs(wins.sum() / losses.sum()) if len(losses) and losses.sum() != 0 else float("inf")
        a, b, c, d_ = st.columns(4)
        a.metric("Expectancy", f"{exp:+.2f}%")
        b.metric("Profit factor", f"{pf:.2f}" if pf != float("inf") else "∞")
        c.metric("Avg win", f"{wins.mean():+.2f}%" if len(wins) else "—")
        d_.metric("Avg loss", f"{losses.mean():+.2f}%" if len(losses) else "—")
        if "htf_bias" in closed.columns:
            st.markdown("**By HTF**")
            st.dataframe(
                closed.groupby("htf_bias").agg(
                    n=("pnl_pct", "count"),
                    win_rate=("result", lambda s: (s == "WIN").mean() * 100),
                    avg_pnl=("pnl_pct", "mean"),
                ).round(2),
                use_container_width=True,
            )
        if "exit_reason" in closed.columns:
            st.markdown("**By exit**")
            st.dataframe(
                closed.groupby("exit_reason").agg(
                    n=("pnl_pct", "count"),
                    avg_pnl=("pnl_pct", "mean"),
                ).round(2),
                use_container_width=True,
            )
        if "confidence" in closed.columns:
            st.markdown("**By confidence**")
            st.dataframe(
                closed.groupby("confidence").agg(
                    n=("pnl_pct", "count"),
                    win_rate=("result", lambda s: (s == "WIN").mean() * 100),
                    avg_pnl=("pnl_pct", "mean"),
                ).round(2),
                use_container_width=True,
            )

    st.subheader("Optimize buy-low / sell-high rules")
    if st.button("⚙️ Optimize from spot trades", type="primary"):
        new_p, msg = optimize_longs(log, params)
        save_params(new_p)
        st.success("Updated.")
        st.write(msg)
        st.json({k: new_p[k] for k in DEFAULT_PARAMS if k in new_p})
        time.sleep(0.3)
        st.rerun()

    st.download_button("⬇️ Export params", json.dumps(params, indent=2), "learned_params.json", "application/json")
    upj = st.file_uploader("Import params JSON", type=["json"], key="pj")
    if upj:
        try:
            incoming = json.load(upj)
            merged = DEFAULT_PARAMS.copy()
            merged.update(incoming)
            save_params(merged)
            st.rerun()
        except Exception as e:
            st.error(str(e))
    if st.button("Reset params"):
        save_params(DEFAULT_PARAMS.copy())
        st.rerun()
    st.stop()

# ══════════════════════════════════════════════════════════════
# SCANNER
# ══════════════════════════════════════════════════════════════
if page == "Multi-ticker scanner":
    st.title("📡 Scanner · Spot only")
    names = st.multiselect("Assets", list(CRYPTO.keys()), default=[n for n in DEFAULT_SCAN if n in CRYPTO])
    if not names:
        st.stop()
    rows = []
    bar = st.progress(0)
    for i, n in enumerate(names):
        rows.append(snapshot(n, CRYPTO[n], interval, params))
        bar.progress((i + 1) / len(names))
    bar.empty()
    ok = pd.DataFrame(rows)
    ok = ok[ok.get("Status", "ok") == "ok"] if "Status" in ok.columns else ok
    for _, r in ok.iterrows():
        if r["Signal"] == "BUY" and (r.get("R:R") or 0) >= min_rr:
            st.markdown(
                f"<div class='alert-box alert-buy'><b>BUY LOW</b> {r['Asset']} · HTF {r['HTF']} · "
                f"RSI {r['RSI']} · Score {r['Score']} · R:R {r['R:R']}</div>",
                unsafe_allow_html=True,
            )
        elif r["Signal"] == "SELL_HIGH":
            st.markdown(
                f"<div class='alert-box alert-sell'><b>SELL HIGH</b> (exit zone) {r['Asset']} · "
                f"RSI {r['RSI']} · BB% {r.get('BB%')}</div>",
                unsafe_allow_html=True,
            )
    if not ok.empty:
        show = ok.drop(columns=["Status"], errors="ignore")
        show["Price"] = show["Price"].map(fmt_px)
        st.dataframe(show, use_container_width=True, height=400)
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
custom = st.sidebar.text_input("Custom (BTCUSDT)", "")
if custom.strip():
    symbol = custom.strip().upper()
    name = symbol

st.title("⚡ Buy Low · Sell High · Spot only")
st.caption(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC · {name} · {interval} · spot only · no shorts · no leverage")

htf_i = HTF_MAP.get(interval, "1h")
with st.spinner("Loading…"):
    raw = fetch_klines(symbol, interval, max(limit + 100, 180))
    htf_raw = fetch_klines(symbol, htf_i, 150)
    htf_bias = trend_bias(htf_raw) if not htf_raw.empty else "RANGE"
    df_full = add_signals(raw, htf_bias, params)

if df_full.empty:
    st.error("No data.")
    st.stop()

df = df_full.tail(limit).copy()
latest = df.iloc[-1]
prev = df.iloc[-2]
pct = (float(latest["Close"]) - float(prev["Close"])) / float(prev["Close"]) * 100
sig = str(latest["Signal"])
price = float(latest["Close"])

paper_rows = simulate_longs(df_full.tail(min(400, len(df_full))), name, symbol, interval, htf_bias, min_rr)
if paper_rows:
    append_trades(paper_rows)

st.success(f"HTF ({htf_i}) **{htf_bias}** · paper spot round-trips closed this window: {len(paper_rows)}")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Price", fmt_px(price), f"{pct:+.2f}%")
label = {"BUY": "BUY LOW", "SELL_HIGH": "SELL HIGH", "HOLD": "HOLD"}.get(sig, sig)
cls = {"BUY": "signal-buy", "SELL_HIGH": "signal-sell", "HOLD": "signal-hold"}.get(sig, "signal-hold")
c2.markdown("**Action**")
c2.markdown(f"<div class='{cls}'>{label}</div>", unsafe_allow_html=True)
c3.metric("RSI", f"{float(latest['RSI']):.1f}")
c4.metric("Score", f"{int(latest['Score'])}")
c5.metric("BB %", f"{float(latest['BB_pct'])*100:.0f}%")
c6.metric("HTF", htf_bias)

st.markdown("### 📋 Spot trade plan")
if sig == "BUY":
    plan = build_plan(latest, account_usd, risk_pct, min_rr)
    checks = confluence_checklist(latest, htf_bias, min_rr)
    n_ok = sum(1 for _, v in checks if v)
    conf = "HIGH" if n_ok >= 6 and plan["passes_rr"] else ("MED" if n_ok >= 4 else "LOW")
    p1, p2 = st.columns([1.2, 1])
    with p1:
        st.markdown(f"""
        <div class="plan-box">
        <b>BUY LOW</b> (spot only)<br>
        Entry {fmt_px(plan['entry'])}<br>
        Stop {fmt_px(plan['stop'])} ({plan['risk_pct']:.2f}%)<br>
        Sell-high T1 / T2: {fmt_px(plan['t1'])} / {fmt_px(plan['t2'])}<br>
        R:R {plan['rr']:.2f} {"✅" if plan['passes_rr'] else "❌ skip"}<br>
        Confidence {conf} ({n_ok}/7)
        </div>""", unsafe_allow_html=True)
    with p2:
        st.markdown(f"""
        <div class="plan-box">
        Risk {risk_pct}% = <b>${plan['risk_usd']:,.2f}</b><br>
        Size ≈ {plan['size']:.6g}<br>
        Notional ≈ ${plan['notional']:,.2f}
        </div>""", unsafe_allow_html=True)
    for text, ok in checks:
        st.markdown(f"{'<span class=check-ok>✓</span>' if ok else '<span class=check-no>✗</span>'} {text}", unsafe_allow_html=True)
elif sig == "SELL_HIGH":
    st.markdown(
        f"<div class='alert-box alert-sell'><b>SELL HIGH</b> — sell coin for cash (spot exit) "
        f"(RSI {float(latest['RSI']):.0f}, BB {float(latest['BB_pct'])*100:.0f}%). "
        f"Not a short. Not leverage. Spot sell only.</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Sell zone ≈ {fmt_px(float(latest['Sell_zone']))}")
else:
    st.info(
        f"HOLD — waiting for buy-low (score/RSI/BB) with HTF={htf_bias}. "
        f"Buy zone {fmt_px(float(latest['Buy_zone']))} · Sell zone {fmt_px(float(latest['Sell_zone']))}"
    )

st.caption(f"Trade log: **{len(load_trades())}** rows · Learning lab to optimize")

st.divider()
st.subheader(f"📈 {name}")
tight_line_chart(df, ["Close", "EMA_fast", "EMA_slow", "EMA_50"])
tight_line_chart(df, ["Close", "BB_l", "BB_u"])
a, b = st.columns(2)
with a:
    tight_line_chart(df, ["RSI"], height=200)
with b:
    tight_bar_chart(df, "Volume", height=200)

st.subheader("Paper spot trades this window")
if paper_rows:
    st.dataframe(pd.DataFrame(paper_rows), use_container_width=True, height=220)
else:
    st.caption("No closed spot round-trips in this window.")

with st.expander("Rules"):
    st.markdown("""
**Spot only**
- **BUY LOW** — spend cash, buy the coin when cheap (score/RSI/BB) and HTF is UP or RANGE  
- **SELL HIGH** — sell the coin for cash when expensive (or stop)  
- **Never** short · **never** use leverage — only spot inventory (cash or coin)  

Learning adjusts buy-low and sell-high thresholds from paper spot trades.
""")

if live:
    time.sleep(secs)
    st.rerun()
