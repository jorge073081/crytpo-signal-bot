"""
Crypto Signal Bot Pro — Learn from trades (Cloud-friendly)
==========================================================
• Trade log (CSV) + import/export for Streamlit Cloud
• Stats: win rate, expectancy, by regime / confidence
• Optimize thresholds from log
• Buy-low/sell-high + ATR stops + HTF trend filter

Streamlit Cloud: disk can reset on redeploy — use Export/Import to keep history.
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

st.set_page_config(page_title="Crypto Signal Bot Pro", page_icon="⚡", layout="wide")

st.markdown("""
<style>
div[data-testid="stMetricValue"] { font-size: 1.2rem; }
.signal-buy  { color: #00c853; font-weight: 800; font-size: 1.45rem; }
.signal-sell { color: #ff1744; font-weight: 800; font-size: 1.45rem; }
.signal-hold { color: #ffab00; font-weight: 800; font-size: 1.45rem; }
.alert-box { padding: 10px 14px; border-radius: 8px; margin: 6px 0; font-weight: 600; }
.alert-buy { background: rgba(0,200,83,0.15); border-left: 4px solid #00c853; }
.alert-sell { background: rgba(255,23,68,0.15); border-left: 4px solid #ff1744; }
.plan-box { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
            padding: 14px 16px; border-radius: 10px; margin-bottom: 8px; }
.check-ok { color: #00c853; } .check-no { color: #ff1744; }
</style>
""", unsafe_allow_html=True)

# ── Paths (Cloud: export often; files may vanish on redeploy) ─
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
TRADES_PATH = DATA_DIR / "trades.csv"
PARAMS_PATH = DATA_DIR / "learned_params.json"

DEFAULT_PARAMS = {
    "buy_score_min": 36,
    "sell_score_max": -36,
    "rsi_buy_max": 36,
    "rsi_sell_min": 64,
    "bb_buy_max": 0.35,
    "bb_sell_min": 0.65,
    "min_rr": 1.5,
    "atr_stop_mult": 1.5,
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


# ══════════════════════════════════════════════════════════════
# Persistence helpers (Cloud-friendly)
# ══════════════════════════════════════════════════════════════
def load_params() -> dict:
    p = DEFAULT_PARAMS.copy()
    if PARAMS_PATH.exists():
        try:
            p.update(json.loads(PARAMS_PATH.read_text()))
        except Exception:
            pass
    if "learned_params" in st.session_state:
        p.update(st.session_state["learned_params"])
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
    if "trade_log" in st.session_state and len(st.session_state["trade_log"]):
        frames.append(pd.DataFrame(st.session_state["trade_log"]))
    if not frames:
        return pd.DataFrame(columns=cols)
    df = pd.concat(frames, ignore_index=True)
    # de-dupe rough key
    if "entry_time" in df.columns and "symbol" in df.columns and "side" in df.columns:
        df = df.drop_duplicates(subset=["symbol", "side", "entry_time", "exit_time"], keep="last")
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
        combined = combined.drop_duplicates(subset=["symbol", "side", "entry_time", "exit_time"], keep="last")
    persist_trades(combined)


# ══════════════════════════════════════════════════════════════
# Market data + indicators
# ══════════════════════════════════════════════════════════════
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
    d["Score"] = d["Score_raw"].copy()
    if htf_bias == "DOWN":
        d["Score"] = d["Score"].clip(upper=25)
    if htf_bias == "UP":
        d["Score"] = d["Score"].clip(lower=-25)
    d["HTF_bias"] = htf_bias

    buy_min = float(params.get("buy_score_min", 36))
    sell_max = float(params.get("sell_score_max", -36))
    rsi_buy = float(params.get("rsi_buy_max", 36))
    rsi_sell = float(params.get("rsi_sell_min", 64))
    bb_buy = float(params.get("bb_buy_max", 0.35))
    bb_sell = float(params.get("bb_sell_min", 0.65))
    allowed_buy = htf_bias in ("UP", "RANGE")
    allowed_sell = htf_bias in ("DOWN", "RANGE")

    def classify(row):
        s = row["Score"]
        buy_setup = s >= buy_min or (s >= buy_min - 8 and row["RSI"] < rsi_buy and row["BB_pct"] < bb_buy)
        sell_setup = s <= sell_max or (s <= sell_max + 8 and row["RSI"] > rsi_sell and row["BB_pct"] > bb_sell)
        if buy_setup and allowed_buy:
            return "BUY"
        if sell_setup and allowed_sell:
            return "SELL"
        return "HOLD"

    d["Signal"] = d.apply(classify, axis=1)

    mult = float(params.get("atr_stop_mult", 1.5))
    atr_v = d["ATR"].replace(0, np.nan)
    d["Stop_long"] = np.minimum(d["Close"] - mult * atr_v, d["Swing_low"] * 0.998)
    d["T1_long"] = d["Close"] + 2.0 * (d["Close"] - d["Stop_long"])
    d["T2_long"] = d["Close"] + 3.0 * (d["Close"] - d["Stop_long"])
    d["Stop_short"] = np.maximum(d["Close"] + mult * atr_v, d["Swing_high"] * 1.002)
    d["T1_short"] = d["Close"] - 2.0 * (d["Stop_short"] - d["Close"])
    d["T2_short"] = d["Close"] - 3.0 * (d["Stop_short"] - d["Close"])
    risk_l = (d["Close"] - d["Stop_long"]).replace(0, np.nan)
    risk_s = (d["Stop_short"] - d["Close"]).replace(0, np.nan)
    d["RR_long"] = ((d["T1_long"] - d["Close"]) / risk_l).round(2)
    d["RR_short"] = ((d["Close"] - d["T1_short"]) / risk_s).round(2)
    d["Buy_zone"] = d[["BB_l", "Swing_low"]].max(axis=1)
    d["Sell_zone"] = d[["BB_u", "Swing_high"]].min(axis=1)
    return d.dropna(subset=["RSI", "ATR"])


def confidence_of(row, htf_bias, side, min_rr):
    items = confluence_checklist(row, htf_bias, side, min_rr)
    n = sum(1 for _, v in items if v)
    if n >= 6:
        return "HIGH"
    if n >= 4:
        return "MED"
    return "LOW"


def confluence_checklist(row, htf_bias, side, min_rr):
    items = []
    if side == "BUY":
        items += [
            ("HTF allows longs", htf_bias in ("UP", "RANGE")),
            ("Local not strongly down", bool(row.get("Local_up") or row["EMA_fast"] >= row["EMA_slow"] * 0.998)),
            ("RSI < 65", float(row["RSI"]) < 65),
            ("Near value (BB%<40 or RSI<40)", float(row["BB_pct"]) < 0.4 or float(row["RSI"]) < 40),
            ("MACD ok / cross up", float(row["MACD"]) >= float(row["MACD_sig"]) or bool(row.get("MACD_cross_up"))),
            (f"R:R ≥ {min_rr}", float(row.get("RR_long") or 0) >= min_rr),
            ("ATR ok", pd.notna(row["ATR"]) and float(row["ATR"]) > 0),
        ]
    else:
        items += [
            ("HTF allows shorts", htf_bias in ("DOWN", "RANGE")),
            ("Local not strongly up", bool(row.get("Local_dn") or row["EMA_fast"] <= row["EMA_slow"] * 1.002)),
            ("RSI > 35", float(row["RSI"]) > 35),
            ("Near premium (BB%>60 or RSI>60)", float(row["BB_pct"]) > 0.6 or float(row["RSI"]) > 60),
            ("MACD ok / cross down", float(row["MACD"]) <= float(row["MACD_sig"]) or bool(row.get("MACD_cross_dn"))),
            (f"R:R ≥ {min_rr}", float(row.get("RR_short") or 0) >= min_rr),
            ("ATR ok", pd.notna(row["ATR"]) and float(row["ATR"]) > 0),
        ]
    return items


def build_trade_plan(row, side, account_usd, risk_pct, min_rr):
    px = float(row["Close"])
    atr_v = float(row["ATR"]) if pd.notna(row["ATR"]) else px * 0.01
    if side == "BUY":
        stop, t1, t2 = float(row["Stop_long"]), float(row["T1_long"]), float(row["T2_long"])
        rr = float(row["RR_long"]) if pd.notna(row["RR_long"]) else 0
        risk_per = max(px - stop, atr_v * 0.5)
    else:
        stop, t1, t2 = float(row["Stop_short"]), float(row["T1_short"]), float(row["T2_short"])
        rr = float(row["RR_short"]) if pd.notna(row["RR_short"]) else 0
        risk_per = max(stop - px, atr_v * 0.5)
    risk_usd = account_usd * (risk_pct / 100.0)
    size = risk_usd / risk_per if risk_per > 0 else 0
    return {
        "side": side, "entry": px, "stop": stop, "t1": t1, "t2": t2, "rr": rr, "atr": atr_v,
        "risk_usd": risk_usd, "size": size, "notional": size * px,
        "passes_rr": rr >= min_rr, "risk_pct_of_entry": (risk_per / px * 100) if px else 0,
    }


def simulate_paper(df, asset, symbol, interval, htf_bias, min_rr):
    """Walk history; return list of closed trade dicts for logging."""
    rows = []
    in_pos = False
    entry = entry_t = stop = side = None
    meta = {}
    for ts, row in df.iterrows():
        if not in_pos and row["Signal"] in ("BUY", "SELL"):
            side = row["Signal"]
            rr = float(row["RR_long"] if side == "BUY" else row["RR_short"])
            if rr < min_rr:
                continue
            conf = confidence_of(row, htf_bias, side, min_rr)
            in_pos = True
            entry = float(row["Close"])
            entry_t = ts
            stop = float(row["Stop_long"] if side == "BUY" else row["Stop_short"])
            meta = {
                "rsi": float(row["RSI"]), "score": float(row["Score"]),
                "bb_pct": float(row["BB_pct"]) if pd.notna(row["BB_pct"]) else None,
                "rr": rr, "confidence": conf, "atr": float(row["ATR"]) if pd.notna(row["ATR"]) else None,
            }
        elif in_pos:
            px = float(row["Close"])
            hit_stop = (side == "BUY" and px <= stop) or (side == "SELL" and px >= stop)
            opposite = row["Signal"] == ("SELL" if side == "BUY" else "BUY")
            if hit_stop or opposite:
                pnl = (px - entry) / entry * 100 if side == "BUY" else (entry - px) / entry * 100
                rows.append({
                    "logged_at": datetime.now(timezone.utc).isoformat(),
                    "asset": asset, "symbol": symbol, "interval": interval,
                    "side": side,
                    "entry_time": str(entry_t), "exit_time": str(ts),
                    "entry": entry, "exit": px, "stop": stop,
                    "pnl_pct": round(pnl, 3),
                    "exit_reason": "STOP" if hit_stop else "SIGNAL",
                    "result": "WIN" if pnl > 0 else "LOSS",
                    "rsi": meta.get("rsi"), "score": meta.get("score"),
                    "bb_pct": meta.get("bb_pct"), "htf_bias": htf_bias,
                    "rr": meta.get("rr"), "confidence": meta.get("confidence"),
                    "atr": meta.get("atr"),
                })
                in_pos = False
    return rows


def optimize_from_log(trades: pd.DataFrame, current: dict) -> tuple[dict, str]:
    """Adjust thresholds from closed trades. Needs enough samples."""
    t = trades.copy()
    if t.empty or "pnl_pct" not in t.columns:
        return current, "No trades in log."
    t = t[t["result"].isin(["WIN", "LOSS"])] if "result" in t.columns else t
    if len(t) < 12:
        return current, f"Need ≥12 closed trades to optimize (have {len(t)}). Keep logging."

    new_p = current.copy()
    notes = []

    # Expectancy helper
    def exp(sub):
        if sub.empty:
            return None
        return float(sub["pnl_pct"].mean())

    overall = exp(t)
    notes.append(f"Overall avg PnL%/trade: {overall:+.2f}% on {len(t)} trades")

    # Score threshold: if low |score| trades worse, raise bar
    if "score" in t.columns:
        t["abs_score"] = t["score"].abs()
        low = t[t["abs_score"] < 40]
        high = t[t["abs_score"] >= 40]
        if len(low) >= 5 and len(high) >= 5:
            if exp(high) is not None and exp(low) is not None and exp(high) > exp(low) + 0.1:
                new_p["buy_score_min"] = min(50, max(36, current.get("buy_score_min", 36) + 2))
                new_p["sell_score_max"] = max(-50, min(-36, current.get("sell_score_max", -36) - 2))
                notes.append("High-|score| trades better → stricter score thresholds")
            elif exp(low) is not None and exp(high) is not None and exp(low) > exp(high) + 0.15:
                new_p["buy_score_min"] = max(28, current.get("buy_score_min", 36) - 2)
                new_p["sell_score_max"] = min(-28, current.get("sell_score_max", -36) + 2)
                notes.append("Lower-score trades not worse → slightly looser thresholds")

    # Confidence: if HIGH much better, user should prefer HIGH (we note it)
    if "confidence" in t.columns:
        for conf in ["HIGH", "MED", "LOW"]:
            sub = t[t["confidence"] == conf]
            if len(sub) >= 4:
                notes.append(f"{conf}: n={len(sub)} avg PnL%={exp(sub):+.2f}%")

    # HTF regime
    if "htf_bias" in t.columns:
        for bias in ["UP", "DOWN", "RANGE"]:
            sub = t[t["htf_bias"] == bias]
            if len(sub) >= 4:
                notes.append(f"HTF {bias}: n={len(sub)} avg PnL%={exp(sub):+.2f}%")

    # R:R filter: if low RR trades lose, raise min_rr
    if "rr" in t.columns:
        low_rr = t[t["rr"] < 1.8]
        high_rr = t[t["rr"] >= 1.8]
        if len(low_rr) >= 5 and len(high_rr) >= 5:
            if exp(high_rr) is not None and exp(low_rr) is not None and exp(high_rr) > exp(low_rr):
                new_p["min_rr"] = min(2.5, max(current.get("min_rr", 1.5), 1.8))
                notes.append("Higher R:R trades better → min_rr raised toward 1.8+")

    # RSI extremes quality
    if "rsi" in t.columns and "side" in t.columns:
        buys = t[t["side"] == "BUY"]
        if len(buys) >= 8:
            deep = buys[buys["rsi"] < 30]
            mild = buys[(buys["rsi"] >= 30) & (buys["rsi"] < 40)]
            if len(deep) >= 3 and len(mild) >= 3 and exp(deep) is not None and exp(mild) is not None:
                if exp(deep) > exp(mild):
                    new_p["rsi_buy_max"] = min(34, current.get("rsi_buy_max", 36))
                    notes.append("Deeper RSI oversolds did better on buys → tighter rsi_buy_max")

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


def tight_line_chart(df, cols, height=340, title=""):
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
        .properties(height=height, title=title).interactive()
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
    rr = float(last["RR_long"]) if last["Signal"] == "BUY" else (
        float(last["RR_short"]) if last["Signal"] == "SELL" else None)
    return {
        "Asset": name, "HTF": bias, "Price": float(last["Close"]), "Chg%": round(chg, 2),
        "RSI": round(float(last["RSI"]), 1),
        "Score": int(last["Score"]), "Signal": str(last["Signal"]), "R:R": rr,
        "Status": "ok",
    }


# ══════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════
params = load_params()

st.sidebar.header("🛠️ Control Room")
page = st.sidebar.radio(
    "View",
    ["Single asset", "Multi-ticker scanner", "Learning lab"],
    index=0,
)
interval = st.sidebar.selectbox("Candle size", INTERVALS, index=3)
limit = st.sidebar.select_slider("Candles shown", options=[50, 100, 150, 200, 300], value=100)

st.sidebar.markdown("---")
st.sidebar.subheader("Risk limits")
account_usd = st.sidebar.number_input("Account size (USD)", min_value=100.0, value=10000.0, step=100.0)
risk_pct = st.sidebar.slider("Risk per trade %", 0.25, 2.0, 0.5, 0.25)
# min_rr: sidebar can override learned
min_rr = st.sidebar.slider("Min R:R", 1.0, 3.0, float(params.get("min_rr", 1.5)), 0.25)
params["min_rr"] = min_rr

st.sidebar.markdown("---")
st.sidebar.subheader("Learned thresholds")
st.sidebar.caption(
    f"Buy score ≥ {params.get('buy_score_min')} · "
    f"Sell ≤ {params.get('sell_score_max')} · "
    f"trades used: {params.get('n_trades_used', 0)}"
)
if params.get("optimized_at"):
    st.sidebar.caption(f"Last optimize: {params['optimized_at'][:19]} UTC")

st.sidebar.markdown("---")
live = st.sidebar.checkbox("Auto-refresh ON", value=False)
secs = st.sidebar.slider("Refresh seconds", 10, 60, 25)
if st.sidebar.button("🔄 Refresh now", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.info(
    "Cloud tip: **Learning lab → Export trades** after sessions. "
    "Redeploys can wipe local files — re-import your CSV."
)

# ══════════════════════════════════════════════════════════════
# LEARNING LAB
# ══════════════════════════════════════════════════════════════
if page == "Learning lab":
    st.title("🧠 Learning lab")
    st.caption(
        "Streamlit Cloud may clear disk on redeploy. Export often, import when you return."
    )

    log = load_trades()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Logged trades", len(log))
    with c2:
        closed = log[log["result"].isin(["WIN", "LOSS"])] if not log.empty and "result" in log.columns else pd.DataFrame()
        st.metric("Closed", len(closed))
    with c3:
        if len(closed):
            wr = (closed["result"] == "WIN").mean() * 100
            st.metric("Win rate", f"{wr:.0f}%")
        else:
            st.metric("Win rate", "—")

    st.subheader("1) Trade log")
    up = st.file_uploader("Import trades CSV (after redeploy)", type=["csv"])
    if up is not None:
        try:
            incoming = pd.read_csv(up)
            merged = pd.concat([log, incoming], ignore_index=True)
            if "entry_time" in merged.columns:
                merged = merged.drop_duplicates(
                    subset=["symbol", "side", "entry_time", "exit_time"], keep="last"
                )
            persist_trades(merged)
            st.success(f"Imported — log now has {len(merged)} rows.")
            log = merged
        except Exception as e:
            st.error(f"Import failed: {e}")

    if not log.empty:
        st.dataframe(log.sort_values("logged_at", ascending=False) if "logged_at" in log.columns else log,
                     use_container_width=True, height=280)
        st.download_button(
            "⬇️ Export trades CSV",
            data=log.to_csv(index=False).encode(),
            file_name=f"trades_{datetime.now(timezone.utc).strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    else:
        st.info("No trades yet. Open **Single asset**, load a chart — paper trades auto-append to the log.")

    st.subheader("2) Stats")
    if closed.empty:
        st.warning("Need closed WIN/LOSS rows for stats.")
    else:
        exp = closed["pnl_pct"].mean()
        pf_wins = closed.loc[closed["pnl_pct"] > 0, "pnl_pct"].sum()
        pf_loss = closed.loc[closed["pnl_pct"] <= 0, "pnl_pct"].sum()
        profit_factor = abs(pf_wins / pf_loss) if pf_loss != 0 else float("inf")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Expectancy (avg %)", f"{exp:+.2f}%")
        m2.metric("Profit factor", f"{profit_factor:.2f}" if profit_factor != float("inf") else "∞")
        m3.metric("Avg win %", f"{closed.loc[closed['result']=='WIN','pnl_pct'].mean():+.2f}%" if (closed['result']=='WIN').any() else "—")
        m4.metric("Avg loss %", f"{closed.loc[closed['result']=='LOSS','pnl_pct'].mean():+.2f}%" if (closed['result']=='LOSS').any() else "—")

        if "htf_bias" in closed.columns:
            st.markdown("**By HTF regime**")
            by = closed.groupby("htf_bias").agg(
                n=("pnl_pct", "count"),
                win_rate=("result", lambda s: (s == "WIN").mean() * 100),
                avg_pnl=("pnl_pct", "mean"),
            ).round(2)
            st.dataframe(by, use_container_width=True)

        if "confidence" in closed.columns:
            st.markdown("**By confidence**")
            byc = closed.groupby("confidence").agg(
                n=("pnl_pct", "count"),
                win_rate=("result", lambda s: (s == "WIN").mean() * 100),
                avg_pnl=("pnl_pct", "mean"),
            ).round(2)
            st.dataframe(byc, use_container_width=True)

        if "side" in closed.columns:
            st.markdown("**By side**")
            bys = closed.groupby("side").agg(
                n=("pnl_pct", "count"),
                win_rate=("result", lambda s: (s == "WIN").mean() * 100),
                avg_pnl=("pnl_pct", "mean"),
            ).round(2)
            st.dataframe(bys, use_container_width=True)

    st.subheader("3) Optimize rules from log")
    st.caption("Adjusts score/RSI/min R:R thresholds when enough history exists. Does not guarantee future profits.")
    if st.button("⚙️ Run optimization", type="primary"):
        new_p, msg = optimize_from_log(log, params)
        save_params(new_p)
        st.success("Params updated.")
        st.write(msg)
        st.json({k: new_p[k] for k in DEFAULT_PARAMS if k in new_p})
        st.rerun()

    st.download_button(
        "⬇️ Export learned params JSON",
        data=json.dumps(params, indent=2),
        file_name="learned_params.json",
        mime="application/json",
    )
    up_p = st.file_uploader("Import learned params JSON", type=["json"], key="params_up")
    if up_p is not None:
        try:
            incoming = json.load(up_p)
            merged_p = DEFAULT_PARAMS.copy()
            merged_p.update(incoming)
            save_params(merged_p)
            st.success("Params imported.")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    if st.button("Reset params to defaults"):
        save_params(DEFAULT_PARAMS.copy())
        st.rerun()

    st.stop()

# ══════════════════════════════════════════════════════════════
# SCANNER
# ══════════════════════════════════════════════════════════════
if page == "Multi-ticker scanner":
    st.title("📡 Scanner")
    scan_names = st.multiselect(
        "Assets", list(CRYPTO.keys()),
        default=[n for n in DEFAULT_SCAN if n in CRYPTO],
    )
    if not scan_names:
        st.stop()
    rows = []
    prog = st.progress(0)
    for i, n in enumerate(scan_names):
        rows.append(snapshot(n, CRYPTO[n], interval, params))
        prog.progress((i + 1) / len(scan_names))
    prog.empty()
    ok = pd.DataFrame(rows)
    ok = ok[ok.get("Status", "ok") == "ok"] if "Status" in ok.columns else ok
    alerts = []
    for _, r in ok.iterrows():
        if r["Signal"] == "BUY" and (r.get("R:R") or 0) >= min_rr:
            alerts.append(("buy", f"**BUY** {r['Asset']} · HTF {r['HTF']} · R:R {r['R:R']} · RSI {r['RSI']}"))
        elif r["Signal"] == "SELL" and (r.get("R:R") or 0) >= min_rr:
            alerts.append(("sell", f"**SELL** {r['Asset']} · HTF {r['HTF']} · R:R {r['R:R']} · RSI {r['RSI']}"))
    if alerts:
        for k, m in alerts:
            st.markdown(f"<div class='alert-box {'alert-buy' if k=='buy' else 'alert-sell'}'>{m}</div>", unsafe_allow_html=True)
    else:
        st.info("No trades pass filter + min R:R.")
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
custom = st.sidebar.text_input("Custom (e.g. ONDOUSDT)", "")
if custom.strip():
    symbol = custom.strip().upper()
    name = symbol

st.title("⚡ Trade Decision Pack")
st.caption(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC · {name} · {interval}")

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

# Auto paper-log from this window (idempotent via de-dupe)
paper_rows = simulate_paper(df_full.tail(min(400, len(df_full))), name, symbol, interval, htf_bias, min_rr)
if paper_rows:
    append_trades(paper_rows)

st.success(f"Data OK · HTF ({htf_i}) **{htf_bias}** · paper rows in window: {len(paper_rows)}")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Price", fmt_px(price), f"{pct:+.2f}%")
label = "BUY LOW" if sig == "BUY" else ("SELL HIGH" if sig == "SELL" else "HOLD")
cls = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}[sig]
c2.markdown("**Signal**")
c2.markdown(f"<div class='{cls}'>{label}</div>", unsafe_allow_html=True)
c3.metric("RSI", f"{float(latest['RSI']):.1f}")
c4.metric("Score", f"{int(latest['Score'])}")
c5.metric("ATR", fmt_px(float(latest["ATR"])))
c6.metric("HTF", htf_bias)

st.markdown("### 📋 Trade plan")
if sig in ("BUY", "SELL"):
    plan = build_trade_plan(latest, sig, account_usd, risk_pct, min_rr)
    checks = confluence_checklist(latest, htf_bias, sig, min_rr)
    n_ok = sum(1 for _, v in checks if v)
    conf = "HIGH" if n_ok >= 6 and plan["passes_rr"] else ("MED" if n_ok >= 4 else "LOW")
    p1, p2 = st.columns([1.2, 1])
    with p1:
        st.markdown(f"""
        <div class="plan-box">
        <b>Side:</b> {plan['side']}<br>
        <b>Entry:</b> {fmt_px(plan['entry'])}<br>
        <b>Stop:</b> {fmt_px(plan['stop'])} ({plan['risk_pct_of_entry']:.2f}%)<br>
        <b>T1 / T2:</b> {fmt_px(plan['t1'])} / {fmt_px(plan['t2'])}<br>
        <b>R:R:</b> {plan['rr']:.2f} {"✅" if plan['passes_rr'] else "❌ below min"}<br>
        <b>Confidence:</b> {conf} ({n_ok}/7)
        </div>""", unsafe_allow_html=True)
    with p2:
        st.markdown(f"""
        <div class="plan-box">
        Risk {risk_pct}% of ${account_usd:,.0f} = <b>${plan['risk_usd']:,.2f}</b><br>
        Size ≈ <b>{plan['size']:.6g}</b><br>
        Notional ≈ <b>${plan['notional']:,.2f}</b>
        </div>""", unsafe_allow_html=True)
    for text, ok in checks:
        st.markdown(
            f"{'<span class=check-ok>✓</span>' if ok else '<span class=check-no>✗</span>'} {text}",
            unsafe_allow_html=True,
        )
else:
    st.info(f"HOLD — HTF={htf_bias}. Wait for aligned buy-low / sell-high setup.")

log_n = len(load_trades())
st.caption(f"Trade log size: **{log_n}** rows · open **Learning lab** for stats & optimize")

st.divider()
st.subheader(f"📈 {name}")
tight_line_chart(df, ["Close", "EMA_fast", "EMA_slow", "EMA_50"], height=340)
tight_line_chart(df, ["Close", "BB_l", "BB_u"], height=260)
a, b = st.columns(2)
with a:
    tight_line_chart(df, ["RSI"], height=200)
with b:
    tight_bar_chart(df, "Volume", height=200)

st.subheader("Recent paper trades (this symbol window)")
if paper_rows:
    st.dataframe(pd.DataFrame(paper_rows), use_container_width=True, height=220)
else:
    st.caption("No closed paper trades in this window (or all below min R:R).")

if live:
    time.sleep(secs)
    st.rerun()
