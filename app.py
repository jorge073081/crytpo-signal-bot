"""
Crypto Spot Desk — best-effort buy / sell decisions
===================================================
Spot only. Focus: high-quality BUY LOW / SELL HIGH calls.

Missing optional feeds (funding, OI) are ignored — not treated as failures.
Core decision uses: price action, RSI, Bollinger, HTF trend, volume,
Fear&Greed, global mcap, FRED/Yahoo yields when available.

Not financial advice. Not guaranteed profit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timezone
import requests
import time
import xml.etree.ElementTree as ET
import os
import json
import re
from pathlib import Path

st.set_page_config(page_title="Spot Desk", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.big { font-size: 1.9rem; font-weight: 800; margin: 0.15rem 0; }
.buy { color: #00c853; } .sell { color: #ffab00; } .wait { color: #90a4ae; }
.card {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 0.6rem;
}
.alert-buy {
  background: rgba(0,200,83,0.18); border-left: 5px solid #00c853;
  padding: 12px 16px; border-radius: 8px; font-weight: 700; margin: 8px 0;
}
.alert-sell {
  background: rgba(255,171,0,0.18); border-left: 5px solid #ffab00;
  padding: 12px 16px; border-radius: 8px; font-weight: 700; margin: 8px 0;
}
.ok { color: #00c853; } .no { color: #ff5252; }
.small { color: #9e9e9e; font-size: 0.88rem; }
.live-dot { color: #ff1744; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

CRYPTO = {
    "Bitcoin": "BTCUSDT", "Ethereum": "ETHUSDT", "Solana": "SOLUSDT",
    "BNB": "BNBUSDT", "XRP": "XRPUSDT", "Cardano": "ADAUSDT",
    "Dogecoin": "DOGEUSDT", "Avalanche": "AVAXUSDT", "Chainlink": "LINKUSDT",
    "ONDO": "ONDOUSDT", "Stellar": "XLMUSDT", "Hedera": "HBARUSDT",
    "Algorand": "ALGOUSDT", "Flare": "FLRUSDT", "Sui": "SUIUSDT",
    "Near": "NEARUSDT", "Render": "RENDERUSDT", "Pepe": "PEPEUSDT",
    "Toncoin": "TONUSDT", "Tron": "TRXUSDT", "Litecoin": "LTCUSDT",
    "Polkadot": "DOTUSDT", "Aave": "AAVEUSDT",
}
INTERVALS = ["15m", "30m", "1h", "4h", "1d"]
HTF_MAP = {"15m": "1h", "30m": "4h", "1h": "4h", "4h": "1d", "1d": "1d"}
BINANCE = ["https://data-api.binance.vision", "https://api.binance.us"]

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
TRADES_PATH = DATA_DIR / "paper_trades.csv"


def load_paper_trades():
    frames = []
    if TRADES_PATH.exists():
        try:
            frames.append(pd.read_csv(TRADES_PATH))
        except Exception:
            pass
    if st.session_state.get("paper_trades"):
        frames.append(pd.DataFrame(st.session_state["paper_trades"]))
    if not frames:
        return pd.DataFrame(columns=[
            "logged_at", "asset", "symbol", "interval", "side",
            "entry_time", "exit_time", "entry", "exit", "stop",
            "pnl_pct", "exit_reason", "result", "confidence", "htf", "score", "strict",
        ])
    df = pd.concat(frames, ignore_index=True)
    if "entry_time" in df.columns:
        df = df.drop_duplicates(subset=["symbol", "interval", "entry_time", "exit_time"], keep="last")
    return df


def save_paper_trades(df):
    st.session_state["paper_trades"] = df.to_dict("records")
    try:
        DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(TRADES_PATH, index=False)
    except Exception:
        pass


def append_paper_trades(rows):
    if not rows:
        return
    old = load_paper_trades()
    combined = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    if "entry_time" in combined.columns:
        combined = combined.drop_duplicates(
            subset=["symbol", "interval", "entry_time", "exit_time"], keep="last"
        )
    save_paper_trades(combined)


def simulate_paper(df, asset, symbol, interval, htf, strict_mode, min_conf_high_only=False):
    """Walk bars: enter BUY, exit SELL or stop."""
    rows = []
    in_pos = False
    entry = entry_t = stop = target = conf = score = None
    for ts, row in df.iterrows():
        # Rebuild minimal decision fields from row signals already on df if present
        sig = row.get("Signal")
        if sig is None:
            continue
        if not in_pos and sig == "BUY":
            c = str(row.get("Conf", "MED"))
            if min_conf_high_only and c != "HIGH":
                continue
            in_pos = True
            entry = float(row["Close"])
            entry_t = ts
            stop = float(row["Stop"]) if pd.notna(row.get("Stop")) else entry * 0.98
            target = float(row["Exit"]) if pd.notna(row.get("Exit")) else entry * 1.02
            conf = c
            score = float(row["Score"]) if pd.notna(row.get("Score")) else None
        elif in_pos:
            px = float(row["Close"])
            hit_stop = px <= stop
            hit_tgt = px >= target
            sell = sig == "SELL"
            if hit_stop or hit_tgt or sell:
                pnl = (px - entry) / entry * 100
                reason = "STOP" if hit_stop else ("TARGET" if hit_tgt else "SELL")
                rows.append({
                    "logged_at": datetime.now(timezone.utc).isoformat(),
                    "asset": asset,
                    "symbol": symbol,
                    "interval": interval,
                    "side": "BUY",
                    "entry_time": str(entry_t),
                    "exit_time": str(ts),
                    "entry": round(entry, 8),
                    "exit": round(px, 8),
                    "stop": round(stop, 8),
                    "pnl_pct": round(pnl, 3),
                    "exit_reason": reason,
                    "result": "WIN" if pnl > 0 else "LOSS",
                    "confidence": conf,
                    "htf": htf,
                    "score": score,
                    "strict": bool(strict_mode),
                })
                in_pos = False
    return rows


def get_fred_key():
    try:
        k = st.secrets.get("FRED_API_KEY", None)
        if k:
            return str(k).strip()
    except Exception:
        pass
    return os.environ.get("FRED_API_KEY", "").strip() or None


def get_llm_config():
    """
    Optional LLM for news sentiment layer 2.
    Secrets (any one of):
      XAI_API_KEY + optional XAI_BASE_URL (default https://api.x.ai/v1)
      OPENAI_API_KEY
      LLM_API_KEY + optional LLM_BASE_URL + LLM_MODEL
    """
    def sec(name):
        try:
            v = st.secrets.get(name, None)
            if v:
                return str(v).strip()
        except Exception:
            pass
        return os.environ.get(name, "").strip() or None

    xai = sec("XAI_API_KEY")
    if xai:
        return {
            "key": xai,
            "base": sec("XAI_BASE_URL") or "https://api.x.ai/v1",
            "model": sec("LLM_MODEL") or "grok-3-mini",
            "provider": "xai",
        }
    oai = sec("OPENAI_API_KEY")
    if oai:
        return {
            "key": oai,
            "base": sec("LLM_BASE_URL") or "https://api.openai.com/v1",
            "model": sec("LLM_MODEL") or "gpt-4o-mini",
            "provider": "openai",
        }
    llm = sec("LLM_API_KEY")
    if llm:
        return {
            "key": llm,
            "base": sec("LLM_BASE_URL") or "https://api.openai.com/v1",
            "model": sec("LLM_MODEL") or "gpt-4o-mini",
            "provider": "custom",
        }
    return None


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


@st.cache_data(ttl=12, show_spinner=False)
def binance_ticker_price(symbol):
    for host in BINANCE:
        try:
            r = requests.get(f"{host}/api/v3/ticker/price", params={"symbol": symbol}, timeout=8)
            if r.status_code == 200:
                return float(r.json()["price"])
        except Exception:
            continue
    return None


@st.cache_data(ttl=20, show_spinner=False)
def binance_klines(symbol, interval, limit=300):
    params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
    for host in BINANCE:
        try:
            r = requests.get(f"{host}/api/v3/klines", params=params, timeout=10)
            if r.status_code != 200:
                continue
            raw = r.json()
            if not isinstance(raw, list) or len(raw) < 40:
                continue
            df = pd.DataFrame(raw, columns=[
                "t", "Open", "High", "Low", "Close", "Volume",
                "T", "q", "n", "V", "Q", "x"
            ])
            df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_localize(None)
            df = df.set_index("t")
            for c in ["Open", "High", "Low", "Close", "Volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        except Exception:
            continue
    return pd.DataFrame()


@st.cache_data(ttl=90, show_spinner=False)
def fear_greed():
    try:
        d = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8).json()["data"][0]
        return int(d["value"]), d["value_classification"]
    except Exception:
        return None, None


@st.cache_data(ttl=120, show_spinner=False)
def coingecko_global():
    try:
        g = requests.get("https://api.coingecko.com/api/v3/global", timeout=10).json()["data"]
        return {
            "mcap_chg_24h": g.get("market_cap_change_percentage_24h_usd"),
            "btc_dom": g.get("market_cap_percentage", {}).get("btc"),
        }
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def fred_series(series_id, api_key):
    if not api_key:
        return None
    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 5,
            },
            timeout=12,
        )
        if r.status_code != 200:
            return None
        for o in r.json().get("observations", []):
            if o.get("value") not in (None, "."):
                return float(o["value"])
    except Exception:
        return None
    return None


@st.cache_data(ttl=600, show_spinner=False)
def yahoo_last(symbol):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=5d&interval=1d"
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            return float(r.json()["chart"]["result"][0]["meta"]["regularMarketPrice"])
    except Exception:
        pass
    return None


@st.cache_data(ttl=600, show_spinner=False)
def news_headlines(query="cryptocurrency", limit=6):
    items = []
    feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        f"https://news.google.com/rss/search?q={query.replace(' ', '+')}+crypto&hl=en-US&gl=US&ceid=US:en",
    ]
    for url in feeds:
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": "SpotDesk/1.0"})
            if r.status_code != 200:
                continue
            root = ET.fromstring(r.content)
            for item in root.iter("item"):
                title = (item.findtext("title") or "").strip()
                link = item.findtext("link") or ""
                if title:
                    items.append({"title": title, "link": link})
                if len(items) >= limit:
                    return items
        except Exception:
            continue
    return items[:limit]


def score_news(headlines):
    """
    Rule-based safety layer. Returns dict:
      score: -1..+1 (avg), label, n, hits
    Strongly negative → hard-block buys.
    """
    if not headlines:
        return {"score": None, "label": "no data", "n": 0, "hits": []}

    neg = [
        "hack", "hacked", "exploit", "exploited", "breach", "stolen", "drain",
        "sec charges", "sec sues", "lawsuit", "indictment", "arrest",
        "ban", "banned", "crackdown", "illegal", "fraud", "ponzi",
        "insolvent", "insolvency", "bankrupt", "bankruptcy", "default",
        "collapse", "crash", "liquidation cascade", "delist", "delisted",
        "halt", "halted", "outage", "insolven", "rug pull", "rugpull",
        "scandal", "investigation", "probe", "subpoena",
        "terror", "war", "sanctions", "seizure", "freeze",
        "depeg", "de-peg", "run on", "contagion",
    ]
    pos = [
        "etf approved", "etf approval", "approved etf",
        "partnership", "partners with", "integrates", "integration",
        "upgrade", "mainnet", "launch", "all-time high", "ath ",
        "record inflow", "inflows", "institutional", "adoption",
        "bullish", "surge", "rally", "breakthrough",
    ]

    scores = []
    hits = []
    for h in headlines:
        title = (h.get("title") or "").lower()
        s = 0.0
        matched = []
        for w in neg:
            if w in title:
                s -= 1.0
                matched.append("-" + w)
        for w in pos:
            if w in title:
                s += 0.6
                matched.append("+" + w)
        # clamp per headline
        s = max(-1.0, min(1.0, s))
        scores.append(s)
        if matched:
            hits.append((h.get("title", "")[:90], s, matched[:4]))

    avg = sum(scores) / len(scores) if scores else 0.0
    if avg <= -0.35:
        label = "TOXIC"
    elif avg <= -0.12:
        label = "NEGATIVE"
    elif avg >= 0.25:
        label = "POSITIVE"
    else:
        label = "NEUTRAL"

    return {"score": round(avg, 3), "label": label, "n": len(scores), "hits": hits[:5]}


@st.cache_data(ttl=300, show_spinner=False)
def score_news_llm(headline_titles, api_key, base_url, model):
    """
    Layer 2: LLM overall market/news risk for trading.
    Returns score -1..+1 and short reason. Fail-soft → None.
    """
    if not api_key or not headline_titles:
        return None
    titles = [t for t in headline_titles if t][:10]
    prompt = (
        "You are a risk filter for a crypto SPOT trader (buy low / sell high). "
        "Given headlines, rate overall news risk for opening a NEW long spot buy.\n"
        "Reply ONLY JSON: {\"score\": <float -1 to 1>, \"label\": \"TOXIC\"|"
        "\"NEGATIVE\"\"NEUTRAL\"\"POSITIVE\", \"reason\": \"<=12 words\"}\n"
        "-1 = severe risk (hack, ban, insolvency, major shock). "
        "0 = mixed/irrelevant. +1 = clearly constructive.\n"
        "Do not recommend trades. Headlines:\n- "
        + "\n- ".join(titles)
    )
    try:
        r = requests.post(
            base_url.rstrip("/") + "/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": 0.1,
                "max_tokens": 120,
                "messages": [
                    {"role": "system", "content": "Return only valid JSON. No markdown."},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=20,
        )
        if r.status_code >= 300:
            return {"score": None, "label": "LLM error", "reason": f"HTTP {r.status_code}", "ok": False}
        content = r.json()["choices"][0]["message"]["content"].strip()
        # strip fences if any
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:].strip()
        import re
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            return {"score": None, "label": "LLM parse", "reason": content[:80], "ok": False}
        data = json.loads(m.group(0))
        sc = float(data.get("score", 0))
        sc = max(-1.0, min(1.0, sc))
        lab = str(data.get("label", "NEUTRAL")).upper()
        if lab not in ("TOXIC", "NEGATIVE", "NEUTRAL", "POSITIVE"):
            if sc <= -0.35:
                lab = "TOXIC"
            elif sc <= -0.12:
                lab = "NEGATIVE"
            elif sc >= 0.25:
                lab = "POSITIVE"
            else:
                lab = "NEUTRAL"
        return {
            "score": round(sc, 3),
            "label": lab,
            "reason": str(data.get("reason", ""))[:120],
            "ok": True,
        }
    except Exception as e:
        return {"score": None, "label": "LLM error", "reason": str(e)[:80], "ok": False}


def merge_news_sentiment(rule_sent, llm_sent):
    """
    Combine rule + LLM. Conservative: toxic if either is toxic.
    Blended score favors the more negative reading (safety layer).
    """
    out = {
        "score": None,
        "label": "no data",
        "n": rule_sent.get("n", 0) if rule_sent else 0,
        "hits": rule_sent.get("hits", []) if rule_sent else [],
        "rule": rule_sent,
        "llm": llm_sent,
        "blend": None,
    }
    r = rule_sent.get("score") if rule_sent else None
    l = llm_sent.get("score") if llm_sent and llm_sent.get("ok") else None

    if r is None and l is None:
        return out

    if r is not None and l is not None:
        # safety: weight the worse (more negative) side slightly more
        blend = 0.45 * r + 0.55 * l if l < r else 0.55 * r + 0.45 * l
        # if either toxic, force toxic ceiling
        if (rule_sent.get("label") == "TOXIC") or (llm_sent.get("label") == "TOXIC") or min(r, l) <= -0.35:
            blend = min(blend, -0.35)
            label = "TOXIC"
        elif min(r, l) <= -0.12 or "NEGATIVE" in (rule_sent.get("label"), llm_sent.get("label")):
            label = "NEGATIVE" if blend <= -0.05 else rule_sent.get("label") or llm_sent.get("label")
            if blend <= -0.12:
                label = "NEGATIVE"
        elif max(r, l) >= 0.25 and min(r, l) >= 0:
            label = "POSITIVE"
        else:
            label = "NEUTRAL"
        out["score"] = round(float(blend), 3)
        out["label"] = label
        out["blend"] = out["score"]
    elif r is not None:
        out["score"] = r
        out["label"] = rule_sent.get("label", "NEUTRAL")
    else:
        out["score"] = l
        out["label"] = llm_sent.get("label", "NEUTRAL")

    return out


def send_webhook(url, content):
    if not url or not url.startswith("http"):
        return False
    try:
        r = requests.post(url, json={"content": content}, timeout=8)
        return r.status_code < 300
    except Exception:
        return False


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, min_periods=n).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, min_periods=n).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

def atr_s(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def trend_bias(df):
    if df.empty or len(df) < 30:
        return "RANGE"
    c = df["Close"]
    e12, e26 = ema(c, 12), ema(c, 26)
    e50 = ema(c, min(50, max(20, len(df) // 2)))
    last = c.iloc[-1]
    if e12.iloc[-1] > e26.iloc[-1] > e50.iloc[-1] and last > e26.iloc[-1]:
        return "UP"
    if e12.iloc[-1] < e26.iloc[-1] < e50.iloc[-1] and last < e26.iloc[-1]:
        return "DOWN"
    return "RANGE"


def technicals(df):
    d = df.copy()
    c = d["Close"]
    d["EMA12"] = ema(c, 12)
    d["EMA26"] = ema(c, 26)
    d["EMA50"] = ema(c, 50) if len(d) >= 50 else ema(c, max(20, len(d) // 3))
    d["RSI"] = rsi(c)
    d["ATR"] = atr_s(d)
    # ATR as % of price — high = chaotic tape
    d["ATR_pct"] = (d["ATR"] / c.replace(0, np.nan)) * 100
    mid, std = c.rolling(20).mean(), c.rolling(20).std()
    d["BB_u"], d["BB_l"] = mid + 2 * std, mid - 2 * std
    d["BB_pct"] = (c - d["BB_l"]) / (d["BB_u"] - d["BB_l"]).replace(0, np.nan)
    # BB bandwidth (squeeze when narrow vs its own history)
    d["BB_width"] = (d["BB_u"] - d["BB_l"]) / mid.replace(0, np.nan)
    width_med = d["BB_width"].rolling(50, min_periods=20).median()
    d["BB_squeeze"] = d["BB_width"] < (width_med * 0.75)
    d["Swing_low"] = d["Low"].rolling(20).min()
    d["Swing_high"] = d["High"].rolling(20).max()
    d["Vol_MA"] = d["Volume"].rolling(20).mean()
    d["Vol_ratio"] = d["Volume"] / d["Vol_MA"].replace(0, np.nan)
    d["RSI_rising"] = d["RSI"] > d["RSI"].shift(2)
    d["MACD"] = d["EMA12"] - d["EMA26"]
    d["MACD_sig"] = ema(d["MACD"], 9)
    d["MACD_up"] = d["MACD"] > d["MACD_sig"]
    # recent return for relative strength vs BTC
    d["ret_6"] = c.pct_change(6)
    return d.dropna()


def load_macro(fred_key):
    dgs10 = fred_series("DGS10", fred_key) if fred_key else None
    fed = fred_series("FEDFUNDS", fred_key) if fred_key else None
    unrate = fred_series("UNRATE", fred_key) if fred_key else None
    if dgs10 is None:
        dgs10 = yahoo_last("^TNX")
    if fed is None:
        fed = yahoo_last("^IRX")
    return {"DGS10": dgs10, "FEDFUNDS": fed, "UNRATE": unrate}


def decide(last, htf, fng_val, fng_label, global_m, macro, rel_vs_btc=None, news_sent=None, strict_mode=False):
    """
    Stricter spot decision engine.
    - Required: value (BB or RSI) + HTF not DOWN
    - Soft blocks: extreme greed + not cheap, hard market crash
    - Weighted score; BUY only on strong score
    - Missing optional data simply omitted
    """
    price = float(last["Close"])
    rsi_v = float(last["RSI"])
    bb = float(last["BB_pct"]) if pd.notna(last["BB_pct"]) else 0.5
    atr = float(last["ATR"]) if pd.notna(last["ATR"]) else price * 0.01
    local_up = float(last["EMA12"]) >= float(last["EMA26"])
    above_50 = float(last["Close"]) >= float(last["EMA50"]) * 0.98
    vol_ratio = float(last.get("Vol_ratio") or 1)
    rsi_rising = bool(last.get("RSI_rising", False))
    macd_up = bool(last.get("MACD_up", False))
    atr_pct = float(last["ATR_pct"]) if pd.notna(last.get("ATR_pct")) else None
    bb_squeeze = bool(last.get("BB_squeeze", False))

    stop = min(price - 1.5 * atr, float(last["Swing_low"]) * 0.998)
    risk = max(price - stop, atr * 0.4)
    exit_px = price + 2.0 * risk

    # Weighted factors: (name, weight, pass?, detail)
    factors = []

    # --- CORE (high weight) ---
    at_value = bb <= 0.35 or rsi_v <= 38
    factors.append(("At value (BB low or RSI low)", 3.0, at_value, f"BB {bb*100:.0f}% · RSI {rsi_v:.0f}"))

    htf_ok = htf in ("UP", "RANGE")
    factors.append(("HTF allows buys (not DOWN)", 3.0, htf_ok, f"HTF={htf}"))

    # avoid catching knives; under BB squeeze demand a clearer turn
    turn_ok = rsi_rising or macd_up or (rsi_v < 28 and htf == "RANGE")
    if bb_squeeze:
        turn_ok = (rsi_rising and macd_up) or (rsi_rising and rsi_v < 30)
    factors.append(("Momentum not still collapsing", 2.0, turn_ok,
                    f"RSI↑={rsi_rising} MACD↑={macd_up}" + (" · squeeze" if bb_squeeze else "")))

    # --- IMPORTANT ---
    factors.append(("RSI not overbought", 1.5, rsi_v < 60, f"RSI {rsi_v:.0f}"))
    factors.append(("Local structure OK", 1.5, local_up or htf == "UP" or rsi_v < 30, "EMA / HTF / deep OS"))
    factors.append(("Not extended above value", 1.5, bb < 0.55 or rsi_v < 45, f"BB {bb*100:.0f}%"))

    # volume: very dead tape is a soft negative, not a hard fail
    factors.append(("Volume not dead", 1.0, vol_ratio >= 0.45, f"{vol_ratio:.1f}x MA"))

    # --- ATR% volatility filter ---
    if atr_pct is not None:
        # typical crypto 15m-1h: block extreme chaos (tune by TF loosely)
        atr_ok = atr_pct < 4.5  # >~4.5% ATR/price = very wild for entries
        factors.append(("Volatility not extreme (ATR%)", 2.0, atr_ok, f"ATR% {atr_pct:.2f}"))

    # --- BB squeeze: informational + soft weight; hard require is in turn_ok ---
    factors.append(
        ("BB squeeze handled", 1.0, (not bb_squeeze) or turn_ok,
         "Squeeze ON — need stronger turn" if bb_squeeze else "No squeeze")
    )

    # --- vs BTC relative strength (alts only; BTC vs self skipped) ---
    if rel_vs_btc is not None:
        # rel_vs_btc = coin_ret - btc_ret over same window; negative = weaker than BTC
        vs_ok = rel_vs_btc > -0.03  # not underperforming BTC by >3% on short window
        factors.append(
            ("Not much weaker than BTC", 2.0, vs_ok, f"vs BTC {rel_vs_btc*100:+.1f}%")
        )

    # --- CONTEXT (only if data exists) ---
    if fng_val is not None:
        # Greed is OK only if we're truly at value; extreme greed blocks weak setups
        fng_ok = fng_val < 80 or (fng_val < 90 and at_value and rsi_v < 32)
        factors.append(("Sentiment not reckless greed", 1.5, fng_ok, f"{fng_val} {fng_label}"))
        factors.append(("Fear supports buy-low", 1.0, fng_val <= 45, f"{fng_val} {fng_label}"))

    if global_m.get("mcap_chg_24h") is not None:
        chg = float(global_m["mcap_chg_24h"])
        factors.append(("Broad market not crashing", 2.0, chg > -6.0, f"Mcap 24h {chg:+.1f}%"))

    if macro.get("DGS10") is not None:
        factors.append(("Yields not in spike panic zone", 1.0, macro["DGS10"] < 5.75, f"10Y {macro['DGS10']:.2f}%"))

    # --- News safety layer ---
    if news_sent and news_sent.get("score") is not None:
        ns = float(news_sent["score"])
        lab = news_sent.get("label", "")
        # pass if not clearly negative
        news_ok = ns > -0.12
        factors.append(
            ("News not toxic", 2.5, news_ok, f"{lab} ({ns:+.2f}) from {news_sent.get('n', 0)} headlines")
        )

    # Score
    score = sum(w for _, w, ok, _ in factors if ok)
    max_score = sum(w for _, w, _, _ in factors)
    ratio = score / max_score if max_score else 0

    # Hard blocks
    hard_block = []
    if htf == "DOWN" and rsi_v > 32:
        hard_block.append("HTF DOWN and not deeply oversold")
    if fng_val is not None and fng_val >= 85 and bb > 0.4:
        hard_block.append("Extreme greed + not cheap")
    if global_m.get("mcap_chg_24h") is not None and float(global_m["mcap_chg_24h"]) <= -8:
        hard_block.append("Severe risk-off day (mcap)")
    if atr_pct is not None and atr_pct >= 6.0:
        hard_block.append(f"ATR% too high ({atr_pct:.2f}) — wait for calmer tape")
    if bb_squeeze and not turn_ok:
        hard_block.append("BB squeeze without momentum turn")
    if news_sent and news_sent.get("score") is not None:
        if float(news_sent["score"]) <= -0.35 or news_sent.get("label") == "TOXIC":
            hard_block.append(f"Toxic news sentiment ({news_sent.get('label')})")

    # SELL HIGH: take profit / exit zone for holders
    sell_high = (
        (rsi_v >= 68 and bb >= 0.7)
        or (rsi_v >= 72)
        or (bb >= 0.88)
        or (fng_val is not None and fng_val >= 80 and bb >= 0.65 and rsi_v >= 60)
    )

    # BUY: must be at value, pass HTF rule, no hard block, strong score
    min_ratio = 0.72 if strict_mode else 0.62
    min_score = 9.0 if strict_mode else 7.0
    buy_low = (
        not sell_high
        and not hard_block
        and at_value
        and htf_ok
        and ratio >= min_ratio
        and score >= min_score
    )
    # Strict: only UP or RANGE already; also require turn + no weak volume
    if strict_mode and buy_low:
        if not turn_ok:
            buy_low = False
        if vol_ratio < 0.55:
            buy_low = False
        if atr_pct is not None and atr_pct >= 4.0:
            buy_low = False
        if htf == "RANGE" and rsi_v > 34:
            # stricter value in chop
            if not (bb <= 0.28 or rsi_v <= 32):
                buy_low = False

    if buy_low:
        action = "BUY"
        if strict_mode:
            conf = "HIGH" if (ratio >= 0.80 and turn_ok) else "MED"
            # Strict mode: surface only HIGH as actionable BUY
            if conf != "HIGH":
                action = "WAIT"
                conf = "MED"  # shown as not strict-qualified
        else:
            conf = "HIGH" if ratio >= 0.78 and turn_ok else "MED"
    elif sell_high:
        action = "SELL"
        exit_px = price
        stop = None
        conf = "HIGH" if rsi_v >= 72 or bb >= 0.85 else "MED"
    else:
        action = "WAIT"
        conf = "—"

    checklist = [(n, ok, d) for n, w, ok, d in factors]

    return {
        "action": action,
        "confidence": conf,
        "entry": price if action == "BUY" else None,
        "exit": exit_px if action != "WAIT" else float(last["BB_u"]),
        "stop": stop,
        "buy_zone": float(last["BB_l"]),
        "sell_zone": float(last["BB_u"]),
        "limit_buy": float(last["BB_l"]),
        "checks": checklist,
        "score": round(score, 1),
        "max_score": round(max_score, 1),
        "ratio": ratio,
        "hard_block": hard_block,
        "rsi": rsi_v,
        "bb": bb,
        "atr": atr,
        "atr_pct": atr_pct,
        "bb_squeeze": bb_squeeze,
        "rel_vs_btc": rel_vs_btc,
        "news_sent": news_sent,
        "strict_mode": strict_mode,
        "price": price,
    }


def chart_price(df):
    plot = df[["Close"]].tail(120).reset_index()
    plot.columns = ["Time", "Price"]
    ymin, ymax = float(plot["Price"].min()), float(plot["Price"].max())
    pad = (ymax - ymin) * 0.08 or 1
    ch = (
        alt.Chart(plot).mark_line(color="#42a5f5", strokeWidth=2)
        .encode(
            x=alt.X("Time:T", title=None),
            y=alt.Y("Price:Q", scale=alt.Scale(domain=[ymin - pad, ymax + pad], zero=False), title=None),
            tooltip=["Time:T", alt.Tooltip("Price:Q", format=".4f")],
        )
        .properties(height=300)
        .interactive()
    )
    st.altair_chart(ch, use_container_width=True)


# ── Sidebar ───────────────────────────────────────────────────
fred_key = get_fred_key()
st.sidebar.title("Spot Desk")
name = st.sidebar.selectbox("Coin", list(CRYPTO.keys()), index=0)
symbol = CRYPTO[name]
custom = st.sidebar.text_input("Or pair", placeholder="BTCUSDT")
if custom.strip():
    symbol = custom.strip().upper()
    name = symbol

interval = st.sidebar.selectbox("Timeframe", INTERVALS, index=0)
account = st.sidebar.number_input("Account $", 100.0, value=10000.0, step=100.0)
risk_pct = st.sidebar.slider("Risk % if stopped", 0.25, 2.0, 0.5, 0.25)

st.sidebar.markdown("---")
live = st.sidebar.checkbox("🔴 Live feed ON", value=False)
secs = st.sidebar.slider("Refresh seconds", 20, 120, 45)
alert_on = st.sidebar.checkbox("Alert on BUY / SELL", value=True)
webhook = st.sidebar.text_input("Webhook (optional)", placeholder="Discord webhook URL")

if fred_key:
    st.sidebar.success("FRED key loaded")
else:
    st.sidebar.caption("Optional: secret FRED_API_KEY for macro")

_llm = get_llm_config()
if _llm:
    st.sidebar.success(f"LLM news layer: {_llm['provider']}")
else:
    st.sidebar.caption("Optional: XAI_API_KEY or OPENAI_API_KEY for AI news layer")

if st.sidebar.button("Refresh now"):
    st.cache_data.clear()
    st.rerun()

# ── Paper log page ────────────────────────────────────────────
if page == "Paper log":
    st.title("Paper trade log")
    st.caption("Simulated spot round-trips from the Trade page. Export often on Streamlit Cloud.")
    log = load_paper_trades()
    up = st.file_uploader("Import CSV", type=["csv"])
    if up is not None:
        try:
            incoming = pd.read_csv(up)
            merged = pd.concat([log, incoming], ignore_index=True)
            save_paper_trades(merged)
            st.success("Imported")
            st.rerun()
        except Exception as e:
            st.error(e)

    closed = log[log["result"].isin(["WIN", "LOSS"])] if not log.empty and "result" in log.columns else pd.DataFrame()
    if closed.empty:
        st.info("No closed paper trades yet. Open **Trade**, load a coin — simulations append here.")
    else:
        wins = closed[closed["result"] == "WIN"]
        losses = closed[closed["result"] == "LOSS"]
        wr = len(wins) / len(closed) * 100
        exp = closed["pnl_pct"].mean()
        avg_w = wins["pnl_pct"].mean() if len(wins) else 0
        avg_l = losses["pnl_pct"].mean() if len(losses) else 0
        gp = wins["pnl_pct"].sum() if len(wins) else 0
        gl = abs(losses["pnl_pct"].sum()) if len(losses) else 0
        pf = (gp / gl) if gl > 0 else float("inf")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Trades", len(closed))
        m2.metric("Win rate", f"{wr:.0f}%")
        m3.metric("Expectancy", f"{exp:+.2f}%")
        m4.metric("Avg win / loss", f"{avg_w:+.2f}% / {avg_l:+.2f}%")
        m5.metric("Profit factor", f"{pf:.2f}" if pf != float("inf") else "∞")

        if "confidence" in closed.columns:
            st.subheader("By confidence")
            st.dataframe(
                closed.groupby("confidence").agg(
                    n=("pnl_pct", "count"),
                    win_rate=("result", lambda s: (s == "WIN").mean() * 100),
                    avg_pnl=("pnl_pct", "mean"),
                ).round(2),
                use_container_width=True,
            )
        if "htf" in closed.columns:
            st.subheader("By HTF")
            st.dataframe(
                closed.groupby("htf").agg(
                    n=("pnl_pct", "count"),
                    win_rate=("result", lambda s: (s == "WIN").mean() * 100),
                    avg_pnl=("pnl_pct", "mean"),
                ).round(2),
                use_container_width=True,
            )
        if "strict" in closed.columns:
            st.subheader("Strict vs normal")
            st.dataframe(
                closed.groupby("strict").agg(
                    n=("pnl_pct", "count"),
                    win_rate=("result", lambda s: (s == "WIN").mean() * 100),
                    avg_pnl=("pnl_pct", "mean"),
                ).round(2),
                use_container_width=True,
            )
        st.subheader("Trades")
        st.dataframe(closed.sort_values("logged_at", ascending=False), use_container_width=True, height=360)
        st.download_button(
            "Download CSV",
            closed.to_csv(index=False).encode(),
            file_name="paper_trades.csv",
            mime="text/csv",
        )
    if st.button("Clear paper log"):
        save_paper_trades(pd.DataFrame())
        st.rerun()
    st.stop()

# ── Data ──────────────────────────────────────────────────────
now = datetime.now(timezone.utc)
st.title(f"{name}")
if strict_mode:
    st.caption("STRICT MODE — only HIGH-quality BUYs")
if live:
    st.markdown(
        f"<span class='live-dot'>● LIVE</span> · {interval} · {now.strftime('%H:%M:%S')} UTC",
        unsafe_allow_html=True,
    )
else:
    st.caption(f"{interval} · {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")

with st.spinner("Analyzing…"):
    live_px = binance_ticker_price(symbol)
    raw = binance_klines(symbol, interval, 300)
    htf_raw = binance_klines(symbol, HTF_MAP.get(interval, "1h"), 150)
    htf = trend_bias(htf_raw) if not htf_raw.empty else "RANGE"
    fng_val, fng_label = fear_greed()
    global_m = coingecko_global()
    macro = load_macro(fred_key)
    news = news_headlines(name if name.replace(" ", "").isalpha() else "crypto", 8)
    rule_sent = score_news(news)
    llm_cfg = get_llm_config()
    llm_sent = None
    if llm_cfg and news:
        titles = [n.get("title") or "" for n in news]
        llm_sent = score_news_llm(
            tuple(titles),  # tuple for cache hash
            llm_cfg["key"],
            llm_cfg["base"],
            llm_cfg["model"],
        )
    news_sent = merge_news_sentiment(rule_sent, llm_sent)
    # BTC same timeframe for relative strength (skip if symbol is BTC)
    btc_raw = pd.DataFrame()
    if symbol != "BTCUSDT":
        btc_raw = binance_klines("BTCUSDT", interval, 80)

if raw.empty:
    st.error("No price data.")
    st.stop()

df = technicals(raw)
if df.empty:
    st.error("Not enough data.")
    st.stop()

last = df.iloc[-1]
# Relative strength vs BTC over ~6 bars
rel_vs_btc = None
if symbol != "BTCUSDT" and not btc_raw.empty and "ret_6" in df.columns:
    try:
        btc_t = technicals(btc_raw)
        if not btc_t.empty and "ret_6" in btc_t.columns:
            coin_r = float(last["ret_6"]) if pd.notna(last["ret_6"]) else None
            btc_r = float(btc_t.iloc[-1]["ret_6"]) if pd.notna(btc_t.iloc[-1]["ret_6"]) else None
            if coin_r is not None and btc_r is not None:
                rel_vs_btc = coin_r - btc_r
    except Exception:
        rel_vs_btc = None
dec = decide(last, htf, fng_val, fng_label, global_m, macro, rel_vs_btc=rel_vs_btc, news_sent=news_sent, strict_mode=strict_mode)

# Live price for entry/exit display
if live_px is not None:
    dec["price"] = live_px
    if dec["action"] == "BUY":
        dec["entry"] = live_px
        atr = dec["atr"]
        stop = min(live_px - 1.5 * atr, float(last["Swing_low"]) * 0.998)
        risk = max(live_px - stop, atr * 0.4)
        dec["stop"] = stop
        dec["exit"] = live_px + 2.0 * risk
    elif dec["action"] == "SELL":
        dec["exit"] = live_px

action = dec["action"]
prev = st.session_state.get("last_action")

if alert_on and action in ("BUY", "SELL"):
    if action == "BUY":
        st.markdown(
            f"<div class='alert-buy'>🔔 BUY — market {fmt(dec['entry'])} · sell target {fmt(dec['exit'])} · conf {dec['confidence']}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='alert-sell'>🔔 SELL — exit {fmt(dec['exit'])} · conf {dec['confidence']}</div>",
            unsafe_allow_html=True,
        )
    if prev != action:
        st.toast(f"{action} {name}", icon="🔔")
        if webhook.strip():
            send_webhook(
                webhook.strip(),
                f"{action} {name} @ {fmt(dec['price'])} entry={fmt(dec.get('entry'))} exit={fmt(dec.get('exit'))} conf={dec['confidence']}",
            )
st.session_state["last_action"] = action

# Paper trades: simple walk-forward using indicator columns (fast)
def _quick_tag(frame, htf_bias, strict):
    out = frame.copy()
    rsi_v = out["RSI"]
    bb = out["BB_pct"]
    atr_pct = out.get("ATR_pct", pd.Series(1.0, index=out.index))
    turn = out.get("RSI_rising", False) | out.get("MACD_up", False)
    vol = out.get("Vol_ratio", 1.0)
    htf_ok = htf_bias in ("UP", "RANGE")
    at_val = (bb <= 0.35) | (rsi_v <= 38)
    if strict:
        at_val = (bb <= 0.30) | (rsi_v <= 34)
    sell = (rsi_v >= 68) & (bb >= 0.70) | (rsi_v >= 72) | (bb >= 0.88)
    buy = at_val & htf_ok & (~sell)
    if strict:
        buy = buy & turn & (vol >= 0.55) & (atr_pct < 4.0)
    conf = np.where(buy & turn & (rsi_v <= 32), "HIGH", np.where(buy, "MED", "—"))
    if strict:
        buy = buy & (conf == "HIGH")
    out["Signal"] = np.where(sell, "SELL", np.where(buy, "BUY", "WAIT"))
    out["Conf"] = conf
    risk = (out["Close"] - out["Close"] + out["ATR"] * 1.5).clip(lower=out["Close"] * 0.002)
    out["Stop"] = out["Close"] - 1.5 * out["ATR"]
    out["Exit"] = out["Close"] + 2.0 * risk
    out["Score"] = np.where(buy, 10, 0)
    return out

_sim_n = min(150, len(df))
_tagged = _quick_tag(df.tail(_sim_n), htf, strict_mode)
_paper_rows = simulate_paper(
    _tagged, name, symbol, interval, htf, strict_mode,
    min_conf_high_only=strict_mode,
)
if _paper_rows:
    append_paper_trades(_paper_rows)


size = 0.0
if action == "BUY" and dec.get("stop") and dec.get("entry"):
    risk_amt = dec["entry"] - dec["stop"]
    if risk_amt > 0:
        size = (account * risk_pct / 100.0) / risk_amt

# Decision card
if action == "BUY":
    lim = dec.get("limit_buy")
    lim_h = f"<p class='small'>Limit idea (below market): {fmt(lim)}</p>" if lim and lim < dec["entry"] else ""
    st.markdown(
        f"""
        <div class="card">
          <div class="small">{('STRICT · ' if strict_mode else '')}BEST EFFORT · conf {dec['confidence']}</div>
          <div class="big buy">BUY</div>
          <p style="font-size:1.15rem;margin-top:0.8rem">
            <b>Entry</b> (buy at market)<br><span class="big buy">{fmt(dec['entry'])}</span>
          </p>
          <p style="font-size:1.15rem">
            <b>Exit target</b> (sell later — above entry)<br><span class="big sell">{fmt(dec['exit'])}</span>
          </p>
          {lim_h}
          <p class="small">Stop {fmt(dec['stop'])} · Size ≈ {size:.6g} · Score {dec['score']}/{dec['max_score']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif action == "SELL":
    st.markdown(
        f"""
        <div class="card">
          <div class="small">{('STRICT · ' if strict_mode else '')}BEST EFFORT · conf {dec['confidence']}</div>
          <div class="big sell">SELL</div>
          <p style="font-size:1.15rem;margin-top:0.8rem">
            <b>Exit</b> (sell coin for cash)<br><span class="big sell">{fmt(dec['exit'])}</span>
          </p>
          <p class="small">Next buy zone ≈ {fmt(dec['buy_zone'])}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    reason = "; ".join(dec["hard_block"]) if dec["hard_block"] else "Setup not strong enough"
    st.markdown(
        f"""
        <div class="card">
          <div class="small">BEST EFFORT SPOT CALL</div>
          <div class="big wait">WAIT</div>
          <p>Price <b>{fmt(dec['price'])}</b></p>
          <p>Buy zone ≈ {fmt(dec['buy_zone'])} · Sell zone ≈ {fmt(dec['sell_zone'])}</p>
          <p class="small">Score {dec['score']}/{dec['max_score']} · {reason}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Price", fmt(dec["price"]))
c2.metric("RSI", f"{dec['rsi']:.0f}")
c3.metric("HTF", htf)
c4.metric("Fear & Greed", f"{fng_val}" if fng_val is not None else "—", fng_label)
c5.metric("Score", f"{dec['score']}/{dec['max_score']}")

_plog = load_paper_trades()
if len(_plog):
    _c = _plog[_plog["result"].isin(["WIN", "LOSS"])] if "result" in _plog.columns else _plog
    if len(_c):
        st.caption(
            f"Paper log: **{len(_c)}** closed · win rate **{( _c['result']=='WIN').mean()*100:.0f}%** · "
            f"expectancy **{_c['pnl_pct'].mean():+.2f}%** · see **Paper log** page"
        )

st.subheader("Why (weighted checklist)")
st.caption("Only factors with data are scored. Blank futures feeds are ignored on purpose.")
for label, ok, detail in dec["checks"]:
    st.markdown(
        f"{'<span class=ok>✓</span>' if ok else '<span class=no>✗</span>'} **{label}** — {detail}",
        unsafe_allow_html=True,
    )

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Context")
    st.write(f"BB position: **{dec['bb']*100:.0f}%** · ATR: **{fmt(dec['atr'])}**")
    if dec.get("atr_pct") is not None:
        st.write(f"ATR%: **{dec['atr_pct']:.2f}%**" + (" ⚠️ high" if dec["atr_pct"] >= 4.5 else ""))
    st.write(f"BB squeeze: **{'YES — need turn' if dec.get('bb_squeeze') else 'No'}**")
    if dec.get("rel_vs_btc") is not None:
        st.write(f"vs BTC (6 bars): **{dec['rel_vs_btc']*100:+.1f}%**")
    elif symbol == "BTCUSDT":
        st.write("vs BTC: n/a (this is BTC)")
    if global_m.get("mcap_chg_24h") is not None:
        st.write(f"Global mcap 24h: **{float(global_m['mcap_chg_24h']):+.2f}%**")
    if global_m.get("btc_dom") is not None:
        st.write(f"BTC dominance: **{float(global_m['btc_dom']):.1f}%**")
with col_b:
    st.subheader("Macro")
    st.write(f"10Y: **{macro['DGS10']:.2f}%**" if macro.get("DGS10") is not None else "10Y: —")
    st.write(f"Fed/short: **{macro['FEDFUNDS']:.2f}%**" if macro.get("FEDFUNDS") is not None else "Fed/short: —")
    st.write(f"Unemployment: **{macro['UNRATE']:.2f}%**" if macro.get("UNRATE") is not None else "Unemployment: —")

st.subheader("News sentiment (safety layers)")
if news_sent and news_sent.get("score") is not None:
    lab = news_sent["label"]
    sc = news_sent["score"]
    if lab == "TOXIC":
        st.error(f"**{lab}** · blended {sc:+.2f} · BUYs blocked until risk cools")
    elif lab == "NEGATIVE":
        st.warning(f"**{lab}** · blended {sc:+.2f} · checklist penalty")
    elif lab == "POSITIVE":
        st.success(f"**{lab}** · blended {sc:+.2f} · does not force a BUY")
    else:
        st.info(f"**{lab}** · blended {sc:+.2f}")

    r = news_sent.get("rule") or {}
    l = news_sent.get("llm") or {}
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Layer 1 · rules")
        if r.get("score") is not None:
            st.write(f"**{r.get('label')}** · {r['score']:+.2f}")
        else:
            st.write("—")
    with c2:
        st.caption("Layer 2 · LLM")
        if l and l.get("ok") and l.get("score") is not None:
            st.write(f"**{l.get('label')}** · {l['score']:+.2f}")
            if l.get("reason"):
                st.caption(l["reason"])
        elif l and not l.get("ok"):
            st.write(f"Unavailable · {l.get('reason', l.get('label', ''))}")
        else:
            st.write("— (add XAI_API_KEY or OPENAI_API_KEY)")

    if news_sent.get("hits"):
        with st.expander("Rule keyword hits"):
            for title, s, matched in news_sent["hits"]:
                st.caption(f"({s:+.1f}) {title}")
else:
    st.caption("No news score (feed empty).")

st.subheader("Headlines")
if news:
    for n in news:
        st.markdown(f"- [{n['title']}]({n['link']})" if n.get("link") else f"- {n['title']}")
else:
    st.caption("No headlines.")

st.subheader("Price")
chart_price(df)

with st.expander("How decisions are made"):
    st.markdown("""
**Goal:** fewer, better spot calls — buy low, sell high.

**BUY only if**
- Price is in a **value** zone (low BB or low RSI)
- Higher timeframe is **not DOWN**
- Weighted score is strong (~62%+ of available points)
- **ATR%** not extreme (hard block if ATR% ≥ 6)
- **BB squeeze** only with a real momentum turn
- **vs BTC** not much weaker (alts)
- **News** not toxic (rule-based headline score)
- No hard block (greed + expensive, crash day, toxic news, etc.)

**SELL when** RSI/BB (and sometimes greed) say **expensive** — for people who already hold the coin.

**Ignored when missing:** funding, open interest (often blocked on cloud). They do not tank the score.

Prefer **HIGH** confidence. **WAIT** is a valid decision.
""")

with st.expander("AI news layer secrets (optional)"):
    st.markdown("""
Layer 2 uses an LLM only if you set a secret (never commit keys to GitHub):

**xAI / Grok**
```toml
XAI_API_KEY = "xai-..."
# optional:
# LLM_MODEL = "grok-3-mini"
```

**OpenAI**
```toml
OPENAI_API_KEY = "sk-..."
# LLM_MODEL = "gpt-4o-mini"
```

Streamlit Cloud → app → Settings → Secrets → save → reboot.

Blended sentiment is **conservative**: if either layer is TOXIC, BUYs stay blocked.
""")

st.caption("Not financial advice · You place all orders")

if live:
    time.sleep(secs)
    st.rerun()
