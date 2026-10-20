import yfinance as yf
import pandas as pd
import pandas_ta_classic as ta

def fetch_crypto_data(ticker: str, period: str = "60d", interval: str = "1h") -> pd.DataFrame:
    """Fetches historical market data from Yahoo Finance."""
    crypto = yf.Ticker(ticker)
    df = crypto.history(period=period, interval=interval)
    return df

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates technical indicators and generates buy/sell triggers."""
    if df.empty or len(df) < 50:
        return df

    df.dropna(inplace=True)

    # 1. Trend Tracking (Moving Averages)
    df['EMA_fast'] = ta.ema(df['Close'], length=12)
    df['EMA_slow'] = ta.ema(df['Close'], length=26)

    # 2. Momentum (Relative Strength Index)
    df['RSI'] = ta.rsi(df['Close'], length=14)

    # Initialize blank rows for signals
    df['Signal'] = 'HOLD'
    df['Score'] = 0

    # Calculate signals based on math formulas
    for i in range(1, len(df)):
        score = 0
        
        # Bullish rules (Price looks like it's going UP)
        if df['EMA_fast'].iloc[i] > df['EMA_slow'].iloc[i] and df['EMA_fast'].iloc[i-1] <= df['EMA_slow'].iloc[i-1]:
            score += 2  
        if df['RSI'].iloc[i] < 30:
            score += 1  

        # Bearish rules (Price looks like it's going DOWN)
        if df['EMA_fast'].iloc[i] < df['EMA_slow'].iloc[i] and df['EMA_fast'].iloc[i-1] >= df['EMA_slow'].iloc[i-1]:
            score -= 2  
        if df['RSI'].iloc[i] > 70:
            score -= 1  

        df.at[df.index[i], 'Score'] = score
        
        # Final Signal Assignment
        if score >= 2:
            df.at[df.index[i], 'Signal'] = '🟢 BUY'
        elif score <= -2:
            df.at[df.index[i], 'Signal'] = '🔴 SELL'
        else:
            df.at[df.index[i], 'Signal'] = '⚪ HOLD'

    return df
