import numpy as np
import pandas as pd

def calculate_rsi(series, period=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, short=12, long=26, signal=9):
    """Calculate MACD line and MACD signal line."""
    ema_short = series.ewm(span=short, adjust=False).mean()
    ema_long = series.ewm(span=long, adjust=False).mean()
    macd_line = ema_short - ema_long
    return macd_line  # Return only the MACD line for simplicity

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range (ATR)."""
    high_low = high - low
    high_close = np.abs(high - close.shift(1))
    low_close = np.abs(low - close.shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_bbands(series, length=20, std_dev=2):
    """Calculate Bollinger Bands."""
    sma = series.rolling(window=length).mean()
    std = series.rolling(window=length).std()
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    return upper_band, sma, lower_band

def calculate_stochastic_oscillator(high, low, close, period=14):
    """Calculate Stochastic Oscillator (K% and D%)."""
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-8)
    d_percent = k_percent.rolling(window=3).mean()  # D% is the 3-day SMA of K%
    return k_percent, d_percent