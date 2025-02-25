# strategy.py

import numpy as np
import pandas as pd

def moving_average_crossover(df: pd.DataFrame, short_window: int, long_window: int):
    """
    Strategy: Buy if short MA crosses above long MA, sell if short MA crosses below long MA.
    Returns just the 'signal' Series.
    """
    ma_short = df["close_price"].rolling(window=short_window).mean()
    ma_long = df["close_price"].rolling(window=long_window).mean()

    # 1 = long, -1 = short
    signal = np.where(ma_short > ma_long, 1, -1)

    # Forward-fill any zeros
    signal_series = pd.Series(signal, index=df.index)
    # Updated method to avoid deprecated replace() with method parameter
    signal_series = signal_series.mask(signal_series == 0).ffill().fillna(0)
    return signal_series

def rsi(df: pd.DataFrame, period: int, buy_threshold: float, sell_threshold: float):
    """
    Strategy: Buy when RSI < buy_threshold, Sell when RSI > sell_threshold, else hold.
    """
    delta = df["close_price"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_vals = 100 - (100 / (1 + rs))

    signal_series = pd.Series(0, index=df.index)
    signal_series[rsi_vals < buy_threshold] = 1
    signal_series[rsi_vals > sell_threshold] = -1
    # Updated method to avoid deprecated replace() with method parameter
    signal_series = signal_series.mask(signal_series == 0).ffill().fillna(0)
    return signal_series

def bollinger_bands(df: pd.DataFrame, period: int, std_dev: float = 2.0):
    """
    Strategy: Buy if price crosses below lower band, Sell if price crosses above upper band.
    """
    middle_band = df["close_price"].rolling(window=period).mean()
    rolling_std = df["close_price"].rolling(window=period).std()

    upper_band = middle_band + std_dev * rolling_std
    lower_band = middle_band - std_dev * rolling_std

    signal_series = pd.Series(0, index=df.index)
    signal_series[df["close_price"] < lower_band] = 1
    signal_series[df["close_price"] > upper_band] = -1
    # Updated method to avoid deprecated replace() with method parameter
    signal_series = signal_series.mask(signal_series == 0).ffill().fillna(0)
    return signal_series

def macd(df: pd.DataFrame, fast_period: int, slow_period: int, signal_period: int):
    """
    Strategy: Buy if MACD > MACD signal line, Sell if MACD < MACD signal line.
    """
    ema_fast = df["close_price"].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df["close_price"].ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal_period, adjust=False).mean()

    signal_series = pd.Series(0, index=df.index)
    signal_series[macd_line > macd_signal] = 1
    signal_series[macd_line < macd_signal] = -1
    # Updated method to avoid deprecated replace() with method parameter
    signal_series = signal_series.mask(signal_series == 0).ffill().fillna(0)
    return signal_series

def high_low_breakout(df: pd.DataFrame, lookback: int):
    """
    Strategy: Buy if current price breaks above max of last 'lookback' bars,
              Sell if current price breaks below min of last 'lookback' bars.
    """
    rolling_high = df["close_price"].rolling(window=lookback).max()
    rolling_low = df["close_price"].rolling(window=lookback).min()

    signal_series = pd.Series(0, index=df.index)
    signal_series[df["close_price"] > rolling_high.shift(1)] = 1
    signal_series[df["close_price"] < rolling_low.shift(1)] = -1
    # Updated method to avoid deprecated replace() with method parameter
    signal_series = signal_series.mask(signal_series == 0).ffill().fillna(0)
    return signal_series

def volume_price_action(df: pd.DataFrame, volume_multiplier: float):
    """
    Strategy: If large volume spike + bullish engulfing => Buy,
              If large volume spike + bearish engulfing => Sell.
    """
    volume_ma = df["volume_crypto"].rolling(window=20).mean()

    prev_close = df["close_price"].shift(1)
    prev_open = df["open_price"].shift(1)

    bullish_engulf = (df["close_price"] > prev_close) & (df["open_price"] < prev_open)
    bearish_engulf = (df["close_price"] < prev_close) & (df["open_price"] > prev_open)

    volume_spike = df["volume_crypto"] > (volume_ma * volume_multiplier)

    signal_series = pd.Series(0, index=df.index)
    signal_series[bullish_engulf & volume_spike] = 1
    signal_series[bearish_engulf & volume_spike] = -1
    # Updated method to avoid deprecated replace() with method parameter
    signal_series = signal_series.mask(signal_series == 0).ffill().fillna(0)
    return signal_series

def vwap_zone(df: pd.DataFrame, rsi_period: int, rsi_lower: float, rsi_upper: float):
    """
    Strategy: 
      - Calculate VWAP 
      - Buy if price < VWAP and RSI > rsi_lower
      - Sell if price > VWAP and RSI < rsi_upper
    """
    cum_price_vol = (df["close_price"] * df["volume_crypto"]).cumsum()
    cum_vol = df["volume_crypto"].cumsum()
    vwap_vals = cum_price_vol / (cum_vol + 1e-10)

    # RSI
    delta = df["close_price"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_vals = 100 - (100 / (1 + rs))

    signal_series = pd.Series(0, index=df.index)
    signal_series[(df["close_price"] < vwap_vals) & (rsi_vals > rsi_lower)] = 1
    signal_series[(df["close_price"] > vwap_vals) & (rsi_vals < rsi_upper)] = -1
    # Updated method to avoid deprecated replace() with method parameter
    signal_series = signal_series.mask(signal_series == 0).ffill().fillna(0)
    return signal_series

def zscore_mean_reversion(df: pd.DataFrame, zscore_window: int, zscore_threshold: float):
    """
    Strategy:
      - If z-score < -zscore_threshold => buy
      - If z-score >  zscore_threshold => sell
    """
    rolling_mean = df["close_price"].rolling(window=zscore_window).mean()
    rolling_std = df["close_price"].rolling(window=zscore_window).std()

    zscore_vals = (df["close_price"] - rolling_mean) / (rolling_std + 1e-10)

    signal_series = pd.Series(0, index=df.index)
    signal_series[zscore_vals < -zscore_threshold] = 1
    signal_series[zscore_vals >  zscore_threshold] = -1
    # Updated method to avoid deprecated replace() with method parameter
    signal_series = signal_series.mask(signal_series == 0).ffill().fillna(0)
    return signal_series