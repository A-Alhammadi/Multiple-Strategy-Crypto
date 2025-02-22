# strategy.py

import numpy as np
import pandas as pd

def moving_average_crossover(df: pd.DataFrame, short_window: int, long_window: int):
    """
    Strategy: Buy if short MA crosses above long MA, sell if short MA crosses below long MA.
    """
    df = df.copy()
    df["ma_short"] = df["close_price"].rolling(window=short_window).mean()
    df["ma_long"] = df["close_price"].rolling(window=long_window).mean()

    # Signal: 1 = long, -1 = short
    df["signal"] = np.where(df["ma_short"] > df["ma_long"], 1, -1)
    return df["signal"]

def rsi(df: pd.DataFrame, period: int, buy_threshold: float, sell_threshold: float):
    """
    Strategy: Buy when RSI < buy_threshold, Sell when RSI > sell_threshold, else hold previous.
    """
    df = df.copy()
    delta = df["close_price"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Generate signals
    df["signal"] = 0
    df.loc[df["rsi"] < buy_threshold, "signal"] = 1
    df.loc[df["rsi"] > sell_threshold, "signal"] = -1

    # Forward fill to hold positions
    df["signal"] = df["signal"].replace(to_replace=0, method="ffill")
    return df["signal"]

def bollinger_bands(df: pd.DataFrame, period: int):
    """
    Strategy: Buy if price crosses below lower band, Sell if price crosses above upper band.
    """
    df = df.copy()
    df["middle_band"] = df["close_price"].rolling(window=period).mean()
    df["std"] = df["close_price"].rolling(window=period).std()

    df["upper_band"] = df["middle_band"] + 2 * df["std"]
    df["lower_band"] = df["middle_band"] - 2 * df["std"]

    df["signal"] = 0
    df.loc[df["close_price"] < df["lower_band"], "signal"] = 1
    df.loc[df["close_price"] > df["upper_band"], "signal"] = -1

    # Forward fill
    df["signal"] = df["signal"].replace(to_replace=0, method="ffill")
    return df["signal"]

def macd(df: pd.DataFrame, fast_period: int, slow_period: int, signal_period: int):
    """
    Strategy: Buy if MACD > MACD signal line, Sell if MACD < MACD signal line.
    """
    df = df.copy()
    df["ema_fast"] = df["close_price"].ewm(span=fast_period, adjust=False).mean()
    df["ema_slow"] = df["close_price"].ewm(span=slow_period, adjust=False).mean()
    df["macd_line"] = df["ema_fast"] - df["ema_slow"]
    df["macd_signal"] = df["macd_line"].ewm(span=signal_period, adjust=False).mean()

    df["signal"] = np.where(df["macd_line"] > df["macd_signal"], 1, -1)
    return df["signal"]

def high_low_breakout(df: pd.DataFrame, lookback: int):
    """
    Strategy: Buy if current price breaks above the max of last 'lookback' hours,
              Sell if current price breaks below the min of last 'lookback' hours.
    """
    df = df.copy()
    df["rolling_high"] = df["close_price"].rolling(window=lookback).max()
    df["rolling_low"] = df["close_price"].rolling(window=lookback).min()

    df["signal"] = 0
    df.loc[df["close_price"] > df["rolling_high"].shift(1), "signal"] = 1
    df.loc[df["close_price"] < df["rolling_low"].shift(1), "signal"] = -1

    # Forward fill
    df["signal"] = df["signal"].replace(to_replace=0, method="ffill")
    return df["signal"]

def volume_price_action(df: pd.DataFrame, volume_multiplier: float):
    """
    Strategy: If large volume spike + bullish engulfing => Buy,
              If large volume spike + bearish engulfing => Sell.
    A very simplistic version just to demonstrate the idea.
    """
    df = df.copy()

    # Calculate a rolling average volume for reference
    df["volume_ma"] = df["volume_crypto"].rolling(window=20).mean()

    # Bullish engulfing: current candle's body fully covers previous candle's body
    # Bearish engulfing: current candle's body is fully below previous candle
    # We'll just approximate it in a simplistic way for demonstration
    df["prev_close"] = df["close_price"].shift(1)
    df["prev_open"] = df["open_price"].shift(1)

    # Basic bull/bear detection
    df["bullish_engulf"] = (df["close_price"] > df["prev_close"]) & (df["open_price"] < df["prev_open"])
    df["bearish_engulf"] = (df["close_price"] < df["prev_close"]) & (df["open_price"] > df["prev_open"])

    # Volume spike
    df["volume_spike"] = df["volume_crypto"] > df["volume_ma"] * volume_multiplier

    df["signal"] = 0
    df.loc[df["bullish_engulf"] & df["volume_spike"], "signal"] = 1
    df.loc[df["bearish_engulf"] & df["volume_spike"], "signal"] = -1

    # Forward fill
    df["signal"] = df["signal"].replace(to_replace=0, method="ffill")
    return df["signal"]

def vwap_zone(df: pd.DataFrame, rsi_period: int, rsi_lower: float, rsi_upper: float):
    """
    Strategy: 
      - Calculate VWAP. 
      - Buy if price < VWAP and RSI > rsi_lower (sign of accumulation),
      - Sell if price > VWAP and RSI < rsi_upper (distribution).
    """
    df = df.copy()

    # VWAP calculation (basic version)
    # volume_usd ~ (price * volume_crypto), so approximate typical price * volume_crypto
    # We'll use close_price * volume_crypto for simplicity.
    df["cum_price_vol"] = (df["close_price"] * df["volume_crypto"]).cumsum()
    df["cum_vol"] = df["volume_crypto"].cumsum()
    df["vwap"] = df["cum_price_vol"] / (df["cum_vol"] + 1e-10)

    # RSI
    delta = df["close_price"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    df["signal"] = 0
    df.loc[(df["close_price"] < df["vwap"]) & (df["rsi"] > rsi_lower), "signal"] = 1
    df.loc[(df["close_price"] > df["vwap"]) & (df["rsi"] < rsi_upper), "signal"] = -1

    # Forward fill
    df["signal"] = df["signal"].replace(to_replace=0, method="ffill")
    return df["signal"]

def zscore_mean_reversion(df: pd.DataFrame, zscore_window: int, zscore_threshold: float):
    """
    Strategy:
      - Compute the z-score of price relative to its rolling mean/std
      - If z-score < -zscore_threshold => buy (oversold)
      - If z-score >  zscore_threshold => sell (overbought)
    """
    df = df.copy()
    rolling_mean = df["close_price"].rolling(window=zscore_window).mean()
    rolling_std = df["close_price"].rolling(window=zscore_window).std()

    df["zscore"] = (df["close_price"] - rolling_mean) / (rolling_std + 1e-10)

    df["signal"] = 0
    df.loc[df["zscore"] < -zscore_threshold, "signal"] = 1
    df.loc[df["zscore"] > zscore_threshold,  "signal"] = -1

    # Forward fill
    df["signal"] = df["signal"].replace(to_replace=0, method="ffill")
    return df["signal"]
