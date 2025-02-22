# backtest.py

import pandas as pd

def backtest_strategy(df: pd.DataFrame):
    """
    Given a DataFrame containing 'signal' (1 or -1) and 'close_price',
    computes the total return of the strategy.
    """
    df = df.copy()

    # Calculate daily (hourly) returns
    df["returns"] = df["close_price"].pct_change()

    # Strategy returns: signal * returns, shift(1) to enter at previous close
    df["strategy_returns"] = df["signal"].shift(1) * df["returns"]

    # Cumulative return
    cumulative_return = (1 + df["strategy_returns"].fillna(0)).cumprod() - 1

    # Final total return
    total_return = cumulative_return.iloc[-1] if len(cumulative_return) > 0 else 0.0
    return total_return

def buy_and_hold(df: pd.DataFrame):
    """
    Computes the buy-and-hold return from first close_price to last close_price.
    """
    if len(df) == 0:
        return 0.0
    buy_price = df["close_price"].iloc[0]
    sell_price = df["close_price"].iloc[-1]
    return (sell_price / buy_price) - 1.0
