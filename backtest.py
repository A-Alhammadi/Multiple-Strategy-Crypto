# backtest.py

import pandas as pd

def backtest_strategy(df: pd.DataFrame, initial_capital: float = 10000):
    """
    Runs a backtest on the given DataFrame and computes total return & final portfolio value.
    Assumes we start with `initial_capital` dollars.
    """
    df = df.copy()

    # Calculate hourly returns
    df["returns"] = df["close_price"].pct_change()

    # Strategy returns (apply shift(1) to avoid lookahead bias)
    df["strategy_returns"] = df["signal"].shift(1) * df["returns"]

    # Cumulative returns
    cumulative_return = (1 + df["strategy_returns"].fillna(0)).cumprod() - 1

    # Final total return
    total_return = cumulative_return.iloc[-1] if len(cumulative_return) > 0 else 0.0

    # Compute final portfolio value
    final_portfolio_value = initial_capital * (1 + total_return)

    return total_return, final_portfolio_value

def buy_and_hold(df: pd.DataFrame, initial_capital: float = 10000):
    """
    Computes buy-and-hold return and final portfolio value starting from `initial_capital`.
    """
    if len(df) == 0:
        return 0.0, initial_capital

    buy_price = df["close_price"].iloc[0]
    sell_price = df["close_price"].iloc[-1]

    total_return = (sell_price / buy_price) - 1.0
    final_portfolio_value = initial_capital * (1 + total_return)

    return total_return, final_portfolio_value
