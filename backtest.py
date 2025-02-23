# backtest.py

import pandas as pd

def backtest_strategy(df: pd.DataFrame, initial_capital: float = 10000):
    """
    Runs a backtest on the given DataFrame and computes:
      - total_return
      - final_portfolio_value
      - number_of_trades
    """
    df = df.copy()

    # Calculate returns
    df["returns"] = df["close_price"].pct_change()

    # Strategy returns
    df["strategy_returns"] = df["signal"].shift(1) * df["returns"]

    # Cumulative returns
    cumulative_return = (1 + df["strategy_returns"].fillna(0)).cumprod() - 1

    # Final total return
    total_return = cumulative_return.iloc[-1] if len(cumulative_return) > 0 else 0.0

    # Compute final portfolio value
    final_portfolio_value = initial_capital * (1 + total_return)

    # Count trades as how many times position changes (from +1 to -1, etc.)
    df["position"] = df["signal"].ffill().fillna(0)
    df["pos_change"] = df["position"].diff().fillna(0)
    number_of_trades = int((df["pos_change"] != 0).sum())

    return total_return, final_portfolio_value, number_of_trades


def buy_and_hold(df: pd.DataFrame, initial_capital: float = 10000):
    """Still returns just two values."""
    if len(df) == 0:
        return 0.0, initial_capital

    buy_price = df["close_price"].iloc[0]
    sell_price = df["close_price"].iloc[-1]
    total_return = (sell_price / buy_price) - 1.0
    final_portfolio_value = initial_capital * (1 + total_return)

    return total_return, final_portfolio_value
