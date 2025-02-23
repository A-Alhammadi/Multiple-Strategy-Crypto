import pandas as pd

# 🔹 Global Trading Fee (change this value to update fees everywhere)
TRADING_FEE_PCT = 0.001  # Example: 0.1% (0.001 = 0.1%)

def backtest_strategy(df: pd.DataFrame, initial_capital: float = 10000, return_trades: bool = False):
    """
    Runs a backtest on the given DataFrame and computes:
      - total_return
      - final_portfolio_value
      - (optional) number_of_trades (if return_trades=True)
    
    Uses global TRADING_FEE_PCT for fees.
    """
    df = df.copy()

    # Calculate returns
    df["returns"] = df["close_price"].pct_change()

    # 🔹 Ensure that strategy_returns is calculated
    if "signal" not in df:
        raise ValueError("Missing 'signal' column in DataFrame. Ensure strategies generate signals.")

    df["strategy_returns"] = df["signal"].shift(1) * df["returns"]

    # Track position changes (trades)
    df["position"] = df["signal"].ffill().fillna(0)
    df["pos_change"] = df["position"].diff().fillna(0)
    
    # Count number of trades
    number_of_trades = int((df["pos_change"] != 0).sum())

    # Initialize portfolio value
    portfolio_value = initial_capital
    portfolio_history = [portfolio_value]

    # 🔹 Loop through each row to apply fees and returns correctly
    for i in range(1, len(df)):
        # Apply returns (compounding)
        portfolio_value *= (1 + df["strategy_returns"].iloc[i])

        # Apply trading fees when a position change occurs
        if df["pos_change"].iloc[i] != 0:  # If a trade occurs
            fee = portfolio_value * TRADING_FEE_PCT
            portfolio_value -= fee  # Deduct fee

        # Store portfolio history
        portfolio_history.append(portfolio_value)

    # Assign the new portfolio value column
    df["portfolio_value"] = portfolio_history

    # Final values after applying fees
    final_portfolio_value = df["portfolio_value"].iloc[-1] if len(df) > 0 else initial_capital
    total_return = (final_portfolio_value / initial_capital) - 1.0

    # 🔹 Debugging: Check if fees are deducted
    total_fees = TRADING_FEE_PCT * number_of_trades * initial_capital  # Approximate total fees
    print(f"DEBUG: Total Trades = {number_of_trades}, Estimated Total Fees Deducted = ${total_fees:.2f}")

    if return_trades:
        return total_return, final_portfolio_value, number_of_trades
    else:
        return total_return, final_portfolio_value

def buy_and_hold(df: pd.DataFrame, initial_capital: float = 10000):
    """Returns just two values: total_return, final_portfolio_value"""
    if len(df) == 0:
        return 0.0, initial_capital

    buy_price = df["close_price"].iloc[0]
    sell_price = df["close_price"].iloc[-1]
    total_return = (sell_price / buy_price) - 1.0
    final_portfolio_value = initial_capital * (1 + total_return)

    return total_return, final_portfolio_value
