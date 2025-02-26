# backtest.py

import pandas as pd
import numpy as np
from config import TRADING_FEE_PCT
import numba

@numba.jit(nopython=True)
def _apply_min_holding_period(position_array, change_array, min_holding_period):
    """
    Numba-optimized function to apply minimum holding period.
    This is extracted from backtest_strategy to allow JIT compilation.
    """
    last_trade_i = -1  # Initialize to -1 to indicate no previous trade
    for i in range(len(position_array)):
        if change_array[i] != 0:
            if last_trade_i >= 0:
                # Bars since last trade
                if (i - last_trade_i) < min_holding_period:
                    # Cancel this trade
                    change_array[i] = 0
                    position_array[i] = position_array[i-1]
                else:
                    last_trade_i = i
            else:
                last_trade_i = i
    return position_array, change_array

@numba.jit(nopython=True)
def _calculate_performance(prices, positions, position_changes, trading_fee_pct, initial_capital):
    """
    Numba-optimized function to calculate cumulative performance.
    
    Parameters:
    - prices: array of price values
    - positions: array of positions (-1, 0, 1)
    - position_changes: array indicating position changes (non-zero when position changes)
    - trading_fee_pct: fee percentage for each trade
    - initial_capital: starting capital
    
    Returns:
    - total_return: final return percentage
    - final_portfolio_value: final portfolio value
    - num_trades: number of trades executed
    """
    # Calculate returns from prices
    returns = np.zeros(len(prices))
    for i in range(1, len(prices)):
        returns[i] = prices[i] / prices[i-1] - 1.0
    
    # Shift positions by 1 for strategy returns
    shifted_positions = np.zeros(len(positions))
    for i in range(1, len(positions)):
        shifted_positions[i] = positions[i-1]
    
    # Calculate strategy returns
    strategy_returns = shifted_positions * returns
    
    # Apply trading fees
    growth_factors = 1.0 + strategy_returns
    fee_factors = np.ones(len(position_changes))
    for i in range(len(position_changes)):
        if position_changes[i] != 0:
            fee_factors[i] = 1.0 - trading_fee_pct
    
    # Calculate cumulative performance
    portfolio_value = initial_capital
    for i in range(len(growth_factors)):
        portfolio_value *= growth_factors[i] * fee_factors[i]
    
    # Calculate number of trades
    num_trades = 0
    for change in position_changes:
        if change != 0:
            num_trades += 1
    
    total_return = (portfolio_value / initial_capital) - 1.0
    
    return total_return, portfolio_value, num_trades

def backtest_strategy_optimized(
    df: pd.DataFrame,
    initial_capital: float = 10000,
    min_holding_period: int = 0,
    precomputed_returns: pd.Series = None
):
    """
    Optimized vectorized backtest that uses Numba JIT compilation
    for the performance-critical parts.
    """
    if "signal" not in df.columns:
        raise ValueError("DataFrame must have a 'signal' column (+1, -1, or 0).")

    # Extract arrays from DataFrame for Numba compatibility
    if precomputed_returns is not None:
        returns = precomputed_returns.values
    else:
        returns = df["close_price"].pct_change().fillna(0).values
    
    # Convert to native NumPy arrays for Numba
    prices = df["close_price"].values.astype(np.float64)
    raw_signal = df["signal"].values.astype(np.float64)
    
    # Forward-fill to ensure we hold +1 or -1
    position = np.zeros_like(raw_signal)
    position[0] = raw_signal[0]
    for i in range(1, len(position)):
        position[i] = raw_signal[i] if raw_signal[i] != 0 else position[i-1]
    
    # Detect position changes
    pos_change = np.zeros_like(position)
    for i in range(1, len(position)):
        pos_change[i] = position[i] - position[i-1]
    
    # Enforce min holding period
    if min_holding_period > 0:
        position, pos_change = _apply_min_holding_period(position, pos_change, min_holding_period)
    
    # Calculate performance using Numba-optimized function
    total_return, final_portfolio_value, num_trades = _calculate_performance(
        prices, position, pos_change, TRADING_FEE_PCT, initial_capital
    )
    
    return total_return, final_portfolio_value, num_trades

def backtest_strategy(
    df: pd.DataFrame,
    initial_capital: float = 10000,
    min_holding_period: int = 0,
    precomputed_returns: pd.Series = None
):
    """
    Vectorized backtest that:
      - Applies a minimum holding period to reduce overtrading
      - Computes total_return, final_portfolio_val, num_trades
      - Deducts TRADING_FEE_PCT each time there's a position change
        (based on the current portfolio value).
      - Accepts optional 'precomputed_returns' so we don't repeatedly do pct_change.
    """
    # Use the optimized version if Numba is available
    try:
        return backtest_strategy_optimized(df, initial_capital, min_holding_period, precomputed_returns)
    except Exception as e:
        # Fall back to the original implementation if there's an error with Numba
        print(f"Warning: Falling back to standard backtest due to: {e}")
    
    if "signal" not in df.columns:
        raise ValueError("DataFrame must have a 'signal' column (+1, -1, or 0).")

    # 1) Either use precomputed returns or compute once
    if precomputed_returns is not None:
        returns = precomputed_returns
    else:
        returns = df["close_price"].pct_change().fillna(0)

    # 2) Position from signal
    # forward-fill to ensure we hold +1 or -1
    raw_signal = df["signal"]
    position = raw_signal.ffill().fillna(0)

    # 3) Detect position changes
    pos_change = position.diff().fillna(0)

    # 4) Enforce min holding period
    if min_holding_period > 0:
        pos_array = position.values.copy()
        change_array = pos_change.values.copy()
        last_trade_i = None
        for i in range(len(pos_array)):
            if change_array[i] != 0:
                if last_trade_i is not None:
                    # Bars since last trade
                    if (i - last_trade_i) < min_holding_period:
                        # Cancel this trade
                        change_array[i] = 0
                        pos_array[i] = pos_array[i-1]
                    else:
                        last_trade_i = i
                else:
                    last_trade_i = i
        position = pd.Series(pos_array, index=position.index)
        pos_change = pd.Series(change_array, index=pos_change.index)

    # 5) Number of trades
    num_trades = int((pos_change != 0).sum())

    # 6) Strategy returns (shift position by 1)
    shifted_pos = position.shift(1, fill_value=0)
    strategy_returns = shifted_pos * returns

    # 7) Apply fee whenever position changes
    growth_factor = 1.0 + strategy_returns
    fee_factor = np.where(pos_change != 0, 1.0 - TRADING_FEE_PCT, 1.0)
    combined_factor = growth_factor * fee_factor

    # 8) Cumulative performance
    cumulative_factor = pd.Series(combined_factor).cumprod()
    final_portfolio_value = (
        initial_capital * cumulative_factor.iloc[-1]
        if len(cumulative_factor) > 0
        else initial_capital
    )
    total_return = (final_portfolio_value / initial_capital) - 1.0

    return total_return, final_portfolio_value, num_trades

def buy_and_hold(df: pd.DataFrame, initial_capital: float = 10000):
    """Simple buy-and-hold for comparison."""
    if len(df) == 0:
        return 0.0, initial_capital

    buy_price = df["close_price"].iloc[0]
    sell_price = df["close_price"].iloc[-1]
    total_return = (sell_price / buy_price) - 1.0
    final_portfolio_value = initial_capital * (1 + total_return)

    return total_return, final_portfolio_value