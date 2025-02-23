# optimizer.py

import itertools
import pandas as pd
import numpy as np

from backtest import backtest_strategy
from strategy import (
    moving_average_crossover,
    rsi,
    bollinger_bands,
    macd,
    high_low_breakout,
    volume_price_action,
    vwap_zone,
    zscore_mean_reversion
)

# Mapping strategy names to functions
STRATEGY_FUNCTIONS = {
    "moving_average_crossover": moving_average_crossover,
    "rsi": rsi,
    "bollinger_bands": bollinger_bands,
    "macd": macd,
    "high_low_breakout": high_low_breakout,
    "volume_price_action": volume_price_action,
    "vwap_zone": vwap_zone,
    "zscore_mean_reversion": zscore_mean_reversion
}

def generate_param_dicts(strategy_param_grid):
    """
    Given a parameter grid, generate all possible parameter combinations.
    Example:
      strategy_param_grid = {'short_window': [5, 10], 'long_window': [50, 100]}
    Produces:
      [{'short_window': 5, 'long_window': 50},
       {'short_window': 5, 'long_window': 100},
       {'short_window': 10, 'long_window': 50},
       {'short_window': 10, 'long_window': 100}]
    """
    if not strategy_param_grid:
        return [{}]  # no parameters to tune
    keys = list(strategy_param_grid.keys())
    values_product = list(itertools.product(*strategy_param_grid.values()))
    return [dict(zip(keys, vals)) for vals in values_product]

def optimize_strategy(df: pd.DataFrame, strategy_name: str, strategy_param_grid: dict, initial_capital: float = 10000):
    """
    Runs a grid search over the given strategy’s parameter space and returns:
    - The best parameters found
    - The best performance (total return)
    - The final portfolio value for the best-performing parameters
    """

    # If the strategy has no tunable parameters, just run it once
    param_dicts = generate_param_dicts(strategy_param_grid)

    best_params = None
    best_performance = float("-inf")
    best_final_portfolio = 0

    for param_dict in param_dicts:
        # Apply strategy with the current set of parameters
        df_temp = df.copy()
        df_temp["signal"] = STRATEGY_FUNCTIONS[strategy_name](df_temp, **param_dict)

        # Evaluate strategy performance and portfolio value
        result = backtest_strategy(df_temp, initial_capital)

        if isinstance(result, tuple) and len(result) == 2:
            performance, final_portfolio = result
        elif isinstance(result, tuple) and len(result) == 3:
            performance, final_portfolio, _ = result  # Ignore num_trades if not needed
        else:
            raise ValueError(f"Unexpected return from backtest_strategy: {result}")

        # Track the best-performing parameters
        if performance > best_performance:
            best_performance = performance
            best_final_portfolio = final_portfolio
            best_params = param_dict

    return best_params, best_performance, best_final_portfolio
