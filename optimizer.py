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
    Given something like:
      {
        'short_window': [5, 10],
        'long_window': [50, 100]
      }
    Returns a list of dicts:
      [
        {'short_window': 5, 'long_window': 50},
        {'short_window': 5, 'long_window': 100},
        {'short_window': 10, 'long_window': 50},
        {'short_window': 10, 'long_window': 100}
      ]
    """
    keys = list(strategy_param_grid.keys())
    values_product = list(itertools.product(*strategy_param_grid.values()))
    param_dicts = []
    for vals in values_product:
        param_dicts.append(dict(zip(keys, vals)))
    return param_dicts

def optimize_strategy(df: pd.DataFrame, strategy_name: str, strategy_param_grid: dict):
    """
    Runs a grid search for the given strategy and returns the best parameters
    along with the best performance (total return).
    """
    # If the strategy has no parameters (i.e., empty grid), just run it once
    if not strategy_param_grid:
        df["signal"] = STRATEGY_FUNCTIONS[strategy_name](df)
        perf = backtest_strategy(df)
        return {}, perf

    best_params = None
    best_performance = float("-inf")

    param_dicts = generate_param_dicts(strategy_param_grid)

    for param_dict in param_dicts:
        # Apply strategy with these parameters
        df_temp = df.copy()
        df_temp["signal"] = STRATEGY_FUNCTIONS[strategy_name](df_temp, **param_dict)

        # Evaluate performance
        performance = backtest_strategy(df_temp)

        # Track the best
        if performance > best_performance:
            best_performance = performance
            best_params = param_dict

    return best_params, best_performance
