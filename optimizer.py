# optimizer.py

import itertools
import pandas as pd
import numpy as np
import concurrent.futures
from functools import partial

from config import (
    PENALTY_FACTOR_GRID,
    MIN_HOLDING_PERIOD_GRID,
    SHARPE_RATIO_WEIGHT_GRID
)
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

# Signal cache for avoiding redundant calculations
class SignalCache:
    """Cache for strategy signals to avoid repeated calculations"""
    
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
    
    def _get_hash(self, strategy_name, params, df_hash):
        """Create a hash from strategy name, parameters, and dataframe hash"""
        param_str = str(sorted(params.items()))
        combined = f"{strategy_name}_{param_str}_{df_hash}"
        return hash(combined)
    
    def _get_df_hash(self, df):
        """Create a hash of the dataframe's close price column"""
        # We only hash the close prices since that's what most strategies use
        close_prices = df["close_price"].values
        return hash(tuple(close_prices))
    
    def get(self, strategy_name, df, **params):
        """Get a signal from cache or compute it"""
        df_hash = self._get_df_hash(df)
        cache_key = self._get_hash(strategy_name, params, df_hash)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Not in cache, compute the signal
        signal = STRATEGY_FUNCTIONS[strategy_name](df, **params)
        
        # Store in cache
        if len(self.cache) >= self.max_size:
            # Clear half the cache if it gets too big
            keys_to_remove = list(self.cache.keys())[:self.max_size//2]
            for key in keys_to_remove:
                self.cache.pop(key)
        
        self.cache[cache_key] = signal
        return signal

# Create a global instance
signal_cache = SignalCache()

def generate_param_dicts(param_grid):
    """
    Given a dictionary of lists, produce every combination.
    E.g. {'a':[1,2],'b':[3]} => [{'a':1,'b':3},{'a':2,'b':3}]
    """
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    values_product = list(itertools.product(*param_grid.values()))
    return [dict(zip(keys, vals)) for vals in values_product]

def _evaluate_strategy_params(df, strategy_name, strat_params, meta_params, initial_capital, precomputed_returns):
    """Helper function to evaluate a single set of strategy parameters"""
    # Build signal using cache
    signal = signal_cache.get(strategy_name, df, **strat_params)
    
    # Create minimal DataFrame for backtest
    backtest_df = pd.DataFrame({"close_price": df["close_price"], "signal": signal})
    
    # Backtest
    performance, final_portfolio, num_trades = backtest_strategy(
        backtest_df,
        initial_capital=initial_capital,
        min_holding_period=meta_params["min_holding_period"],
        precomputed_returns=precomputed_returns
    )
    
    # Compute Sharpe ratio
    strat_returns = signal.shift(1, fill_value=0) * precomputed_returns
    std_dev = strat_returns.std()
    sharpe_ratio = (strat_returns.mean() / std_dev) if std_dev > 0 else 0.0
    
    # Compute final score
    w = meta_params["sharpe_ratio_weight"]
    penalty_factor = meta_params["penalty_factor"]
    score = (1 - w) * performance + (w * sharpe_ratio) - (penalty_factor * num_trades)
    
    return score, performance, final_portfolio, num_trades, strat_params, meta_params

def optimize_strategy(df, strategy_name, strategy_param_grid, initial_capital=10000, precomputed_returns=None):
    """
    Optimized version of strategy optimization using parallelization
    and caching for better performance.
    """
    # Ensure we have precomputed returns
    if precomputed_returns is None:
        precomputed_returns = df["close_price"].pct_change().fillna(0)
    
    # Generate parameter combinations
    strategy_param_dicts = generate_param_dicts(strategy_param_grid)
    meta_param_dicts = generate_param_dicts({
        "penalty_factor": PENALTY_FACTOR_GRID,
        "min_holding_period": MIN_HOLDING_PERIOD_GRID,
        "sharpe_ratio_weight": SHARPE_RATIO_WEIGHT_GRID
    })
    
    best_overall_params = None
    best_score = float("-inf")
    best_final_portfolio = 0.0
    best_num_trades = 0
    
    # Create parameter combinations
    param_combinations = []
    for strat_params in strategy_param_dicts:
        for meta_params in meta_param_dicts:
            param_combinations.append((strat_params, meta_params))
    
    # Evaluate parameters in batches to control memory usage
    batch_size = 100  # Adjust based on your system's memory
    for i in range(0, len(param_combinations), batch_size):
        batch = param_combinations[i:i + batch_size]
        
        # Process batch
        for strat_params, meta_params in batch:
            score, performance, final_portfolio, num_trades, _, _ = _evaluate_strategy_params(
                df, strategy_name, strat_params, meta_params, initial_capital, precomputed_returns
            )
            
            if score > best_score:
                best_score = score
                best_overall_params = {**strat_params, **meta_params}
                best_final_portfolio = final_portfolio
                best_num_trades = num_trades
    
    return best_overall_params, best_score, best_final_portfolio, best_num_trades

def optimize_strategy_parallel(df, strategy_name, strategy_param_grid, initial_capital=10000, 
                              precomputed_returns=None, max_workers=None):
    """
    Parallel version of strategy optimization for better performance on multi-core systems.
    """
    # Ensure we have precomputed returns
    if precomputed_returns is None:
        precomputed_returns = df["close_price"].pct_change().fillna(0)
    
    # Generate parameter combinations
    strategy_param_dicts = generate_param_dicts(strategy_param_grid)
    meta_param_dicts = generate_param_dicts({
        "penalty_factor": PENALTY_FACTOR_GRID,
        "min_holding_period": MIN_HOLDING_PERIOD_GRID,
        "sharpe_ratio_weight": SHARPE_RATIO_WEIGHT_GRID
    })
    
    # Create parameter combinations
    param_combinations = []
    for strat_params in strategy_param_dicts:
        for meta_params in meta_param_dicts:
            param_combinations.append((strat_params, meta_params))
    
    # Create partial function with fixed parameters
    evaluate_func = partial(
        _evaluate_strategy_params,
        df=df,
        strategy_name=strategy_name,
        initial_capital=initial_capital,
        precomputed_returns=precomputed_returns
    )
    
    # Process in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(evaluate_func, strat_params, meta_params)
            for strat_params, meta_params in param_combinations
        ]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error optimizing strategy {strategy_name}: {e}")
    
    # Find best result
    if results:
        best_result = max(results, key=lambda x: x[0])  # Sort by score
        best_score, _, best_final_portfolio, best_num_trades, best_strat_params, best_meta_params = best_result
        best_overall_params = {**best_strat_params, **best_meta_params}
        return best_overall_params, best_score, best_final_portfolio, best_num_trades
    
    # Fallback to empty results
    return {}, 0.0, initial_capital, 0