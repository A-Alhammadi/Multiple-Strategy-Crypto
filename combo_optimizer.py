# combo_optimizer.py - Fix for default parameters

import itertools
import pandas as pd
import numpy as np
import gc
from typing import List, Dict

from config import (
    PENALTY_FACTOR_GRID,
    MIN_HOLDING_PERIOD_GRID,
    SHARPE_RATIO_WEIGHT_GRID,
    STRATEGY_PARAM_GRID  # Import default parameters grid
)
from backtest import backtest_strategy
from combo_signals import combine_signals
from optimizer import generate_param_dicts, STRATEGY_FUNCTIONS, signal_cache

def optimize_single_strategies(
    df: pd.DataFrame,
    strategy_param_grid: Dict[str, Dict],
    initial_capital: float = 10000,
    precomputed_returns: pd.Series = None
) -> pd.DataFrame:
    """
    Optimized version of single strategy optimization with more efficient operations.
    """
    results = []
    
    # Cache frequently accessed data
    close_prices = df["close_price"]
    
    # Ensure returns are precomputed
    if precomputed_returns is None:
        precomputed_returns = close_prices.pct_change().fillna(0)
    
    # Global meta-param combos - generate once outside the loop
    meta_param_dicts = generate_param_dicts({
        "penalty_factor": PENALTY_FACTOR_GRID,
        "min_holding_period": MIN_HOLDING_PERIOD_GRID,
        "sharpe_ratio_weight": SHARPE_RATIO_WEIGHT_GRID
    })

    for strategy_name in STRATEGY_FUNCTIONS.keys():
        params_dict = strategy_param_grid.get(strategy_name, {})
        strategy_param_combos = generate_param_dicts(params_dict)

        best_score = float("-inf")
        best_params = {}
        best_val = 0
        best_num_trades = 0

        # Nested loops
        for strat_params in strategy_param_combos:
            for meta_params in meta_param_dicts:
                # Build signal - use cached version
                signal = signal_cache.get(strategy_name, df, **strat_params)
                
                # Create a minimal DataFrame with only required columns
                backtest_df = pd.DataFrame({"close_price": close_prices, "signal": signal})
                
                # Backtest
                perf, portfolio_val, num_trades = backtest_strategy(
                    backtest_df,
                    initial_capital=initial_capital,
                    min_holding_period=meta_params["min_holding_period"],
                    precomputed_returns=precomputed_returns
                )

                # Compute Sharpe
                shifted_pos = signal.shift(1, fill_value=0)
                strat_returns = shifted_pos * precomputed_returns
                std_dev = strat_returns.std()
                sharpe = (strat_returns.mean() / std_dev) if std_dev > 0 else 0.0

                # Final score
                w = meta_params["sharpe_ratio_weight"]
                penalty = meta_params["penalty_factor"]
                score = (1 - w)*perf + (w*sharpe) - (penalty*num_trades)

                # Track best
                if score > best_score:
                    best_score = score
                    best_val = portfolio_val
                    best_num_trades = num_trades
                    best_params = {
                        "strategy_params": strat_params,
                        "meta_params": meta_params
                    }

        results.append({
            "StrategyCombo": [strategy_name],
            "BuyOperator": None,
            "SellOperator": None,
            "BestParams": best_params,
            "TrainPerformance": best_score,
            "FinalPortfolioValue": best_val,
            "NumberOfTrades": best_num_trades
        })

    return pd.DataFrame(results)

def optimize_strategy_combo(
    df: pd.DataFrame, 
    strategy_names: List[str],
    param_grids: Dict[str, Dict],
    buy_operator: str,
    sell_operator: str,
    initial_capital: float = 10000,
    precomputed_returns: pd.Series = None
):
    """
    Optimized version of strategy combo evaluation.
    """
    # Extract parameter combinations for each strategy
    strategy_param_combos = []
    for sname in strategy_names:
        param_dicts = generate_param_dicts(param_grids.get(sname, {}))
        if not param_dicts:
            param_dicts = [{}]
        strategy_param_combos.append(param_dicts)

    # Meta parameter combinations
    meta_param_dicts = generate_param_dicts({
        "penalty_factor": PENALTY_FACTOR_GRID,
        "min_holding_period": MIN_HOLDING_PERIOD_GRID,
        "sharpe_ratio_weight": SHARPE_RATIO_WEIGHT_GRID
    })
    
    # Cache close prices and ensure returns are precomputed
    close_prices = df["close_price"]
    if precomputed_returns is None:
        precomputed_returns = close_prices.pct_change().fillna(0)

    best_params_combo = None
    best_score = float("-inf")
    best_portfolio_value = 0
    best_num_trades = 0

    # Precompute signals for each strategy/param combination to avoid duplicated calculations
    precomputed_signals = {}
    for i, sname in enumerate(strategy_names):
        precomputed_signals[sname] = {}
        for param_dict in strategy_param_combos[i]:
            param_key = tuple(sorted(param_dict.items()))
            precomputed_signals[sname][param_key] = signal_cache.get(sname, df, **param_dict)
    
    # Loop through parameter combinations
    combo_count = 0
    total_combos = len(meta_param_dicts) * np.prod([len(combos) for combos in strategy_param_combos])
    
    for meta_params in meta_param_dicts:
        for combo_tuple in itertools.product(*strategy_param_combos):
            combo_count += 1
            if combo_count % 100 == 0:
                print(f"Processing combo {combo_count}/{total_combos} for {strategy_names}")
            
            # Build multi-strategy signals
            signal_dfs = []
            for i, sname in enumerate(strategy_names):
                strat_params = combo_tuple[i]
                param_key = tuple(sorted(strat_params.items()))
                
                # Get precomputed signal
                s_signal = precomputed_signals[sname][param_key]
                signal_dfs.append(pd.DataFrame({"signal": s_signal}, index=df.index))

            # Combine signals
            final_signal = combine_signals(signal_dfs, buy_operator=buy_operator, sell_operator=sell_operator)

            # Backtest
            perf, portfolio_val, num_trades = backtest_strategy(
                pd.DataFrame({"close_price": close_prices, "signal": final_signal}),
                initial_capital=initial_capital,
                min_holding_period=meta_params["min_holding_period"],
                precomputed_returns=precomputed_returns
            )

            # Compute Sharpe
            strat_returns = final_signal.shift(1, fill_value=0) * precomputed_returns
            std_dev = strat_returns.std()
            sharpe = (strat_returns.mean() / std_dev) if std_dev > 0 else 0.0

            # Calculate score
            w = meta_params["sharpe_ratio_weight"]
            penalty = meta_params["penalty_factor"]
            score = (1 - w)*perf + (w*sharpe) - (penalty*num_trades)

            if score > best_score:
                best_score = score
                best_portfolio_value = portfolio_val
                best_num_trades = num_trades
                
                # Build param dict
                param_set_dict = {}
                for i, sname in enumerate(strategy_names):
                    param_set_dict[sname] = combo_tuple[i]
                    
                best_params_combo = {
                    "Strategies": param_set_dict,
                    "Meta": meta_params
                }
    
    # Clean up to free memory
    precomputed_signals.clear()
    gc.collect()

    return best_params_combo, best_score, best_portfolio_value, best_num_trades

def optimize_strategy_combo_improved(
    df: pd.DataFrame, 
    strategy_names: List[str],
    param_grids: Dict[str, Dict],
    buy_operator: str,
    sell_operator: str,
    initial_capital: float = 10000,
    precomputed_returns: pd.Series = None
):
    """
    Further optimized version with early pruning of unpromising combinations
    """
    # Ensure returns are precomputed
    close_prices = df["close_price"]
    if precomputed_returns is None:
        precomputed_returns = close_prices.pct_change().fillna(0)
    
    # Extract parameters for each strategy
    strategy_param_sets = []
    for sname in strategy_names:
        param_dicts = generate_param_dicts(param_grids.get(sname, {}))
        if not param_dicts:
            # Use default params from STRATEGY_PARAM_GRID instead of empty dict
            param_dicts = generate_param_dicts(STRATEGY_PARAM_GRID.get(sname, {}))
            if not param_dicts:  # If still empty, use a minimal default
                if sname == "moving_average_crossover":
                    param_dicts = [{"short_window": 10, "long_window": 50}]
                elif sname == "rsi":
                    param_dicts = [{"period": 14, "buy_threshold": 30, "sell_threshold": 70}]
                elif sname == "bollinger_bands":
                    param_dicts = [{"period": 20, "std_dev": 2.0}]
                elif sname == "macd":
                    param_dicts = [{"fast_period": 12, "slow_period": 26, "signal_period": 9}]
                elif sname == "high_low_breakout":
                    param_dicts = [{"lookback": 20}]
                elif sname == "volume_price_action":
                    param_dicts = [{"volume_multiplier": 2.0}]
                elif sname == "vwap_zone":
                    param_dicts = [{"rsi_period": 14, "rsi_lower": 40, "rsi_upper": 60}]
                elif sname == "zscore_mean_reversion":
                    param_dicts = [{"zscore_window": 20, "zscore_threshold": 2.0}]
                else:
                    param_dicts = [{}]
        strategy_param_sets.append(param_dicts)
    
    # Generate meta parameter combinations
    meta_param_dicts = generate_param_dicts({
        "penalty_factor": PENALTY_FACTOR_GRID,
        "min_holding_period": MIN_HOLDING_PERIOD_GRID,
        "sharpe_ratio_weight": SHARPE_RATIO_WEIGHT_GRID
    })
    
    # Precompute individual strategy signals to reuse
    strategy_signals = {}
    for i, sname in enumerate(strategy_names):
        strategy_signals[sname] = {}
        for params in strategy_param_sets[i]:
            params_tuple = tuple(sorted((k, v) for k, v in params.items()))
            strategy_signals[sname][params_tuple] = signal_cache.get(sname, df, **params)
    
    best_params_combo = None
    best_score = float("-inf")
    best_portfolio_value = 0
    best_num_trades = 0
    
    # First, test each meta param set with default strategy params
    # This allows us to quickly identify promising meta param sets
    promising_meta_params = []
    
    # Use first parameter set for each strategy instead of empty dict
    default_params = [strategy_param_sets[i][0] for i in range(len(strategy_names))]
    
    for meta_params in meta_param_dicts:
        signal_dfs = []
        for i, sname in enumerate(strategy_names):
            params = default_params[i]
            params_tuple = tuple(sorted((k, v) for k, v in params.items()))
            s_signal = strategy_signals[sname][params_tuple]
            signal_dfs.append(pd.DataFrame({"signal": s_signal}, index=df.index))
        
        final_signal = combine_signals(signal_dfs, buy_operator=buy_operator, sell_operator=sell_operator)
        
        # Quick backtest
        perf, _, _ = backtest_strategy(
            pd.DataFrame({"close_price": close_prices, "signal": final_signal}),
            initial_capital=initial_capital,
            min_holding_period=meta_params["min_holding_period"],
            precomputed_returns=precomputed_returns
        )
        
        # If performance is decent, consider this meta param set promising
        if perf > -0.1:  # Adjust threshold as needed
            promising_meta_params.append(meta_params)
    
    # If no promising meta params found, use all
    if not promising_meta_params:
        promising_meta_params = meta_param_dicts
    
    # Now evaluate full parameter grid only for promising meta params
    for meta_params in promising_meta_params:
        # Use itertools.product for efficient parameter combinations
        param_combinations = list(itertools.product(*strategy_param_sets))
        
        # Process in batches to control memory usage
        batch_size = 100  # Adjust based on available memory
        for i in range(0, len(param_combinations), batch_size):
            batch = param_combinations[i:i+min(batch_size, len(param_combinations)-i)]
            
            for combo_tuple in batch:
                # Build multi-strategy signals
                signal_dfs = []
                for i, sname in enumerate(strategy_names):
                    strat_params = combo_tuple[i]
                    params_tuple = tuple(sorted((k, v) for k, v in strat_params.items()))
                    
                    # Get the cached signal or compute it
                    if params_tuple not in strategy_signals[sname]:
                        strategy_signals[sname][params_tuple] = signal_cache.get(sname, df, **strat_params)
                    
                    s_signal = strategy_signals[sname][params_tuple]
                    signal_dfs.append(pd.DataFrame({"signal": s_signal}, index=df.index))
                
                final_signal = combine_signals(signal_dfs, buy_operator=buy_operator, sell_operator=sell_operator)
                
                # Backtest
                perf, portfolio_val, num_trades = backtest_strategy(
                    pd.DataFrame({"close_price": close_prices, "signal": final_signal}),
                    initial_capital=initial_capital,
                    min_holding_period=meta_params["min_holding_period"],
                    precomputed_returns=precomputed_returns
                )
                
                # Compute Sharpe
                strat_returns = final_signal.shift(1, fill_value=0) * precomputed_returns
                std_dev = strat_returns.std()
                sharpe = (strat_returns.mean() / std_dev) if std_dev > 0 else 0.0
                
                w = meta_params["sharpe_ratio_weight"]
                penalty = meta_params["penalty_factor"]
                score = (1 - w)*perf + (w*sharpe) - (penalty*num_trades)
                
                if score > best_score:
                    best_score = score
                    best_portfolio_value = portfolio_val
                    best_num_trades = num_trades
                    
                    # Build param dict
                    param_set_dict = {}
                    for i, sname in enumerate(strategy_names):
                        param_set_dict[sname] = combo_tuple[i]
                    
                    best_params_combo = {
                        "Strategies": param_set_dict,
                        "Meta": meta_params
                    }
            
            # Clean up intermediate results after each batch
            if i % 500 == 0:
                gc.collect()
    
    # Clean up
    strategy_signals.clear()
    gc.collect()
    
    return best_params_combo, best_score, best_portfolio_value, best_num_trades

def optimize_all_combinations(
    df: pd.DataFrame, 
    strategy_combinations,
    strategy_param_grid,
    initial_capital=10000,
    precomputed_returns: pd.Series = None
):
    """
    Optimized version to iterate over strategy combinations.
    """
    results = []
    
    # Ensure returns are precomputed
    if precomputed_returns is None:
        precomputed_returns = df["close_price"].pct_change().fillna(0)
    
    for combo in strategy_combinations:
        strategy_names, buy_op, sell_op = combo
        
        # Use the improved combo optimizer for better performance
        best_params, best_score, best_val, num_trades = optimize_strategy_combo_improved(
            df,
            strategy_names,
            strategy_param_grid,
            buy_operator=buy_op,
            sell_operator=sell_op,
            initial_capital=initial_capital,
            precomputed_returns=precomputed_returns
        )

        results.append({
            "StrategyCombo": strategy_names,
            "BuyOperator": buy_op,
            "SellOperator": sell_op,
            "BestParams": best_params,
            "TrainPerformance": best_score,
            "FinalPortfolioValue": best_val,
            "NumberOfTrades": num_trades
        })
        
        # Clean up memory periodically
        if len(results) % 5 == 0:
            gc.collect()

    return pd.DataFrame(results)