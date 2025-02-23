# combo_optimizer.py

import itertools
import pandas as pd
from typing import List, Dict
from backtest import backtest_strategy
from combo_signals import combine_signals
from optimizer import generate_param_dicts, STRATEGY_FUNCTIONS, optimize_strategy

def optimize_single_strategies(
    df: pd.DataFrame,
    strategy_param_grid: Dict[str, Dict],
    initial_capital: float = 10000
) -> pd.DataFrame:
    """
    For each individual strategy, run a parameter search over the entire 'df' (no volatility regime).
    Returns a DataFrame of results with columns:
      StrategyName, BestParams, TrainPerformance, FinalPortfolioValue
    """
    results = []

    for strategy_name in STRATEGY_FUNCTIONS.keys():
        # param grid for this strategy
        params_dict = strategy_param_grid.get(strategy_name, {})

        # if no param grid, pass an empty dict => no tuning
        best_params, best_perf, best_val = optimize_strategy(
            df,
            strategy_name=strategy_name,
            strategy_param_grid=params_dict,
            initial_capital=initial_capital
        )

        results.append({
            "StrategyCombo": [strategy_name],  # single strategy in a list
            "BuyOperator": None,
            "SellOperator": None,
            "BestParams": best_params,
            "TrainPerformance": best_perf,
            "FinalPortfolioValue": best_val
        })

    return pd.DataFrame(results)


def optimize_strategy_combo(
    df: pd.DataFrame, 
    strategy_names: List[str],
    param_grids: Dict[str, Dict],
    buy_operator: str,
    sell_operator: str,
    initial_capital: float = 10000
):
    """
    Finds best parameters for the given multi-strategy combo (strategy_names)
    using the param grids. Returns (best_params_combo, best_performance, best_portfolio_value).
    """
    strategy_param_combos = []
    for sname in strategy_names:
        # Generate all param combinations for each strategy
        param_dicts = generate_param_dicts(param_grids.get(sname, {}))
        if not param_dicts:
            param_dicts = [{}]  # handle no-parameter strategies
        strategy_param_combos.append(param_dicts)

    best_params_combo = None
    best_performance = float("-inf")
    best_portfolio_value = 0

    # Cartesian product over each strategy's parameter sets
    for combo_tuple in itertools.product(*strategy_param_combos):
        signal_dfs = []
        for i, sname in enumerate(strategy_names):
            df_temp = df.copy()
            params = combo_tuple[i]
            df_temp["signal"] = STRATEGY_FUNCTIONS[sname](df_temp, **params)
            signal_dfs.append(df_temp[["signal"]])

        # Combine signals for multi-strategy
        final_signal = combine_signals(signal_dfs, buy_operator=buy_operator, sell_operator=sell_operator)
        df_combo = df.copy()
        df_combo["signal"] = final_signal

        perf, portfolio_value = backtest_strategy(df_combo, initial_capital)

        if perf > best_performance:
            best_performance = perf
            best_portfolio_value = portfolio_value
            # Build a dict mapping each strategy to the chosen parameter set
            param_set_dict = {}
            for i, sname in enumerate(strategy_names):
                param_set_dict[sname] = combo_tuple[i]
            best_params_combo = param_set_dict

    return best_params_combo, best_performance, best_portfolio_value


def optimize_all_combinations(
    df: pd.DataFrame, 
    strategy_combinations,
    strategy_param_grid,
    initial_capital=10000
):
    """
    For each (strategy combo, buy_operator, sell_operator) triple,
    run an optimization over 'df'. No regime logic. Returns a DataFrame with columns:
      StrategyCombo, BuyOperator, SellOperator, BestParams, TrainPerformance, FinalPortfolioValue
    """
    results = []
    for combo in strategy_combinations:
        strategy_names, buy_op, sell_op = combo
        best_params, best_perf, best_val = optimize_strategy_combo(
            df,
            strategy_names,
            strategy_param_grid,
            buy_operator=buy_op,
            sell_operator=sell_op,
            initial_capital=initial_capital
        )

        results.append({
            "StrategyCombo": strategy_names,
            "BuyOperator": buy_op,
            "SellOperator": sell_op,
            "BestParams": best_params,
            "TrainPerformance": best_perf,
            "FinalPortfolioValue": best_val
        })

    return pd.DataFrame(results)
