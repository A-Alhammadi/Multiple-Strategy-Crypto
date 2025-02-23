# main.py

import pandas as pd
from database import DatabaseHandler
from config import (
    TRAINING_START, TRAINING_END,
    TESTING_START, TESTING_END,
    CURRENCIES,
    STRATEGY_PARAM_GRID,
    STRATEGY_COMBINATIONS,
    INITIAL_CAPITAL
)
from combo_optimizer import (
    optimize_single_strategies,
    optimize_all_combinations
)
from combo_signals import combine_signals
from optimizer import STRATEGY_FUNCTIONS
from backtest import backtest_strategy, buy_and_hold

def main():
    db = DatabaseHandler()

    # List to store all training results for all symbols
    all_detailed_train_results = []

    # List to store all testing results for all symbols
    all_test_results = []

    for symbol in CURRENCIES:
        print(f"\n=== Processing {symbol} ===")

        # 1) Fetch training + test data
        train_df = db.get_historical_data(symbol, TRAINING_START, TRAINING_END)
        test_df = db.get_historical_data(symbol, TESTING_START, TESTING_END)

        if len(train_df) < 2 or len(test_df) < 2:
            print(f"Skipping {symbol} due to insufficient data.")
            continue

        # 2) Optimize single strategies
        single_df = optimize_single_strategies(
            df=train_df.copy(),
            strategy_param_grid=STRATEGY_PARAM_GRID,
            initial_capital=INITIAL_CAPITAL
        )
        single_df["StrategyType"] = "single"

        # 3) Optimize multi-strategy combos
        combo_df = optimize_all_combinations(
            df=train_df.copy(),
            strategy_combinations=STRATEGY_COMBINATIONS,
            strategy_param_grid=STRATEGY_PARAM_GRID,
            initial_capital=INITIAL_CAPITAL
        )
        combo_df["StrategyType"] = "combo"

        # Merge single + combo results for training
        train_results = pd.concat([single_df, combo_df], ignore_index=True)
        train_results["Symbol"] = symbol

        # Save them for later reference
        all_detailed_train_results.append(train_results)

        # 4) Now test these best strategies on the entire test set
        # First get Buy & Hold for reference
        bh_perf, bh_portfolio_val = buy_and_hold(test_df.copy(), INITIAL_CAPITAL)

        # We'll create signals for each row's best strategy and backtest
        for _, row in train_results.iterrows():
            strategy_combo = row["StrategyCombo"]
            buy_op = row["BuyOperator"]
            sell_op = row["SellOperator"]
            best_params_dict = row["BestParams"]

            # Build the final signal over the entire test set:
            if len(strategy_combo) == 1:
                # Single strategy
                sname = strategy_combo[0]
                temp_df = test_df.copy()

                # If we have a dictionary of params for the single strategy
                if isinstance(best_params_dict, dict):
                    temp_df["signal"] = STRATEGY_FUNCTIONS[sname](temp_df, **best_params_dict)
                else:
                    # No params
                    temp_df["signal"] = STRATEGY_FUNCTIONS[sname](temp_df)

                final_signal = temp_df["signal"]

            else:
                # Multi-strategy
                signal_dfs = []
                for sname in strategy_combo:
                    temp_df = test_df.copy()

                    # best_params_dict is a dict of dicts => best_params_dict[sname]
                    strat_params = best_params_dict[sname] if sname in best_params_dict else {}
                    temp_df["signal"] = STRATEGY_FUNCTIONS[sname](temp_df, **strat_params)
                    signal_dfs.append(temp_df[["signal"]])

                final_signal = combine_signals(signal_dfs, buy_operator=buy_op, sell_operator=sell_op)

            # Insert final signal in a copy
            test_with_signal = test_df.copy()
            test_with_signal["signal"] = final_signal

            # 5) Backtest
            test_perf, test_portfolio_val = backtest_strategy(test_with_signal, INITIAL_CAPITAL)

            # 6) Record results
            all_test_results.append({
                "Symbol": symbol,
                "StrategyType": row["StrategyType"],
                "StrategyCombo": strategy_combo,
                "BuyOperator": buy_op,
                "SellOperator": sell_op,
                "BestParams": best_params_dict,
                "TrainPerformance": row["TrainPerformance"],
                "TrainFinalValue": row["FinalPortfolioValue"],
                "TestPerformance": test_perf,
                "TestFinalValue": test_portfolio_val,
                "BuyHoldPerformance": bh_perf,
                "BuyHoldValue": bh_portfolio_val,
                "Diff_vs_BH_Perf": test_perf - bh_perf,
                "Diff_vs_BH_Value": test_portfolio_val - bh_portfolio_val
            })

    # 7) Save training results
    if all_detailed_train_results:
        detailed_train_df = pd.concat(all_detailed_train_results, ignore_index=True)
        detailed_train_df.to_csv("detailed_training_results.csv", index=False)
        print("\n=== Detailed Training Results (All Symbols) ===")
        print(detailed_train_df)

    # 8) Save test results
    test_results_df = pd.DataFrame(all_test_results)
    test_results_df.to_csv("detailed_testing_results.csv", index=False)
    print("\n=== Detailed Testing Results (All Symbols) ===")
    print(test_results_df)

    db.close()

if __name__ == "__main__":
    main()
