# main.py

import pandas as pd
from database import DatabaseHandler
from config import (
    TRAINING_START, TRAINING_END,
    TESTING_START, TESTING_END,
    CURRENCIES,
    STRATEGY_PARAM_GRID
)
from optimizer import optimize_strategy, STRATEGY_FUNCTIONS
from backtest import backtest_strategy, buy_and_hold

INITIAL_CAPITAL = 10000  # Starting capital for trading

def main():
    db = DatabaseHandler()

    all_results = []

    for symbol in CURRENCIES:
        print(f"\n=== Optimizing strategies for {symbol} ===")
        # Fetch training data
        train_df = db.get_historical_data(symbol, TRAINING_START, TRAINING_END)
        # Fetch testing data
        test_df = db.get_historical_data(symbol, TESTING_START, TESTING_END)

        if len(train_df) == 0 or len(test_df) == 0:
            print(f"Skipping {symbol} due to lack of data.")
            continue

        # For each strategy
        for strategy_name, strategy_func in STRATEGY_FUNCTIONS.items():
            print(f"\n--- Strategy: {strategy_name} ---")
            param_grid = STRATEGY_PARAM_GRID.get(strategy_name, {})

            # ✅ Fix: Unpack three return values now
            best_params, best_train_perf, best_train_value = optimize_strategy(
                train_df.copy(), strategy_name, param_grid, INITIAL_CAPITAL
            )
            print(f"Best Params: {best_params}")
            print(f"Best Training Performance: {best_train_perf:.4f}")
            print(f"Final Portfolio Value (Train): ${best_train_value:.2f}")

            # Evaluate on test data
            test_df_copy = test_df.copy()
            test_df_copy["signal"] = strategy_func(test_df_copy, **best_params)
            test_perf, final_strategy_value = backtest_strategy(test_df_copy, INITIAL_CAPITAL)
            print(f"Test Performance: {test_perf:.4f}")
            print(f"Final Portfolio Value (Strategy): ${final_strategy_value:.2f}")

            # Compare with Buy & Hold
            bh_perf, final_bh_value = buy_and_hold(test_df_copy, INITIAL_CAPITAL)
            print(f"Buy & Hold Performance: {bh_perf:.4f}")
            print(f"Final Portfolio Value (Buy & Hold): ${final_bh_value:.2f}")

            # Calculate difference vs. Buy & Hold
            diff_vs_bh = test_perf - bh_perf
            value_diff_vs_bh = final_strategy_value - final_bh_value
            print(f"Difference vs B&H: {diff_vs_bh:.4f}")
            print(f"Final Portfolio Difference: ${value_diff_vs_bh:.2f}")

            # Store results
            all_results.append({
                "Currency": symbol,
                "Strategy": strategy_name,
                "Best Params": best_params,
                "Train Performance": best_train_perf,
                "Test Performance": test_perf,
                "Buy & Hold Return": bh_perf,
                "Strategy vs B&H (Diff)": diff_vs_bh,
                "Final Portfolio Value (Strategy)": final_strategy_value,
                "Final Portfolio Value (Buy & Hold)": final_bh_value,
                "Final Portfolio Difference": value_diff_vs_bh
            })

    # Close DB connection
    db.close()

    # Save to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("strategy_results.csv", index=False)
    print("\n=== Final Results ===")
    print(results_df)

if __name__ == "__main__":
    main()
