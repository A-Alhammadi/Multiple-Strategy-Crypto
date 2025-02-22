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
            # Grab the parameter grid for this strategy
            param_grid = STRATEGY_PARAM_GRID.get(strategy_name, {})

            # 1) Optimize on training data
            best_params, best_train_perf = optimize_strategy(train_df.copy(), strategy_name, param_grid)
            print(f"Best Params: {best_params}")
            print(f"Best Training Performance: {best_train_perf:.4f}")

            # 2) Evaluate on test data using best_params
            test_df_copy = test_df.copy()
            test_df_copy["signal"] = strategy_func(test_df_copy, **best_params)
            test_perf = backtest_strategy(test_df_copy)
            print(f"Test Performance: {test_perf:.4f}")

            # 3) Compare with Buy & Hold on test data
            bh_perf = buy_and_hold(test_df_copy)
            print(f"Buy & Hold Performance: {bh_perf:.4f}")

            # Store results
            all_results.append({
                "Currency": symbol,
                "Strategy": strategy_name,
                "Best Params": best_params,
                "Train Performance": best_train_perf,
                "Test Performance": test_perf,
                "Buy & Hold": bh_perf
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
