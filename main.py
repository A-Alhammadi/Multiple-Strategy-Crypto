# main.py

import pandas as pd
import numpy as np
import os
import psutil
import time
import gc

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
    optimize_single_strategies,  # Use the non-parallel version
    optimize_all_combinations  # Use the non-parallel version
)
from combo_signals import combine_signals
from optimizer import STRATEGY_FUNCTIONS, signal_cache
from backtest import backtest_strategy, buy_and_hold

def optimize_memory_usage():
    """
    Function to optimize memory usage by cleaning up Python's memory
    and running garbage collection.
    """
    gc.collect()
    
    # Track memory usage
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    
    print(f"Current memory usage: {memory_mb:.2f} MB")
    
    return memory_mb

def optimize_dataframe(df):
    """
    Optimize DataFrame memory usage by using more efficient data types.
    """
    # Float64 to float32 for numeric columns
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df[col] = df[col].astype('float32')
    
    # Int64 to int32 or int16 where possible
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if min_val > -32768 and max_val < 32767:
            df[col] = df[col].astype('int16')
        else:
            df[col] = df[col].astype('int32')
    
    return df

def process_currency(symbol, db):
    """Process a single currency"""
    try:
        print(f"\n=== Processing {symbol} ===")
        start_time = time.time()

        # 1) Fetch training + test data
        train_df = db.get_historical_data(symbol, TRAINING_START, TRAINING_END)
        test_df = db.get_historical_data(symbol, TESTING_START, TESTING_END)

        if len(train_df) < 2 or len(test_df) < 2:
            print(f"Skipping {symbol} due to insufficient data.")
            return None, None

        # Optimize DataFrame memory usage
        train_df = optimize_dataframe(train_df)
        test_df = optimize_dataframe(test_df)

        # 2) Precompute returns once for training and testing
        # Use float32 for better memory efficiency
        train_returns = train_df["close_price"].pct_change().fillna(0).astype('float32')
        test_returns = test_df["close_price"].pct_change().fillna(0).astype('float32')

        # 3) Optimize single strategies on training - use non-parallel version
        print(f"Optimizing single strategies for {symbol}...")
        single_df = optimize_single_strategies(
            df=train_df,
            strategy_param_grid=STRATEGY_PARAM_GRID,
            initial_capital=INITIAL_CAPITAL,
            precomputed_returns=train_returns
        )
        single_df["StrategyType"] = "single"

        # Clean up memory between major operations
        optimize_memory_usage()

        # 4) Optimize multi-strategy combos on training - use non-parallel version
        print(f"Optimizing strategy combinations for {symbol}...")
        combo_df = optimize_all_combinations(
            df=train_df,
            strategy_combinations=STRATEGY_COMBINATIONS,
            strategy_param_grid=STRATEGY_PARAM_GRID,
            initial_capital=INITIAL_CAPITAL,
            precomputed_returns=train_returns
        )
        combo_df["StrategyType"] = "combo"

        train_results = pd.concat([single_df, combo_df], ignore_index=True)
        train_results["Symbol"] = symbol

        # Clean up memory again
        optimize_memory_usage()
        
        # 5) Test each best strategy on the test set
        print(f"Testing optimized strategies for {symbol}...")
        bh_perf, bh_portfolio_val = buy_and_hold(test_df, INITIAL_CAPITAL)

        test_results = []
        # Process testing in batches to control memory usage
        batch_size = 10  # Number of strategies to test at once
        
        for i in range(0, len(train_results), batch_size):
            batch = train_results.iloc[i:i+batch_size]
            print(f"Testing batch {i//batch_size + 1} of {(len(train_results)-1)//batch_size + 1}")
            
            for _, row in batch.iterrows():
                strategy_combo = row["StrategyCombo"]
                buy_op = row["BuyOperator"]
                sell_op = row["SellOperator"]
                best_params_dict = row["BestParams"]

                # Build signals for entire test set
                if len(strategy_combo) == 1:
                    # Single strategy
                    sname = strategy_combo[0]
                    if isinstance(best_params_dict, dict):
                        if "Strategies" in best_params_dict:
                            # For single combos, might appear if stored that way
                            strat_params = best_params_dict["Strategies"].get(sname, {})
                        else:
                            strat_params = best_params_dict.get("strategy_params", {})
                    else:
                        strat_params = {}

                    # Use cached signals when possible
                    signal = signal_cache.get(sname, test_df, **strat_params)

                else:
                    # Multi-strategy
                    signal_dfs = []
                    for sname in strategy_combo:
                        if "Strategies" in best_params_dict:
                            strat_params = best_params_dict["Strategies"].get(sname, {})
                        else:
                            strat_params = {}
                        
                        # Use cached signals when possible
                        s_signal = signal_cache.get(sname, test_df, **strat_params)
                        signal_dfs.append(pd.DataFrame({"signal": s_signal}, index=test_df.index))

                    signal = combine_signals(signal_dfs, buy_operator=buy_op, sell_operator=sell_op)

                # Backtest on test data
                if best_params_dict and "Meta" in best_params_dict:
                    mhp = best_params_dict["Meta"].get("min_holding_period", 0)
                else:
                    mhp = 0

                # Create minimal DataFrame for backtest
                backtest_df = pd.DataFrame({"close_price": test_df["close_price"], "signal": signal})
                
                test_perf, test_portfolio_val, num_trades = backtest_strategy(
                    backtest_df,
                    initial_capital=INITIAL_CAPITAL,
                    min_holding_period=mhp,
                    precomputed_returns=test_returns
                )

                test_results.append({
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
                    "NumberOfTrades": num_trades,
                    "BuyHoldPerformance": bh_perf,
                    "BuyHoldValue": bh_portfolio_val,
                    "Diff_vs_BH_Perf": test_perf - bh_perf,
                    "Diff_vs_BH_Value": test_portfolio_val - bh_portfolio_val
                })
            
            # Clean up memory after each batch
            signal_dfs = None
            signal = None
            gc.collect()

        end_time = time.time()
        print(f"Processed {symbol} in {(end_time - start_time):.2f} seconds")
        
        # Add memory optimization after completing a currency
        optimize_memory_usage()
        
        return train_results, pd.DataFrame(test_results)
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    start_time = time.time()
    print(f"Starting optimization with {os.cpu_count()} CPU cores available")
    
    # Initialize database connection
    db = DatabaseHandler()
    
    all_detailed_train_results = []
    all_test_results = []
    
    # Process currencies sequentially - no parallelization
    print("Using sequential processing for all operations")
    for symbol in CURRENCIES:
        print(f"\nMemory usage before processing {symbol}: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.2f} MB")
        train_results, test_results = process_currency(symbol, db)
        if train_results is not None and test_results is not None:
            all_detailed_train_results.append(train_results)
            all_test_results.append(test_results)
        
        # Force memory cleanup between currencies
        train_results = None
        test_results = None
        signal_cache.cache.clear()  # Clear the signal cache
        gc.collect()
    
    # Save results to CSV
    if all_detailed_train_results:
        detailed_train_df = pd.concat(all_detailed_train_results, ignore_index=True)
        detailed_train_df.to_csv("detailed_training_results.csv", index=False)
        print("\n=== Detailed Training Results (All Symbols) ===")
        print(detailed_train_df.head(10))  # Only show first 10 rows to save memory in console
        print(f"Total training results: {len(detailed_train_df)} rows")

    if all_test_results:
        test_results_df = pd.concat(all_test_results, ignore_index=True)
        test_results_df.to_csv("detailed_testing_results.csv", index=False)
        print("\n=== Detailed Testing Results (All Symbols) ===")
        print(test_results_df.head(10))  # Only show first 10 rows
        print(f"Total test results: {len(test_results_df)} rows")
    
    end_time = time.time()
    print(f"\nTotal execution time: {(end_time - start_time) / 60:.2f} minutes")
    
    # Final memory report
    final_memory = optimize_memory_usage()
    print(f"Final memory usage: {final_memory:.2f} MB")
    
    db.close()

if __name__ == "__main__":
    main()