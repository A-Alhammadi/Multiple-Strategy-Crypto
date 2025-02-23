# tester.py

import ast  # for safely parsing dict/list strings from input
import pandas as pd

from database import DatabaseHandler
from config import INITIAL_CAPITAL  # or you can just define inside this file
from backtest import backtest_strategy, buy_and_hold
from optimizer import STRATEGY_FUNCTIONS
from combo_signals import combine_signals

def test_combo(
    symbol: str,
    start_date: str,
    end_date: str,
    strategy_combo,
    buy_operator: str,
    sell_operator: str,
    best_params,
    initial_capital=10000
):
    """
    Fetch data for the given symbol from start_date to end_date,
    combine signals from the given strategies + best_params,
    run a backtest, return stats.
    
    strategy_combo: list of strategy names, e.g. ["zscore_mean_reversion", "rsi", "vwap_zone"]
    buy_operator, sell_operator: "AND" or "OR"
    best_params: e.g. {
       "zscore_mean_reversion": {"zscore_window": 30, "zscore_threshold": 3},
       "rsi": {"period": 7, "buy_threshold": 35, "sell_threshold": 65},
       "vwap_zone": {"rsi_period": 7, "rsi_lower": 45, "rsi_upper": 60}
    }
    """

    db = DatabaseHandler()
    df = db.get_historical_data(symbol, start_date, end_date)
    db.close()

    if len(df) < 2:
        print(f"No data returned for {symbol} in {start_date} to {end_date}. Exiting.")
        return None

    # Calculate buy-and-hold for reference
    bh_perf, bh_val = buy_and_hold(df.copy(), initial_capital)

    # Build the final combined signal
    if len(strategy_combo) == 1:
        # Single strategy
        sname = strategy_combo[0]
        params = best_params[sname] if sname in best_params else {}
        df["signal"] = STRATEGY_FUNCTIONS[sname](df.copy(), **params)
    else:
        # Multi-strategy
        signal_dfs = []
        for sname in strategy_combo:
            temp_df = df.copy()
            params = best_params.get(sname, {})
            temp_df["signal"] = STRATEGY_FUNCTIONS[sname](temp_df, **params)
            signal_dfs.append(temp_df[["signal"]])

        final_signal = combine_signals(signal_dfs, buy_operator=buy_operator, sell_operator=sell_operator)
        df["signal"] = final_signal

    # Run backtest
    total_return, final_portfolio_val, num_trades = backtest_strategy(df, initial_capital)

    # Print results
    print("\n=== Test Results ===")
    print(f"Symbol:            {symbol}")
    print(f"Date Range:        {start_date} to {end_date}")
    print(f"Strategy Combo:    {strategy_combo}")
    print(f"Buy Operator:      {buy_operator}")
    print(f"Sell Operator:     {sell_operator}")
    print(f"Best Params:       {best_params}")
    print(f"---")
    print(f"Strategy Return:   {total_return:.2%}")
    print(f"Strategy Final Val: ${final_portfolio_val:,.2f}")
    print(f"Number of Trades:  {num_trades}")
    print(f"---")
    print(f"Buy & Hold Return: {bh_perf:.2%}")
    print(f"Buy & Hold Val:    ${bh_val:,.2f}")

    # Optionally return results as a dict
    return {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "strategy_combo": strategy_combo,
        "buy_operator": buy_operator,
        "sell_operator": sell_operator,
        "best_params": best_params,
        "strategy_return": total_return,
        "final_portfolio_val": final_portfolio_val,
        "num_trades": num_trades,
        "bh_return": bh_perf,
        "bh_val": bh_val
    }


def main():
    # Example of interactive prompts (or you can hard-code them).
    symbol_input = input("Enter comma-separated symbols (e.g. 'BTC/USD,ETH/USD'): ")
    start_date = input("Start date (YYYY-MM-DD): ")
    end_date   = input("End date   (YYYY-MM-DD): ")

    # Strategy combo can be a Python list string, e.g. "['zscore_mean_reversion','rsi','vwap_zone']"
    strategy_combo_str = input("Enter strategy combo list (e.g. ['zscore_mean_reversion','rsi','vwap_zone']): ")
    # Safely parse that string into a Python list
    strategy_combo = ast.literal_eval(strategy_combo_str)

    buy_operator  = input("Buy operator (AND/OR): ")
    sell_operator = input("Sell operator (AND/OR): ")

    # best_params is a dictionary-of-dictionaries.
    # For example: 
    # {
    #   'zscore_mean_reversion': {'zscore_window': 30, 'zscore_threshold': 3},
    #   'rsi': {'period': 7, 'buy_threshold': 35, 'sell_threshold': 65},
    #   'vwap_zone': {'rsi_period': 7, 'rsi_lower': 45, 'rsi_upper': 60}
    # }
    best_params_str = input("Enter best_params as a dict (e.g. { ... }): ")
    best_params = ast.literal_eval(best_params_str)

    # Convert the user’s symbol input into a list
    symbols = [s.strip() for s in symbol_input.split(",")]

    # Run test for each symbol
    for sym in symbols:
        results = test_combo(
            symbol=sym,
            start_date=start_date,
            end_date=end_date,
            strategy_combo=strategy_combo,
            buy_operator=buy_operator,
            sell_operator=sell_operator,
            best_params=best_params,
            initial_capital=INITIAL_CAPITAL  # or define your own number
        )
        # You could store or further process `results` here

if __name__ == "__main__":
    main()
