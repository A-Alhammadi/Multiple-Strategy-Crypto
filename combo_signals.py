# combo_signals.py

import numpy as np
import pandas as pd

def combine_signals(signal_dfs, buy_operator="AND", sell_operator="AND"):
    """
    Given a list of dataframes each containing a 'signal' column,
    combine them into a single signal using the specified logical operators
    for the buy side (+1) and the sell side (-1).

    signal = +1 if buy condition
    signal = -1 if sell condition
    signal =  0 otherwise

    buy_operator, sell_operator ∈ {"AND", "OR"}

    Note: Each DataFrame in `signal_dfs` must have the same index (date_time alignment).
    """
    # For convenience, put signals side by side
    combined = pd.concat([df["signal"] for df in signal_dfs], axis=1)
    combined.columns = [f"signal_{i}" for i in range(len(signal_dfs))]

    # We'll create buy_mask and sell_mask
    if buy_operator == "AND":
        buy_mask = (combined == 1).all(axis=1)  # all must be 1
    else:  # OR
        buy_mask = (combined == 1).any(axis=1)  # any 1

    if sell_operator == "AND":
        sell_mask = (combined == -1).all(axis=1)  # all must be -1
    else:  # OR
        sell_mask = (combined == -1).any(axis=1)  # any -1

    # Initialize combined signal
    final_signal = pd.Series(data=0, index=combined.index)
    final_signal[buy_mask] = 1
    final_signal[sell_mask] = -1

    # Forward-fill to hold positions
    final_signal = final_signal.replace(to_replace=0, method="ffill")

    return final_signal
