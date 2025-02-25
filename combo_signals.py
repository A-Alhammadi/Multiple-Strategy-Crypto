# combo_signals.py

import numpy as np
import pandas as pd

def combine_signals(signal_dfs, buy_operator="AND", sell_operator="AND"):
    """
    Combine multiple strategy signals into a single final signal
    using the specified buy/sell logical operators.
    signal = +1 (buy) if buy conditions meet, -1 (sell) if sell conditions meet,
    0 otherwise, then forward-filled to hold positions.
    """
    # Assume all have the same index
    idx = signal_dfs[0].index
    # Stack signals horizontally in a NumPy array
    signals_array = np.column_stack([df["signal"].values for df in signal_dfs])

    # Buy side
    if buy_operator == "AND":
        buy_mask = np.all(signals_array == 1, axis=1)
    else:  # OR
        buy_mask = np.any(signals_array == 1, axis=1)

    # Sell side
    if sell_operator == "AND":
        sell_mask = np.all(signals_array == -1, axis=1)
    else:  # OR
        sell_mask = np.any(signals_array == -1, axis=1)

    # Build final signal array
    final_signal = np.zeros(len(idx), dtype=np.int8)
    final_signal[buy_mask] = 1
    final_signal[sell_mask] = -1

    # Forward-fill any 0's
    for i in range(1, len(final_signal)):
        if final_signal[i] == 0:
            final_signal[i] = final_signal[i-1]

    return pd.Series(final_signal, index=idx)
