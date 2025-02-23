# config.py

DB_CONFIG = {
    'dbname': 'cryptocurrencies',
    'user': 'myuser',
    'password': 'mypassword',
    'host': 'localhost',
    'port': '5432'
}

############################################################
#                   BACKTEST SETTINGS                      #
############################################################

TRAINING_START = "2018-05-20"
TRAINING_END   = "2022-05-20"

TESTING_START  = "2022-05-21"
TESTING_END    = "2025-01-01"

CURRENCIES = ["ETH/USD", "XRP/USD"]
INITIAL_CAPITAL = 10000

############################################################
#                  PARAMETER GRID                          #
############################################################
STRATEGY_PARAM_GRID = {
    "moving_average_crossover": {
        "short_window": [5, 10, 20],
        "long_window": [50, 100, 200]
    },
    "rsi": {
        "period": [7, 14],
        "buy_threshold": [25, 30, 35],
        "sell_threshold": [65, 70, 75]
    },
    "bollinger_bands": {
        "period": [14, 20],
        "std_dev": [2, 2.5, 3]
    },
    "macd": {
        "fast_period": [12, 20],
        "slow_period": [26, 50],
        "signal_period": [9, 12]
    },
    "high_low_breakout": {
        "lookback": [12, 24, 48]
    },
    "volume_price_action": {
        "volume_multiplier": [1.5, 2.0, 2.5]
    },
    "vwap_zone": {
        "rsi_period": [7, 14],
        "rsi_lower": [40, 45, 50],
        "rsi_upper": [55, 60, 65]
    },
    "zscore_mean_reversion": {
        "zscore_window": [14, 20, 30],
        "zscore_threshold": [2, 2.5, 3]
    }
}
############################################################
#               STRATEGY COMBINATIONS                      #
############################################################
# List of strategy combinations using logical conditions (AND/OR)
# Each entry is a tuple: (list_of_strategies, logical_operator_for_BUY, logical_operator_for_SELL).

STRATEGY_COMBINATIONS = [
    # Moving Average and RSI (Momentum-based)
    (["moving_average_crossover", "rsi"], "AND", "OR"),
    (["moving_average_crossover", "rsi"], "OR", "AND"),

    # MACD with Bollinger Bands (Trend-Following & Volatility)
    (["macd", "bollinger_bands"], "AND", "OR"),
    (["macd", "bollinger_bands"], "OR", "AND"),

    # RSI with Z-Score Mean Reversion (Reversal Trading)
    (["rsi", "zscore_mean_reversion"], "AND", "AND"),
    (["rsi", "zscore_mean_reversion"], "OR", "OR"),

    # High-Low Breakout with Volume Price Action (Breakout & Confirmation)
    (["high_low_breakout", "volume_price_action"], "AND", "AND"),
    (["high_low_breakout", "volume_price_action"], "OR", "OR"),

    # VWAP with RSI (Mean Reversion in VWAP Zones)
    (["vwap_zone", "rsi"], "AND", "AND"),
    (["vwap_zone", "rsi"], "OR", "OR"),

    # Multi-factor Strategy Combinations (3 Strategies)
    (["moving_average_crossover", "rsi", "bollinger_bands"], "AND", "OR"),
    (["macd", "high_low_breakout", "volume_price_action"], "OR", "AND"),
    (["zscore_mean_reversion", "rsi", "vwap_zone"], "AND", "AND"),
    (["bollinger_bands", "high_low_breakout", "macd"], "OR", "OR"),

    # Experiment with different logical operators
    (["moving_average_crossover", "macd"], "AND", "AND"),
    (["moving_average_crossover", "macd"], "OR", "OR"),
    (["rsi", "bollinger_bands"], "AND", "AND"),
    (["rsi", "bollinger_bands"], "OR", "OR"),
    (["macd", "zscore_mean_reversion"], "AND", "AND"),
    (["macd", "zscore_mean_reversion"], "OR", "OR"),

    # Additional advanced combinations
    (["rsi", "macd", "volume_price_action"], "AND", "AND"),
    (["zscore_mean_reversion", "bollinger_bands", "rsi"], "AND", "OR"),
    (["high_low_breakout", "vwap_zone", "moving_average_crossover"], "OR", "AND"),
]

