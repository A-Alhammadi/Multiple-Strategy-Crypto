# config.py

############################################################
#                  DATABASE CONFIGURATION                  #
############################################################

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

# Training period (for parameter optimization)
TRAINING_START = "2018-05-20"
TRAINING_END   = "2022-05-20"

# Testing period (out-of-sample evaluation)
TESTING_START  = "2022-05-21"
TESTING_END    = "2025-01-01"

# Which crypto pairs to test
CURRENCIES = ["BTC/USD", "ETH/USD", "XRP/USD", "LTC/USD"]

############################################################
#                  PARAMETER GRID                         #
############################################################
# We'll define parameter grids for each strategy you want to optimize.
# Some strategies have multiple parameters; others may have just one or two.

# Example parameter grids (adjust values as desired):
STRATEGY_PARAM_GRID = {
    "moving_average_crossover": {
        "short_window": [5, 10, 20],
        "long_window": [50, 100, 200]
    },
    "rsi": {
        "period": [7, 14],
        "buy_threshold": [25, 30],
        "sell_threshold": [70, 75]
    },
    "bollinger_bands": {
        "period": [14, 20]
    },
    "macd": {
        "fast_period": [12, 20],
        "slow_period": [26, 50],
        "signal_period": [9, 12]
    },
    "high_low_breakout": {
        "lookback": [12, 24]  # how many hours to look back
    },
    "volume_price_action": {
        "volume_multiplier": [1.5, 2.0]  # how big must volume spike be relative to an average
    },
    "vwap_zone": {
        "rsi_period": [7, 14],
        "rsi_lower": [40, 45],
        "rsi_upper": [55, 60]
    },
    "zscore_mean_reversion": {
        "zscore_window": [14, 20],
        "zscore_threshold": [2, 2.5]
    }
}

