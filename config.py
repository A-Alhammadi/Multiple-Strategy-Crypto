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

CURRENCIES = ["BTC/USD", "ETH/USD", "LTC/USD", "XRP/USD"]
INITIAL_CAPITAL = 10000

TRADING_FEE_PCT = 0.001  # Example: 0.1%

############################################################
#                  PARAMETER GRID                          #
############################################################
STRATEGY_PARAM_GRID = {
    "moving_average_crossover": {
        "short_window": [5, 10, 20, 50],
        "long_window": [50, 100, 200, 300, 400]
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
#      GLOBAL (META) HYPERPARAMS TO REDUCE OVERTRADING     #
############################################################
# We'll treat these as "meta-parameters" tested alongside each strategy's normal parameters.
PENALTY_FACTOR_GRID = [0.0, 0.0005]  # Penalty per trade in your objective
MIN_HOLDING_PERIOD_GRID = [0, 5, 10] # In hours (0 means no minimum hold)
SHARPE_RATIO_WEIGHT_GRID = [0.0, 0.5, 1.0]
#  - 0.0 => pure returns
#  - 1.0 => pure Sharpe ratio
#  - 0.5 => 50% weight to Sharpe, 50% to raw returns

############################################################
#               STRATEGY COMBINATIONS                      #
############################################################
STRATEGY_COMBINATIONS = [
    (["moving_average_crossover", "rsi"], "AND", "OR"),
    (["moving_average_crossover", "rsi"], "OR", "AND"),
    (["macd", "bollinger_bands"], "AND", "OR"),
    (["macd", "bollinger_bands"], "OR", "AND"),
    (["rsi", "zscore_mean_reversion"], "AND", "AND"),
    (["rsi", "zscore_mean_reversion"], "OR", "OR"),
    (["high_low_breakout", "volume_price_action"], "AND", "AND"),
    (["high_low_breakout", "volume_price_action"], "OR", "OR"),
    (["vwap_zone", "rsi"], "AND", "AND"),
    (["vwap_zone", "rsi"], "OR", "OR"),
    (["moving_average_crossover", "rsi", "bollinger_bands"], "AND", "OR"),
    (["macd", "high_low_breakout", "volume_price_action"], "OR", "AND"),
    (["zscore_mean_reversion", "rsi", "vwap_zone"], "AND", "AND"),
    (["bollinger_bands", "high_low_breakout", "macd"], "OR", "OR"),
    (["moving_average_crossover", "macd"], "AND", "AND"),
    (["moving_average_crossover", "macd"], "OR", "OR"),
    (["rsi", "bollinger_bands"], "AND", "AND"),
    (["rsi", "bollinger_bands"], "OR", "OR"),
    (["macd", "zscore_mean_reversion"], "AND", "AND"),
    (["macd", "zscore_mean_reversion"], "OR", "OR"),
    (["rsi", "macd", "volume_price_action"], "AND", "AND"),
    (["zscore_mean_reversion", "bollinger_bands", "rsi"], "AND", "OR"),
    (["high_low_breakout", "vwap_zone", "moving_average_crossover"], "OR", "AND"),
]
