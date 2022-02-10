#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

# Path to the environment config:
ENVIRONMENT_CONFIG_PATH: str = None

# Choose the environement to use:
ENVIRONMENT_ID: str = "stock_exchange_engine:stock_exchange_engine_env-v0"
SIMULATION: bool = True

# Choose the source where data come from ands the index to use:
SOURCE: str = "yahoo"
INDEX: str = "YNDX"
DATA_PATH: str = "~/.rltrade/data"

# Choose the time that will be use in the environment
# fetch data by interval (including intraday if period < 60 days)
# valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
INTERVAL: str = "1m"
START_DATE: str = "2015-01-01"
END_DATE: str = "2015-12-31"

# Bars history
NB_BARS: int = 10

# Stock Exchange parameters
INITIAL_ACCOUNT: float = 1000
COMMISSION: float = 0.1

# Render the environment
RENDER: bool = False
