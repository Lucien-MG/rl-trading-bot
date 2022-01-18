#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

# Path to the environment config:
ENVIRONMENT_CONFIG_PATH: str = None

# Choose the environement to use:
NAME: str = "stock_exchange_api_env" # Choose the environment to use
SIMULATION: bool = True # Trade with simulation not with a real api

# Choose the source where data come from ands the index to use:
SOURCE: str = "local" # "yahoo"
INDEX: str = "YNDX" # "^FCHI"
DATA_PATH: str = "~/.rltrade/data" # only needed in case of using local data

# Choose the time that will be use in the environment
START_DATE: str = "2015-01-01"
END_DATE: str = "2015-12-31"

# Bars history
NB_BARS: int = 10

# Data encoding
DATA_ENCODING: str = "matrix"

# Stock Exchange parameters
INITIAL_ACCOUNT: float = 100
COMMISSION: float = 0.1

# Render the environment
RENDER: bool = False
