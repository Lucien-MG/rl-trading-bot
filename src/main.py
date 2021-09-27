#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import gym
import gym_stock_exchange

env = gym.make("gym_stock_exchange:gym_stock_exchange-v0", stock_exchange_history_dir="data/cac40/2019_2021.csv")
