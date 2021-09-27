#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import gym
import pandas as pd

from gym import error, spaces, utils
from gym.utils import seeding

class StockExchangeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, stock_exchange_history_dir):
        self.stock_exchange_history_dir = stock_exchange_history_dir

        data = pd.read_csv(self.stock_exchange_history_dir, sep=';', header=None, names=["code", "date", "opening", "max", "min", "closing", "?"])
        print(data)

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass
