#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import os
import enum

import gym
from gym import error, utils
from gym.utils import seeding

import numpy as np
import pandas as pd

class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2

class StockExchangeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, stock_exchange_data_dir, state_size=10):
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Discrete(state_size)

        # Read and concat csv:
        csv_names = os.listdir(path=stock_exchange_data_dir)
        open_csv = [self._read_csv(os.path.join(stock_exchange_data_dir, csv_name)) for csv_name in csv_names]

        # Cleaning data, removing duplicates and sorting by dates
        self.df_stock_exchange = pd.concat(open_csv, axis=0, ignore_index=True)
        self.df_stock_exchange.drop_duplicates(subset=['date'])
        self.df_stock_exchange.sort_values(by=['date'])

        # Define states:
        self.commission_perc = 0.01
        self.state_size = state_size
        self.current_step = self.state_size

    def _read_csv(self, path):
        return pd.read_csv(path, sep=';', header=None, names=["code", "date", "opening", "max", "min", "closing", "?"])

    def step(self, action):
        assert self.action_space.contains(action)

        state = self.df_stock_exchange[["opening", "max", "min", "closing"]][self.current_step - self.state_size:self.current_step].to_numpy()
        reward = 0
        done = self.current_step >= len(self.df_stock_exchange) - 1
        info = {}

        if action == Actions.Buy and not self.have_position:
            reward -= reward * self.commission_perc
        elif action == Actions.Close and self.have_position:
            reward -= reward * self.commission_perc

        self.current_step += 1

        return state, reward, done, info

    def reset(self):
        self.current_step = self.state_size

    def render(self, mode='human'):
        pass
