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

    def __init__(self, stock_exchange_data_dir, cash=1000, state_size=10, commission_func=lambda p : p * 0.05):
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Discrete(state_size)

        # Read and concat csv:
        csv_names = os.listdir(path=stock_exchange_data_dir)
        open_csv = [self._read_csv(os.path.join(stock_exchange_data_dir, csv_name)) for csv_name in csv_names]

        # Cleaning data, removing duplicates and sorting by dates
        self.df_stock_exchange = pd.concat(open_csv, axis=0, ignore_index=True)
        self.df_stock_exchange.drop_duplicates(subset=['date'])
        self.df_stock_exchange.sort_values(by=['date'])

        # Pass price to float64:
        for column in ['opening', 'max', 'min', 'closing']:
            self.df_stock_exchange[column] = [x.replace(',', '.') for x in self.df_stock_exchange[column]]
            self.df_stock_exchange[column] = self.df_stock_exchange[column].astype(float)

        # Define states:
        self.commission_func = commission_func
        self.state_size = state_size
        self.current_step = self.state_size

        self.cash = cash
        self.stock = 0

    def _read_csv(self, path):
        return pd.read_csv(path, sep=';', header=None, names=["code", "date", "opening", "max", "min", "closing", "?"], decimal=".")

    def step(self, action):
        assert self.action_space.contains(action)

        # Transoform int into action
        action = Actions(action)

        # Define state, reward, done, info that will be returned
        state = self.df_stock_exchange[["opening", "max", "min", "closing"]][self.current_step - self.state_size:self.current_step].to_numpy()
        reward = 0
        done = self.current_step >= len(self.df_stock_exchange) - 1
        info = {}

        # Actions:
        if action == Actions.Skip:
            reward = 0

        if action == Actions.Buy:
            self.cash -= self.commission_func((state[-1][1] + state[-1][2]) / 2)
            self.stock += 1
            reward -= 0.1

        if action == Actions.Close:
            if self.stock:
                self.cash += self.commission_func((state[-1][1] + state[-1][2]) / 2)
                self.stock -= 1
                reward -= 0.1
            else:
                reward -= 1

        # Calculate reward
        reward += 

        # Next step:
        self.current_step += 1

        # Update info:
        info = { "cash": self.cash, "stock": self.stock }

        return state, reward, done, info

    def reset(self):
        self.current_step = self.state_size

    def render(self, mode='human'):
        pass
