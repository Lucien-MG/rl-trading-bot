#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import os
import enum
import random

import gym
from gym import error, utils
from gym.utils import seeding

import numpy as np
import pandas as pd

class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Sell = 2

class RelativeSimulationStockExchangeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, stock_exchange_data_dir, state_size=10, cash=10000,
                conv_1d=False, commission_func=lambda x: x * 0.02):
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
        self.commission_func = commission_func
        self.state_size = state_size
        self.current_step = self.state_size

        self.max_cash = cash
        self.cash = cash

        self.stock = 0

    def _read_csv(self, path):
        df_stock_exchange = pd.read_csv(path, sep=';', header=None, names=["code", "date", "opening", "max", "min", "closing", "?"])

        for col in ["opening", "max", "min", "closing"]:
            df_stock_exchange[col] = df_stock_exchange[col].str.replace(',', '.').astype(float)

        return df_stock_exchange

    def step(self, action):
        assert self.action_space.contains(action)
        action = Actions(action)

        reward = 0
        state = np.zeros(shape=(3,self.state_size))
        state[0] = self.df_stock_exchange["max"][self.current_step - self.state_size:self.current_step].to_numpy()
        state[1] = self.df_stock_exchange["min"][self.current_step - self.state_size:self.current_step].to_numpy()
        state[2] = self.df_stock_exchange["closing"][self.current_step - self.state_size:self.current_step].to_numpy()

        # Next step:
        self.current_step += 1

        if action == Actions.Buy:
            self.stock += 1
            price = random.uniform(self.df_stock_exchange["min"][self.current_step], self.df_stock_exchange["max"][self.current_step])
            self.cash -= price + self.commission_func(price)
            reward += -0.1

        if action == Actions.Sell and self.stock != 0:
            self.stock -= 1
            price = random.uniform(self.df_stock_exchange["min"][self.current_step], self.df_stock_exchange["max"][self.current_step])
            self.cash += price - self.commission_func(price)
            reward += -0.1

        if action == Actions.Sell and self.stock == 0:
            reward += -1
            done = True

        if action == Actions.Skip:
            reward += -0.01

        reward += (self.cash - self.max_cash) / self.max_cash
        self.max_cash = max(self.cash, self.max_cash)

        # Determine if the env is finished
        done = self.current_step >= len(self.df_stock_exchange) - 1

        # Update info:
        info = {}

        return state, reward, done, info

    def reset(self):
        self.current_step = self.state_size
        self.cash = 1000
        self.max_cash = 1000
        self.stock = 0
        state = np.zeros(shape=(3,self.state_size))
        state[0] = self.df_stock_exchange["max"][self.current_step - self.state_size:self.current_step].to_numpy()
        state[1] = self.df_stock_exchange["min"][self.current_step - self.state_size:self.current_step].to_numpy()
        state[2] = self.df_stock_exchange["closing"][self.current_step - self.state_size:self.current_step].to_numpy()
        return state


    def render(self, mode='human'):
        pass
