#!/usr/bin/python3
# ‑∗‑ coding: utf‑8 ‑∗‑

import gym
import numpy as np
import matplotlib.pyplot as plt

from .actions.actions import Actions
from stock_exchange_engine.envs.states.simple_state import SimpleStockExchangeState
from .states.stock_exchange_state import StockExchangeState
from .states.stock_exchange_state_intraday import StockExchangeStateIntraday
from .data_manager.stock_exchange_data_manager import StockExchangeDataManager


class StockExchangeEngineEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        super(StockExchangeEngineEnv, self).__init__()

        # Set config
        self.config = config

        # Request Data
        self.data_manager = StockExchangeDataManager(config.data_path)
        self.data = self.data_manager.get_index(
            self.config.index, self.config.source, self.config.start_date, self.config.end_date, self.config.interval)

        # Create a state:
        self.state = SimpleStockExchangeState(self.data, self.config)

        # Build gym env attributes
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=[7, 32], dtype=np.float32)

        self.gain = 0

    def reset(self):
        if hasattr(self.state, 'gain'):
            self.gain = self.state.gain

        # Reset state and return the first observation
        observation = self.state.reset()

        return observation

    def step(self, action_index):
        # Transform given action in actual enumerate action
        action = Actions(action_index)

        # Execute one step
        observation, reward, done, info = self.state.step(action)

        return observation, reward, done, info

    def render(self, mode='human') -> None:
        self.data['Close'].plot(label='Stocks At Closing', figsize=(16, 5))
        self.data['Open'].plot(label='Stock Prices At Open', figsize=(16, 5))
        self.data['High'].plot(label='Stock Prices On A High', figsize=(16, 5))
        self.data['Low'].plot(label='Stock Prices On A Low', figsize=(16, 5))

        plt.title(self.config.index + ' stock values')
        plt.xlabel('Date')
        plt.ylabel('Stock Values')
        plt.legend()

        plt.show()
