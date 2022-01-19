#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import gym
import numpy as np
import matplotlib.pyplot as plt

from .actions.actions import Actions
from .states.stock_exchange_state import StockExchangeState
from .data_manager.stock_exchange_data_manager import StockExchangeDataManager

class StockExchangeEngineEnv(gym.Env):
    metadata = { 'render.modes': ['human'] }

    def __init__(self, config):
        super(StockExchangeEngineEnv, self).__init__()

        # Set config
        self.config = config

        # Request Data
        self.data_manager = StockExchangeDataManager(config.data_path)
        self.data = self.data_manager.get_index(self.config.index, self.config.source, self.config.start_date, self.config.end_date)
        self.data = self.data.set_index('Date')

        # Create a state:
        self.state = StockExchangeState(self.data, self.config)

        # Build gym env
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=[7, 10], dtype=np.float32)

    def reset(self):
        observation = self.state.reset()
        return observation
    
    def step(self, action_index):
        action = Actions(action_index)
        observation, reward, done, info = self.state.step(action)
        return observation, reward, done, info

    def render(self, mode='human') -> None:
        self.data['Close'].plot(label='Stocks At Closing', figsize=(16,5))
        self.data['Open'].plot(label='Stock Prices At Open', figsize=(16,5))
        self.data['High'].plot(label='Stock Prices On A High', figsize=(16,5))
        self.data['Low'].plot(label='Stock Prices On A Low', figsize=(16,5))

        plt.title(self.config.index + ' stock values')
        plt.xlabel('Time')
        plt.ylabel('Stock Values')
        plt.legend()

        plt.show()


    
