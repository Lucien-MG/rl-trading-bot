#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import numpy as np
from collections import deque

from ..actions.actions import Actions
from .stock_exchange_state_interface import StockExchangeStateInterface

class SimpleStockExchangeState(StockExchangeStateInterface):

    def __init__(self, data, config):
        self.data = data
        self.config = config

    def reset(self):
        self.offset = self.config.nb_bars
        self.last_price = 0
        self.stock = 0

        return self.encode()

    def encode(self):
        high = self.data['high'][self.offset - self.config.nb_bars: self.offset].to_numpy()
        low = self.data['low'][self.offset - self.config.nb_bars: self.offset].to_numpy()
        close = self.data['close'][self.offset - self.config.nb_bars: self.offset].to_numpy()
        volume = self.data['volume'][self.offset - self.config.nb_bars: self.offset].to_numpy()

        matrix = np.stack([high, low, close, volume])

        matrix = matrix.transpose()
        matrix = matrix.flatten()

        matrix = np.append(matrix, (self.data["close"][self.offset] - self.last_price) / self.last_price if self.stock else 0)

        return matrix

    def step(self, action):
        observation, reward, done, info  = None, 0.0, False, {}

        if action == Actions.Buy and not self.stock:
            self.stock += 1
            reward += -0.01
            self.last_price = self.data["close"][self.offset]

        if action == Actions.Close and self.stock:
            self.stock -= 1
            reward += -0.01
            reward += (self.data["close"][self.offset] - self.last_price) / self.last_price
            self.last_price = 0

        # Next step
        self.offset += 1

        # Nest step state
        observation = self.encode()
        reward += ((self.data["close"][self.offset] - self.last_price) / self.last_price) / 10 if self.last_price else 0
        done |= self.offset >= len(self.data) - 1
        info = {}

        return observation, reward, done, info

StockExchangeState = SimpleStockExchangeState
