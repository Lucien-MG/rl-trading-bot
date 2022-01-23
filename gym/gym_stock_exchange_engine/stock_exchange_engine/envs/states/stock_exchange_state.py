#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import numpy as np
from collections import deque

from ..actions.actions import Actions
from .stock_exchange_state_interface import StockExchangeStateInterface

class StockExchangeState(StockExchangeStateInterface):

    def __init__(self, data, config):
        self.data = data
        self.config = config

    def encode(self):
        high = self.data['High'][self.offset - self.config.nb_bars: self.offset].to_numpy()
        low = self.data['Low'][self.offset - self.config.nb_bars: self.offset].to_numpy()
        open = self.data['Open'][self.offset - self.config.nb_bars: self.offset].to_numpy()
        close = self.data['Close'][self.offset - self.config.nb_bars: self.offset].to_numpy()
        volume = self.data['Volume'][self.offset - self.config.nb_bars: self.offset].to_numpy()
        stock = np.array(self.stock_history)
        account = np.array(self.account_history)

        matrix = np.stack([high, low, open, close, volume, stock, account])
        return matrix

    def reset(self):
        self.offset = self.config.nb_bars
        self.account = self.config.initial_account
        self.stock = 0

        self.stock_history = deque([0] * self.config.nb_bars, maxlen=self.config.nb_bars)
        self.account_history = deque([self.account] * self.config.nb_bars, maxlen=self.config.nb_bars)

        return self.encode()

    def step(self, action):
        observation, reward, done, info  = None, 0.0, False, {}

        if action == Actions.Buy and not self.stock:
            cost = self.data["Close"][self.offset] + self.data["Close"][self.offset] * self.config.commission

            if cost <= self.account:
                self.account -= cost
                self.stock += 1

        if action == Actions.Close and self.stock:
            self.stock -= 1
            self.account += self.data["Close"][self.offset]
            self.account -= self.data["Close"][self.offset] * self.config.commission

        # Next step
        self.stock_history.append(self.stock * self.data["Close"][self.offset])
        self.account_history.append(self.account)
        self.offset += 1

        # Nest step state
        observation = self.encode()
        done |= self.offset >= len(self.data)
        reward += ((self.account - self.config.initial_account) * 100) / self.config.initial_account if done else 0
        info = {}

        return observation, reward, done, info
