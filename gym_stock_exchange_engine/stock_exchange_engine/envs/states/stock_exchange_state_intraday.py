#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import datetime
import numpy as np
import pandas as pd
from collections import deque

from ..actions.actions import Actions
from .stock_exchange_state_interface import StockExchangeStateInterface

class StockExchangeStateIntraday(StockExchangeStateInterface):

    def __init__(self, data, config):
        self.data = data
        self.config = config

        # Get all unique days:
        self.days = pd.DataFrame(data.index.date, columns=["Date"]).drop_duplicates('Date')

    def encode(self):
        high = self.data_day['High'][self.offset - self.config.nb_bars: self.offset].to_numpy()
        low = self.data_day['Low'][self.offset - self.config.nb_bars: self.offset].to_numpy()
        open = self.data_day['Open'][self.offset - self.config.nb_bars: self.offset].to_numpy()
        close = self.data_day['Close'][self.offset - self.config.nb_bars: self.offset].to_numpy()
        volume = self.data_day['Volume'][self.offset - self.config.nb_bars: self.offset].to_numpy()

        stock = np.array(self.stock_history)

        matrix = np.stack([high, low, open, close, stock, volume])
        return matrix

    def reset(self):
        self.day = self.days.sample().iloc[0]["Date"]
        self.next_day = self.day + datetime.timedelta(days=1)
        self.data_day = self.data[(self.data.index >= pd.to_datetime(self.day)) & \
                                  (self.data.index < pd.to_datetime(self.next_day))]

        if len(self.data_day) <= self.config.nb_bars:
            raise Exception("Not enough data available. Be sure that your data contain intraday ticks.")

        self.offset = self.config.nb_bars
        self.account = self.config.initial_account
        self.stock = 0
        self.last_price = 0

        # State Reset
        self.stock_history = deque([0] * self.config.nb_bars, maxlen=self.config.nb_bars)
        self.account_history = deque([self.account] * self.config.nb_bars, maxlen=self.config.nb_bars)

        # Use for result report
        self.gain = deque(maxlen=100)
        self.relative_gain = deque(maxlen=100)

        return self.encode()

    def step(self, action):
        observation, reward, done, info  = None, 0, False, {}

        if action == Actions.Buy:
            price = self.data_day["Close"][self.offset]

            if price <= self.account and not self.stock:
                self.stock += 1
                self.account -= price

                self.last_price = price

        if action == Actions.Close:
            price = self.data_day["Close"][self.offset]

            if self.stock:
                self.stock -= 1
                self.account += price

                reward += ((price - self.last_price) / self.last_price)

                self.last_price = 0

        if action == Actions.Skip:
            pass

        self.stock_history.append(self.stock * self.last_price)

        # Next step
        self.account_history.append(self.account)
        self.offset += 1

        # Nest step state and reward
        observation = self.encode()
        done |= self.offset >= len(self.data_day)
        if not done:
            reward += ((self.data_day["Close"][self.offset] - self.last_price) / self.last_price) / 100 if self.last_price else 0
        info = {}

        return observation, reward, done, info
