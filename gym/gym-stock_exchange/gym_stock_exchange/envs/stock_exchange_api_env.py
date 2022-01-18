#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import gym
import enum
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pddr

class StockExchangeDataManager:

    def __init__(self, data_dir_path):
        # Create the main directory
        self.data_dir_path = pathlib.Path(data_dir_path)
        self.data_dir_path.mkdir(parents=True, exist_ok=True)

        # Constants
        self.INDEX = []
        self.HEADERS = ["Date", "High", "Low", "Open", "Close", "Volume", "Adj Close"]
        self.AVERAGE_TRADING_DAY = 0.68
        self.MASK = lambda df : (df["Date"] > pd.to_datetime(self._start_time)) & \
                                (df["Date"] <= pd.to_datetime(self._end_time))

        # Intern variables
        self._index = None
        self._source = None
        self._start_time = None
        self._end_time = None

        self._file_name = None
        self._file_path = None

        self._data = None
        self._data_selected = None
        self._data_downloaded = None
        self._data_available = False

    def __file_infos_update__(self):
        self._file_name = self._index + ".csv"
        self._file_path = self.data_dir_path / self._file_name

    def __load_content__(self):
        try:
            self._data = pd.read_csv(self._file_path)
            self._data["Date"] = pd.to_datetime(self._data["Date"])
            self._data.sort_values(by="Date")
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            self._data_available = False
            self._data = pd.DataFrame(index=self.INDEX, columns=self.HEADERS)

    def __is_content_available__(self):
        delta_time = self._end_time - self._start_time
        delta_count = self._data.loc[self.MASK]["Date"].count()
        self._data_available = delta_count / delta_time.days >= self.AVERAGE_TRADING_DAY

    def __download_content__(self):
        self._data_downloaded = pddr.data.DataReader(self._index, self._source, self._start_time, self._end_time)

        self._data_downloaded = self._data_downloaded.reset_index()
        self._data_downloaded["Date"] = pd.to_datetime(self._data_downloaded["Date"])

    def __merge_content__(self):
        new_data = self._data_downloaded[self._data_downloaded["Date"].isin(self._data["Date"]) == False]
        self._data = pd.concat([self._data, new_data], ignore_index=True)
        self._data.sort_values(by="Date")

    def __save_content__(self):
        self._data.to_csv(self._file_path, index=False)

    def __select_content__(self):
        self._data_selected = self._data.loc[self.MASK]

    def get_index(self, index, source, start_date, end_date):
        self._index = index
        self._source = source
        self._start_time = pd.to_datetime(start_date).date()
        self._end_time = pd.to_datetime(end_date).date()

        # Init all our information
        self.__file_infos_update__()

        # Load the data that the manager can
        self.__load_content__()

        # Check the data availability:
        self.__is_content_available__()

        # Check and download content if necessary
        if not self._data_available:
            print("Download")
            self.__download_content__()
            self.__merge_content__()
            self.__save_content__()

        self.__select_content__()

        return self._data_selected

class Actions(enum.Enum):
    """ Possible action for stock exchange bot. """
    Skip = 0
    Buy = 1
    Close = 2

class StockExchangeState:

    def __init__(self, data, config):
        self.data = data
        self.config = config

    def encode(self):
        high = self.data['High'][self.offset - self.config.nb_bars: self.offset].to_numpy()
        low = self.data['Low'][self.offset - self.config.nb_bars: self.offset].to_numpy()
        open = self.data['Open'][self.offset - self.config.nb_bars: self.offset].to_numpy()
        close = self.data['Close'][self.offset - self.config.nb_bars: self.offset].to_numpy()
        volume = self.data['Volume'][self.offset - self.config.nb_bars: self.offset].to_numpy()
        stock = np.full(self.config.nb_bars, self.stock)

        matrix = np.stack([high, low, open, close, volume, stock])
        return matrix

    def reset(self):
        self.offset = self.config.nb_bars
        self.account = self.config.initial_account
        self.stock = 0

        return self.encode()

    def step(self, action):
        observation, reward, done, info  = None, 0.0, False, {}

        if action == Actions.Buy and not self.stock:
            self.stock += 1
            self.account -= self.data["High"][self.offset]
            reward -= self.config.commission

        if action == Actions.Close and self.stock:
            self.stock -= 1
            self.account += self.data["Low"][self.offset]
            reward -= self.config.commission

        self.offset += 1

        observation = self.encode()
        reward += ((self.account - self.config.initial_account) * 100) / self.config.initial_account
        done = self.offset >= len(self.data)

        return observation, reward, done, info


class StockExchangeApiEnv(gym.Env):
    metadata = { 'render.modes': ['human'] }

    def __init__(self, config):
        super(StockExchangeApiEnv, self).__init__()

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
        # self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32)

    def reset(self):
        return self.state.reset()
    
    def step(self, action_index):
        action = Actions(action_index)
        observation, reward, done, info = self.state.step(action)
        return observation, reward, done, info

    def render(self, mode='human'):
        self.data['Close'].plot(label='Stocks At Closing', figsize=(16,5))
        self.data['Open'].plot(label='Stock Prices At Open', figsize=(16,5))
        self.data['High'].plot(label='Stock Prices On A High', figsize=(16,5))
        self.data['Low'].plot(label='Stock Prices On A Low', figsize=(16,5))

        plt.title(self.config.index + ' stock prices')
        plt.xlabel('Date')
        plt.ylabel('Stock Prices')
        plt.legend()

        plt.show()


    
