#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import pathlib
import pandas as pd
import pandas_datareader as pddr

class StockExchangeDataManager:

    def __init__(self, data_dir_path):
        # Create the main directory
        self.data_dir_path = pathlib.Path(data_dir_path)
        self.data_dir_path = self.data_dir_path.expanduser()
        self.data_dir_path.mkdir(parents=True, exist_ok=True)

        # Constants
        self.INDEX = []
        self.HEADERS = ["Date", "High", "Low", "Open", "Close", "Volume", "Adj Close"]
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
        # delta_time = self._end_time - self._start_time
        delta_count = self._data.loc[self.MASK]["Date"].count()
        self._data_available = delta_count > 0

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
            self.__load_content__()

        self.__select_content__()

        return self._data_selected
