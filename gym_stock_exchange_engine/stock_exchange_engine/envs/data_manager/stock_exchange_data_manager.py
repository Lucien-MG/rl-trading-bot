#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

import pathlib
import importlib
import pandas as pd

from pkgutil import iter_modules
import stock_exchange_engine.envs.data_manager.sources as sources

class StockExchangeDataManager:
    LOCAL = "local"
    HEADERS = ["date", "high", "low", "open", "close", "volume", "adj close"]

    SOURCES = {module_finder.name: module_finder for module_finder in iter_modules(sources.__path__)}

    def __init__(self, data_dir_path):
        # Local must exits for the data manager to work
        assert self.LOCAL in self.SOURCES.keys()

        # Get the path for data folder
        self.data_dir_path = pathlib.Path(data_dir_path)
        self.data_dir_path = self.data_dir_path.expanduser()

        # Create the main directory if it does not exist
        self.data_dir_path.mkdir(parents=True, exist_ok=True)

    def _import_module(self, module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path / (module_name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _mask(self, start_time, end_time):
        return lambda df: \
            (df["date"] > pd.to_datetime(start_time)) & \
            (df["date"] <= pd.to_datetime(end_time))

    def _save_content(self, content, file_path):
        content.to_csv(file_path, index=False)

    def _load_content(self, file_path):
        try:
            data = pd.read_csv(file_path)
        except (FileNotFoundError, pd.errors.EmptyDataError) as EmptyData:
            print("No file or data found. Creating empty frame.")
            data = pd.DataFrame(columns=self.HEADERS)

        data.columns = data.columns.str.lower()
        data.columns = data.columns.str.replace('[#,@,&,<,>]', '', regex=True)

        data["date"] = pd.to_datetime(data["date"], format='%Y-%m-%d')

        if 'time' in data.columns:
            data["time"] = pd.to_datetime(data["time"], format='%H%M%S')
            data['date'] = pd.to_datetime(data["date"].dt.date.astype(str) + ' ' + data["time"].dt.time.astype(str))
            data = data.drop(['time'], axis=1)

        data = data.rename(columns={"vol": "volume"})
        data.sort_values(by="date")

        return data

    def _is_content_available(self, content, start_date, end_date):
        delta_count = content.loc[self._mask(start_date, end_date)]["date"].count()
        return delta_count > 0

    def _download_content(self, index, source, start_time, end_time, interval="1d"):
        # Import the module to use for data fetching
        module_path = pathlib.Path(self.SOURCES[source].module_finder.path)
        source_module = self._import_module(source, module_path)

        # Create a new module
        source_class = source_module.SourceClass(index, start_time, end_time, interval)

        # Use loaded module to fetch data
        data_downloaded = source_class.load()
        data_downloaded.columns = data_downloaded.columns.str.replace('[#,@,&,<,>]', '', regex=True)

        return data_downloaded

    def _missing_content(self, original_content, download_content):
        return download_content[download_content["date"].isin(original_content["date"]) == False]

    def _merge_content(self, original_content, missing_content):
        content = pd.concat([original_content, missing_content], ignore_index=True)
        content.sort_values(by="date")

        return content

    def _select_content(self, content, start_date, end_date):
        return content.loc[self._mask(start_date, end_date)]

    def get_index(self, index, source, start_date, end_date, interval="1d"):
        if source not in self.SOURCES.keys():
            raise ValueError('Source: ' + source + ' does not exist. Available sources are:', self.SOURCES_NAME)

        start_time = pd.to_datetime(start_date).date()
        end_time = pd.to_datetime(end_date).date()

        file_name = index + ".csv"
        file_path = self.data_dir_path / file_name

        # Load the data that the manager can
        content = self._load_content(file_path)

        # Check and download content if necessary
        if source != self.LOCAL:
            print("\nFetching data from", source)
            downloaded_content = self._download_content(index, source, start_time, end_time, interval)
            missing_content = self._missing_content(content, downloaded_content)
            content = self._merge_content(content, missing_content)
            self._save_content(content, file_path)
            content = self._load_content(file_path)

        # Get the exact content asked
        content = self._select_content(content, start_date, end_time)

        # Reset index
        content = content.reset_index(drop=True)

        return content
