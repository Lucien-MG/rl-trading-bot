import pandas as pd
import yfinance as yf


class SourceYahoo:

    def __init__(self, index, start_date, end_date):
        self.index = index
        self.start_date = start_date
        self.end_date = end_date

    def _download(self, start_date, end_date):
        return yf.download(
            tickers=self.index,
            start=start_date,
            end=end_date,

            # fetch data by interval (including intraday if period < 60 days)
            # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            interval="1m",

            # adjust all OHLC automatically
            auto_adjust=True,

            # download pre/post regular market hours data
            prepost=True,

            # use threads for mass downloading? (True/False/Integer)
            threads=False,

            # proxy URL scheme use when downloading
            proxy=None
        )

    def _format_data(self, dataframe):
        dataframe = dataframe.reset_index()
        dataframe.columns = dataframe.columns.str.lower()
        dataframe = dataframe.rename(columns={'datetime': 'date'})

        dataframe['date'] = pd.to_datetime(dataframe['date'])
        dataframe['date'] = dataframe['date'].dt.tz_localize(None)

        return dataframe

    def _delta_date(self, actual_date, end_date):
        return pd.Timedelta(7, "d") \
            if end_date - actual_date > pd.Timedelta(7, "d") \
            else end_date - actual_date

    def load(self):
        past_date = self.start_date + pd.Timedelta(1, "d")
        actual_date = past_date
        actual_date += self._delta_date(actual_date, self.end_date)

        data = self._download(past_date, actual_date)
        data = self._format_data(data)

        while actual_date < self.end_date:
            past_date = actual_date
            actual_date += self._delta_date(actual_date, self.end_date)

            data_downloaded = self._download(past_date, actual_date)
            data_downloaded = self._format_data(data_downloaded)

            data = pd.concat([data, data_downloaded], ignore_index=True)
            data.drop_duplicates(subset=['date'])

        return data


SourceClass = SourceYahoo
