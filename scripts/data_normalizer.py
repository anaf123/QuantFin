import datetime
from collections import Counter

import numpy as np
import pandas as pd
import pickle
import os

from statsmodels.tsa.stattools import coint


class DataNormalizer:

    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x)
        self.sd = np.std(x)
        normalized_x = (x - self.mu) / self.sd
        return normalized_x

    @staticmethod
    def buy_sell_hold(*args):
        cols = [c for c in args]
        # Ideally want buy/hold/sell spread to be even but unlikely as stock market trends up - bias towards 1
        # Percent change we are looking out for, maybe run loop to estimate for industry specific value?
        # or bias towards sell and buy -> don't want idle days for shorter-term trading
        requirement = 0.005
        # Buy = 1, Sell = -1, Hold = 0
        for col in cols:
            if col > requirement:
                return 1
            if col < -requirement:
                return -1
        return 0

    @staticmethod
    def combine_alpha_intraday(input_dir, output):
        files = sorted([x for x in os.listdir(input_dir) if x.endswith("csv")])

        main_df = pd.DataFrame()
        for f in files:
            df = pd.read_csv(f"{input_dir}/{f}", index_col="time")
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
            df.index = pd.to_datetime(df.index)

            if main_df.empty:
                main_df = df
            else:
                main_df = pd.concat([main_df, df])

        main_df = main_df.sort_index()
        main_df.to_csv(output)

    @staticmethod
    def compile_data(input_dir, output_path):
        """

        :param input_dir: where stock dfs are located
        :param output_path: where to output compiled csv
        :return: CSV file containing adjusted close values for df
        """
        with open("pickle/sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
        main_df = pd.DataFrame()

        # open each csv and append adj. close price to main_df
        for count, ticker in enumerate(tickers):
            df = pd.read_csv(f'{input_dir}/{ticker}.csv')

            # yfinance has strange format for csv download so need to change columns
            df.rename(columns={"Price": "Date"}, inplace=True)
            df.drop([0, 1], inplace=True)
            df.set_index("Date", inplace=True)

            df.rename(columns={'Adj Close': ticker}, inplace=True)
            df.drop(["Open", "High", "Low", "Close", "Volume"], axis='columns', inplace=True)

            # empty case adds index
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')

            if count % 10 == 0:
                print(count)

        main_df.to_csv(output_path)

    @staticmethod
    def add_data_to_main_df(ticker, input_csv, main_df_path):
        main_df = pd.read_csv(main_df_path, index_col=0, parse_dates=['Date'])
        df = pd.read_csv(input_csv, index_col=0, parse_dates=['Date'])
        df.rename(columns={'Adj Close': ticker}, inplace=True)
        df.drop(["Open", "High", "Low", "Close", "Volume"], axis='columns', inplace=True)

        main_df = main_df.join(df, how='outer')
        main_df.to_csv(main_df_path)

    @staticmethod
    def add_alpha_to_main_df(ticker, input_csv, main_df_path):
        main_df = pd.read_csv(main_df_path, index_col=0, parse_dates=['Date'])
        df = pd.read_csv(input_csv, index_col=0, parse_dates=['timestamp'])
        df.rename(columns={'value': ticker}, inplace=True)
        df.index.names = ['Date']
        df = df[df.index >= datetime.datetime(2014, 8, 1)]
        df.index = df.index.tz_localize('UTC')

        main_df = main_df.join(df, how='outer')
        main_df.to_csv(main_df_path)

    @staticmethod
    def clean_df(main_df_path):
        main_df = pd.read_csv(main_df_path, index_col=0, parse_dates=['Date'])
        main_df.replace('.', 0, inplace=True)
        main_df.to_csv(main_df_path)

        print(main_df.head())



    @staticmethod
    def find_cointegrated_stocks(ticker):
        df = pd.read_csv('data/sp500_adj_close_joined.csv')
        df.set_index("Date", inplace=True)
        df.fillna(0, inplace=True)
        df = df.replace([np.inf, -np.inf], np.nan)
        df.dropna(inplace=True)

        top_stocks = []
        for col in df:
            if col == ticker:
                continue

            coint_t, p_value, crit = coint(df[col], df[ticker])

            if p_value < 0.05:
                top_stocks.append(col)

        print(top_stocks)

    def normalize_data(self, ticker):
        df = pd.read_csv(f'../stock_dfs/sp500/{ticker}.csv')
        df.set_index("Date", inplace=True)
        # df.rename(columns={'Adj Close': 'Price'}, inplace=True)
        df.drop(["Open", "High", "Low", "Close", "Volume"], axis='columns', inplace=True)

        # add variables to dataset
        df['%Return'] = df.pct_change()
        df['LogReturns'] = np.log(df['Adj Close']) - np.log(df['Adj Close'].shift(1))
        df.dropna(inplace=True)

        df['Normalised'] = self.fit_transform(df['LogReturns'])

        df['Target'] = list(map(self.buy_sell_hold, df['LogReturns'].values))

        vals = df['Target'.format(ticker)].values
        str_vals = [str(i) for i in vals]
        print('Data spread:', Counter(str_vals))
        print(df.head())

        return df


if __name__ == "__main__":
    # DataNormalizer.compile_data("stock_dfs/sp500/", "data/sp500_adj_close.csv")
    # DataNormalizer.add_alpha_to_main_df("INT", "stock_dfs/alpha/INTEREST/daily/price.csv", "data/sp500_adj_close.csv")
    DataNormalizer.clean_df("data/sp500_adj_close.csv")