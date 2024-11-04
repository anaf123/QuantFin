import datetime
import datetime as dt
import json
import os
import pickle
from io import StringIO
from pathlib import Path

import bs4 as bs
import kagglehub
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

DAYS_PER_YEAR = 365.24


class DataScraper:
    API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

    @staticmethod
    def save_sp500_tickers():
        # Send HTTP request to acquire wikipedia SP500 list of companies
        resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        # Use web scraper to convert into text, usisng lxml parser
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        # find specific table containing ticker data
        table = soup.find('table', {'id': 'constituents'})
        tickers = []

        # Iterate through all rows except headers appending tickers in list
        for row in table.findAll('tr')[1:]:
            # Wikipedia appends newline to ticker so remove
            ticker = row.findAll('td')[0].text[:-1]
            # Yahoo Finance needs symbols with - instead of .
            ticker = ticker.replace('.', '-')
            tickers.append(ticker)

        # Serialise list of SP500
        with open("../pickle/sp500tickers.pickle", "wb") as f:
            pickle.dump(tickers, f)

        return tickers

    @staticmethod
    def save_asx200_tickers():
        # Send HTTP request list of companies
        resp = requests.get('https://en.wikipedia.org/wiki/S%26P/ASX_200')

        # Use web scraper to convert into text, usisng lxml parser
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        # find specific table containing ticker data
        table = soup.find('table', {'class': 'wikitable sortable'})

        # Iterate through all rows except headers appending tickers in list
        tickers = []
        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text[:-1]
            # # Yahoo Finance needs symbols with - instead of .
            ticker = ticker.replace('.', '-')
            tickers.append(ticker)

        # Serialise list of SP500
        with open("../pickle/asx200tickers.pickle", "wb") as f:
            pickle.dump(tickers, f)

        return tickers

    @staticmethod
    def updated_tickers_sp500():
        path = kagglehub.dataset_download("andrewmvd/sp-500-stocks")
        print(f"Dataset downloaded to {path}")

        with open("pickle/sp500tickers.pickle", "wb") as f:
            df = pd.read_csv(path + "/sp500_companies.csv")
            pickle.dump(df['Symbol'].values, f)

        return df["Symbol"].values

    @staticmethod
    def manual_tickers_import():
        return []

    def get_data_sp500(self,
                       output_path,
                       start,
                       end,
                       interval="1d",
                       reload=0):
        """
            Finds the latest tickers list from wikipedia, kaggle or manual input then uses Yahoo Finance API to import data
            from the previous X years.

            :param output_path: directory to save data
            :param reload: should tickers be refreshed and referred to from wikipedia, manual txt files, or stay put
            :param interval: what time period we want data 1m, 5m, 15m, ... 1h, 1d, 5d, 1wk, 1mo
            :param days: how many years of data we want to download
        """
        if reload == 3:
            tickers_sp500 = self.save_sp500_tickers()
        elif reload == 2:
            tickers_sp500 = self.updated_tickers_sp500()
        elif reload == 1:
            tickers_sp500 = self.manual_tickers_import()
        else:
            with open("pickle/sp500tickers.pickle", "rb") as s:
                tickers_sp500 = pickle.load(s)

        # TODO not downloading aussie data as exchange needs to be specified (default is NYSE)

        # for ticker in tickers_asx200:
        #     print(ticker)
        #     if not os.path.exists('../stock_dfs/asx200/{}.csv'.format(ticker)):
        #         stock_data = yf.download(ticker, start, end, threads=2)
        #         stock_data.to_csv('../stock_dfs/asx200/{}.csv'.format(ticker))
        #     else:
        #         print("Already have {}".format(ticker))
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        for ticker in tickers_sp500:
            if not os.path.exists(f'{output_path}/{ticker}.csv'):
                stock_data = yf.download(ticker, start, end, interval=interval)
                stock_data.to_csv(f'{output_path}/{ticker}.csv')
            else:
                print("Already have {}".format(ticker))

    @staticmethod
    # Can pull singular stock data when required
    def get_specific_stock_data(ticker, exchange, start, end=dt.datetime.now().date(), interval="1d"):
        stock_data = yf.download(ticker, start=start, end=end, interval=interval)
        stock_data.to_csv(f'stock_dfs/{exchange}/{ticker}.csv')

    def get_intraday_over_period(self,
                                 start: datetime.date,
                                 end: datetime.date,
                                 ticker: str,
                                 interval: str):

        months = set()
        while start < end:
            delta = datetime.timedelta(days=30)
            months.add(f"{start.year}-{start.month:02}")
            start += delta

        for month in months:
            self.alpha_vantage_intraday_data(ticker, month, interval)

    def alpha_vantage_intraday_data(self, ticker, month, interval):
        """

        :param ticker: Which symbol we want data for
        :param month: The month in YYYY-MM format
        :param interval: the intraday interval required
        :return: csv containing our requested data
        """
        assert interval in ["1min", "5min", "15min", "30min", "60min"]

        # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval={interval}&apikey={self.API_KEY}&adjusted=true&month={month}&outputsize=full&datatype=csv'
        r = requests.get(url)
        decoded = StringIO(r.content.decode('utf-8'))
        df = pd.read_csv(decoded)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df.set_index("timestamp")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        if not os.path.exists(f"stock_dfs/alpha/{ticker}/{interval}/price"):
            os.mkdir(f"stock_dfs/alpha/{ticker}/{interval}/price")
        df.to_csv(f"stock_dfs/alpha/{ticker}/{interval}/price/{month}.csv")

    def alpha_vantage_adj_interday_data(self, interval, ticker):
        """

        :param interval: one of MONTHLY, WEEKLY, DAILY data
        :param ticker: Which symbol we want data for
        :return: csv containing our requested data
        """

        assert interval in ['MONTHLY', 'WEEKLY', 'DAILY']

        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_{interval}_ADJUSTED&symbol={ticker}&outputsize=full&apikey={self.API_KEY}&datatype=csv'

        r = requests.get(url)
        decoded = StringIO(r.content.decode('utf-8'))
        df = pd.read_csv(decoded)
        if not os.path.exists(f"stock_dfs/alpha/{ticker}/{interval.lower()}"):
            os.mkdir(f"stock_dfs/alpha/{ticker}/{interval.lower()}/")

        df.set_index('timestamp', inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        df.to_csv(f'stock_dfs/alpha/{ticker}/{interval.lower()}/price.csv')

    def alpha_search_ticker(self, key_words):
        # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
        url = f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={key_words}&apikey={self.API_KEY}'
        print(url)
        r = requests.get(url)
        j = json.loads(r.text)
        j_formatted = json.dumps(j, indent=2)
        print(j_formatted)

    def alpha_url_data(self, function, output, **kwargs):

        url = f'https://www.alphavantage.co/query?function={function}&apikey={self.API_KEY}&datatype=csv'

        path = Path(output)
        if not path.parent.exists():
            os.mkdir(path.parent)

        for k, v in kwargs.items():
            string = f"&{k}={v}"
            url += string

        print(url)

        r = requests.get(url)
        decoded = StringIO(r.content.decode('utf-8'))
        df = pd.read_csv(decoded)
        df.set_index("time", inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df.to_csv(output)

    def re_download_intraday_over_period(self,
                                         start: datetime.date,
                                         end: datetime.date,
                                         ticker: str,
                                         interval: str):
        """
        Need to redownload data as api has limits per minute

        :params : as alpha_vantage_intraday_data
        """
        months = set()
        while start < end:
            delta = datetime.timedelta(days=28)
            months.add(f"{start.year}-{start.month:02}")
            start += delta

        for month in months:
            f = pd.read_csv(f'stock_dfs/alpha/{ticker}/{interval}/price/{month}.csv')
            if 'timestamp' not in f.columns.values:
                self.alpha_vantage_intraday_data(ticker, month, interval)
            else:
                print(f"Already have {ticker}/{interval}/price/{month}.csv")


if __name__ == "__main__":
    ds = DataScraper()
    start = datetime.date(2014, 8, 1)
    end = datetime.datetime.now().date()

    # ds.get_specific_stock_data("PJM", "nymex", start, end)
    ds.alpha_vantage_intraday_data("CL=F", start, "5min")

    months = set()
    while start < end:
        delta = datetime.timedelta(days=28)
        months.add(f"{start.year}-{start.month:02}")
        start += delta

    folder = 'XOM'
    interval = 'daily'
    function_string = "RSI"

    output_string = f'stock_dfs/alpha/{folder}/{interval}/{function_string.lower()}_14.csv'
    # ds.alpha_url_data(
    #     function=function_string,
    #     output=output_string,
    #     interval=interval,
    #     symbol="XOM",
    #     series_type="close",
    #     time_period="14"
    # )
