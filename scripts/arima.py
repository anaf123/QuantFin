import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson

N_LAGS = 10
FORECAST_STEPS = 20


def run():
    df = pd.read_csv("stock_dfs/nymex/CL=F.csv", parse_dates=['Date'], index_col='Date')
    df.index = df.index.tz_localize(None)
    df['LogReturns'] = np.log(df['Adj Close']) - np.log(df['Adj Close'].shift(1))
    df.drop(["Open", "High", "Low", "Close", "Volume", "Adj Close"], axis='columns', inplace=True)
    df.dropna(inplace=True)

    df_lag = df.copy()
    for i in range(1, N_LAGS + 1):
        df_lag[f'Lag_{i}'] = df['LogReturns'].shift(i)

    df_lag = df_lag[
        (df_lag.index > datetime.datetime(2014, 9, 1)) & (df_lag.index < datetime.datetime(2024, 10, 1))]
    df_lag.dropna(inplace=True)

    data = df_lag['LogReturns']
    order = auto_extract_params(data, df_lag)

    model = ARIMA(df_lag['LogReturns'],
                  exog=df_lag[[f'Lag_{i}' for i in range(1, N_LAGS + 1)]],
                  order=order)
    model_fit = model.fit()
    residuals = model_fit.resid

    # For checking auto-correlation
    dw_stat = durbin_watson(residuals)
    print("DW STAT: ", dw_stat)

    forecast_df = df_lag.iloc[-1:]
    forecast_df.index = pd.to_datetime(forecast_df.index)
    forecast_df = forecast_df.asfreq("D")

    for i in range(1, FORECAST_STEPS + 1):
        row = forecast_df.tail(1).copy()
        values = np.concatenate(([np.nan], row.values[0][:-1]))
        row.index = forecast_df.tail(1).index + datetime.timedelta(1)
        while pd.to_datetime(row.index.values[0]).weekday() > 4:
            row.index = row.index + datetime.timedelta(1)
        row.loc[row.index] = values

        forecast_df = pd.concat([forecast_df, row])
        lagged = forecast_df[[f'Lag_{i}' for i in range(1, N_LAGS + 1)]].iloc[-1:]

        forecast = model_fit.forecast(steps=1, exog=lagged)
        forecast_df.loc[row.index, "LogReturns"] = forecast.values[0]

    print(forecast_df)
    print(df[df.index >= datetime.datetime(2024, 10, 1)])

    # plt.plot(df_short[(df_short.index > datetime.datetime(2014, 9, 1)) & (df_short.index < datetime.datetime(2024, 10, 15))],
    #          label="blue")
    # plt.plot(forecast, label="Forecast", color="red")
    # plt.legend()
    # plt.show()
    #
    # print(forecast)


def run_retrain(N_DAYS=20):
    df = pd.read_csv("stock_dfs/nymex/CL=F.csv", parse_dates=['Date'], index_col='Date')
    df.index = df.index.tz_localize(None)
    df['LogReturns'] = np.log(df['Adj Close']) - np.log(df['Adj Close'].shift(1))
    df.drop(["Open", "High", "Low", "Close", "Volume", "Adj Close"], axis='columns', inplace=True)
    df.dropna(inplace=True)

    for i in range(1, N_LAGS + 1):
        df[f'Lag_{i}'] = df['LogReturns'].shift(i)

    df.dropna(inplace=True)

    end = datetime.datetime(2024, 10, 1)
    finish = end
    forecasts = []
    while end < finish + datetime.timedelta(N_DAYS):
        while end.weekday() > 4:
            end += datetime.timedelta(1)
        df_lag = df[
            (df.index > datetime.datetime(2014, 9, 1)) & (df.index < end)].copy()
        df_lag.dropna(inplace=True)

        data = df_lag['LogReturns']
        order = auto_extract_params(data, df_lag)

        model = ARIMA(df_lag['LogReturns'],
                      exog=df_lag[[f'Lag_{i}' for i in range(1, N_LAGS + 1)]],
                      order=order)
        model_fit = model.fit()

        forecast_df = df_lag.iloc[-1:]
        forecast = model_fit.forecast(1, exog=forecast_df[[f'Lag_{i}' for i in range(1, N_LAGS + 1)]].iloc[-1:])
        forecasts.append(forecast)
        end += datetime.timedelta(1)
    print(forecasts)


def auto_extract_params(data, df_lag):
    res = adfuller(data)
    if res[1] > 0.05:
        data_diff = data.diff().dropna()
        print(data_diff.head())
    else:
        print("data is stationary")

    _fit = auto_arima(df_lag['LogReturns'],
                      exogenous=df_lag[[f'Lag_{i}' for i in range(1, N_LAGS + 1)]],
                      seasonal=False,
                      n_jobs=6,
                      stepwise=False)

    # print(_fit.summary())
    # print(_fit.order)

    return _fit.order


if __name__ == "__main__":
    run_retrain(29)
