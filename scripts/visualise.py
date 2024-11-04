import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from matplotlib import style

dark_mode = {
    "base_mpl_style": "dark_background",
    "marketcolors": {
        "candle": {"up": "#3dc985", "down": "#ef4f60"},
        "edge": {"up": "#3dc985", "down": "#ef4f60"},
        "wick": {"up": "#3dc985", "down": "#ef4f60"},
        "ohlc": {"up": "green", "down": "red"},
        "volume": {"up": "#247252", "down": "#82333f"},
        "vcedge": {"up": "green", "down": "red"},
        "vcdopcod": False,
        "alpha": 1,
    },
    "mavcolors": ("#ad7739", "#a63ab2", "#62b8ba"),
    "facecolor": "#1b1f24",
    "gridcolor": "#2c2e31",
    "gridstyle": "--",
    "y_on_right": True,
    "rc": {
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.edgecolor": "#474d56",
        "axes.titlecolor": "red",
        "figure.facecolor": "#161a1e",
        "figure.titlesize": "x-large",
        "figure.titleweight": "semibold",
    },
    "base_mpf_style": "binance-dark",
}


def stock_visualise(ticker, exchange):
    style.use('ggplot')

    # Read in CSV file, setting the dates as the index
    df = pd.read_csv(f'stock_dfs/{exchange}/{ticker}.csv')
    df.rename(columns={"Price": "Date"}, inplace=True)
    df.drop([0, 1], inplace=True)
    df.set_index("Date", inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    print(df.head())

    # Resample the dataset to achieve different timeframes ie 7Days = 1 week, 30S = 30 Seconds
    df_ohlcv = df.resample('1D').agg({'Open': 'first',
                                      'High': 'max',
                                      'Low': 'min',
                                      'Close': 'last',
                                      'Volume': 'sum'
                                      })

    # Alternatively using mpf Matplotlib Finance
    mpf.plot(df_ohlcv,
             type='candle',
             style='yahoo',
             volume=True,
             mav=(20, 50),
             warn_too_much_data=5000,
             figscale=2,
             scale_padding=0.5)

    mpf.show()


def visualize_data():
    df = pd.read_csv('data/sp500_adj_close_joined.csv')
    df.set_index("Date", inplace=True)
    # Correlation on prices
    # df_corr = df.corr()
    # Correlation on returns
    df_corr = df.pct_change().corr()

    data = df_corr.values
    print(np.nanmean(data))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    heatmap = ax.pcolor(data, cmap=plt.cm.Spectral)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1, 1)

    plt.tight_layout()
    fig.set_size_inches(16, 12, forward=True)
    plt.show()


if __name__ == "__main__":
    stock_visualise('CL=F', 'nymex')
