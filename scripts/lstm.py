import pickle
import datetime
import numpy as np
import pandas as pd

from keras.src.layers import Dropout, Input, LSTM, Dense
from keras.src.models import Sequential

from sklearn.preprocessing import StandardScaler

from preprocessing_ml import process_data_for_labels, buy_sell_hold


def run_lstm(ticker):
    tickers, df = process_data_for_labels("CL=F")

    ticker_df = pd.read_csv("stock_dfs/nymex/CL=F.csv", parse_dates=['Date'], index_col=0)
    ticker_df = np.log(ticker_df).diff().fillna(0)

    other_df = df.join(ticker_df, how="outer").copy()
    other_df.drop(columns=['Close', 'Adj Close'], inplace=True)
    other_df = other_df.replace([np.inf, -np.inf], np.nan)
    other_df.dropna(inplace=True)
    with open("pickle/df_clean.pkl", 'wb') as f:
        pickle.dump(other_df, f)

    train_df = other_df[other_df.index < datetime.datetime(2024, 10, 1, tzinfo=datetime.timezone.utc)].copy()
    y = list(map(buy_sell_hold, *[train_df[f'{ticker}_1d']]))
    train_df.drop(columns=[f'{ticker}_1d'], inplace=True)

    scaler = StandardScaler()
    scaler = scaler.fit(train_df)
    scaled_data = scaler.transform(train_df)

    df_target = pd.DataFrame(scaled_data)
    df_target['Label'] = y
    scaled_data = np.array(df_target.values)

    trainingX = []
    trainingY = []

    future = 1
    lookback = 10

    for i in range(lookback, len(scaled_data) - future + 1):
        trainingX.append(scaled_data[i - lookback:i, 0:scaled_data.shape[1]-1])
        trainingY.append(scaled_data[i + future - 1:i + future, -1])

    trainingX, trainingY = np.array(trainingX), np.array(trainingY)
    print(train_df.columns.values)

    print(trainingX.shape)
    print(trainingY.shape)

    model = Sequential()
    model.add(Input((trainingX.shape[1], trainingX.shape[2])))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(16, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.summary()

    history = model.fit(trainingX, trainingY, epochs=45, batch_size=32, verbose=True)
    with open("pickle/lstm_clf.pkl", 'wb') as f:
        pickle.dump(model, f)

    return model


def test_ml(ticker, date_to_predict):
    with open("pickle/df_clean.pkl", 'rb') as f:
        df = pickle.load(f)

    df.drop(columns=[f'{ticker}_1d'], inplace=True)
    index = df.index.values

    scaler = StandardScaler()
    scaler = scaler.fit(df)
    scaled_data = scaler.transform(df)

    df_scaled = pd.DataFrame(scaled_data, index=index)

    i = 25
    while df_scaled.shape[0] > 10:
        df_scaled = df_scaled[(df_scaled.index > date_to_predict - datetime.timedelta(i)) & (
                    df_scaled.index <= date_to_predict)]
        i -= 1

    forecastX = np.array([df_scaled])

    with open("pickle/lstm_clf.pkl", 'rb') as f:
        clf = pickle.load(f)

    predictions = clf.predict(forecastX)
    print(predictions)


if __name__ == "__main__":
    # run_lstm("CL=F")

    start = datetime.datetime(2024, 10, 1)
    end = datetime.datetime(2024, 10, 29)
    while start < end:
        test_ml("CL=F", start)
        start += datetime.timedelta(1)

