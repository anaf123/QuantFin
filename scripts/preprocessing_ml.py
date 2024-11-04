import datetime
import pickle
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn import svm, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

# How many days we want to look forward
HM_DAYS = 1
LOOKBACK = 10


def process_data_for_labels(ticker):
    df = pd.read_csv('data/sp500_adj_close.csv', index_col=0, parse_dates=['Date'])
    df.fillna(0, inplace=True)
    df[df <= 0] = 1e-10
    df = np.log(df).diff().fillna(0)

    for i in range(1, LOOKBACK + 1):
        df[f'{ticker}_LAG_{i}d'] = df[ticker].shift(i)

    df[f'{ticker}_1d'] = df[ticker].shift(-1)

    # Last HM_DAYS will have missing values
    df.dropna(inplace=True)
    features = df.columns.values

    return features, df


def buy_sell_hold(*args):
    cols = [c for c in args]
    # Ideally want buy/hold/sell spread to be even but unlikely as stock market trends up - bias towards 1
    # Percent change we are looking out for, maybe run loop to estimate for industry specific value?
    requirement = 0.01
    # Buy = 1, Sell = -1, Hold = 0
    for col in cols:
        if col > requirement:
            return 2
        if col < -requirement:
            return 0
    return 1


def extract_feature_sets(ticker):
    # add in columns of forward data
    features, df = process_data_for_labels(ticker)
    df = df[df.index < datetime.datetime(2024, 10, 1, tzinfo=datetime.timezone.utc)]

    df['{}_target'.format(ticker)] = list(
        map(buy_sell_hold, *[df['{}_{}d'.format(ticker, i)] for i in range(1, HM_DAYS + 1)]))

    vals = df[f'{ticker}_target'].values
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[f for f in features if f != f'{ticker}_target']].copy()
    df_vals.drop(columns=[f'{ticker}_1d'], inplace=True)

    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df[f'{ticker}_target'].values

    return X, y, df


def run_ml(ticker):
    X, y, df = extract_feature_sets(ticker)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Simple single classifier
    # clf = neighbors.KNeighborsClassifier()

    clf = VotingClassifier([('lsvc', svm.NuSVC(nu=0.5)),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])
    clf.fit(X_train, y_train)

    with open("pickle/ensemble_clf.pkl", 'wb') as f:
        pickle.dump(clf, f)

    confidence = clf.score(X_test, y_test)
    print("Accuracy:", confidence)
    predictions = clf.predict(X_test)

    print("Predicted spread: ", Counter(predictions))

    return confidence


def test_ml(ticker):
    features, df = process_data_for_labels(ticker)
    df = df[df.index >= datetime.datetime(2024, 10, 1, tzinfo=datetime.timezone.utc)]
    df.drop(columns=[f'{ticker}_1d'], inplace=True)
    print(df.head())

    with open("pickle/ensemble_clf.pkl", 'rb') as f:
        clf = pickle.load(f)

    time_window = df.values
    print(time_window)

    preds = clf.predict(time_window)
    print(preds)


if __name__ == "__main__":
    run_ml("CL=F")
    test_ml("CL=F")
