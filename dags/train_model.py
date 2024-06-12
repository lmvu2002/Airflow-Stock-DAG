import pandas as pd
import requests
import numpy as np
import json
import tensorflow as tf
import keras
from keras.models import Sequential, save_model
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from sklearn.preprocessing import MinMaxScaler


def read_stock_data(symbol):
    api_key = 'WALEOPNN49XAKRNZ'
    # Example API request URL to get daily stock data
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}'

    # Sending a GET request to the API
    response = requests.get(url)

    # Checking if the request was successful
    if response.status_code == 200:
        data = response.json()  # Parsing JSON response
        # Process and use the data as needed
        return data
    else:
        print("Failed to fetch data:", response.status_code)
        return -1


def show_table(symbol):
    # json_str = json.dumps(data, indent=4, sort_keys=True)
    data = read_stock_data(symbol)
    table = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
    table = table.sort_index(axis = 1, inplace = False)
    return table




def preprocessing(data):
    n_past = 10
    n_future = 1
    scaler = MinMaxScaler()
    scaler = scaler.fit(data)
    data_scaled = scaler.transform(data)
    x = []
    y = []
    for i in range(n_past, len(data_scaled) - n_future + 1):
        x.append(data_scaled[i - n_past:i, 0:data_scaled.shape[1]])
        y.append(data_scaled[i + n_future - 1:i +
                 n_future, 0:data_scaled.shape[1]])

    x, y = np.array(x), np.array(y)
    return x, y


def split(x, y):
    num_test_elements = int(len(x) * 0.2)
    x_test = x[-num_test_elements:]
    y_test = y[-num_test_elements:]

    # The rest is for training
    x_train = x[:-num_test_elements]
    y_train = y[:-num_test_elements]

    return x_train, x_test, y_train, y_test


def modelA(x_train, y_train):
    lstm = Sequential()
    lstm.add(LSTM(84, return_sequences=True,
             activation='tanh', input_shape=(10, 5)))
    lstm.add(Dropout(0.3))
    lstm.add(LSTM(42, activation='tanh', return_sequences=False,))
    lstm.add(Dropout(0.3))
    lstm.add(Dense(64, activation='relu', kernel_regularizer='l1'))
    lstm.add(BatchNormalization())
    lstm.add(Dense(5, activation='sigmoid', kernel_regularizer='l1'))

    opt = keras.optimizers.Adam(
        learning_rate=0.0001, clipvalue=0.8, clipnorm=0.8)
    lstm.compile(optimizer=opt,
                 loss='mse',
                 metrics=['accuracy'])

    history = lstm.fit(x_train, y_train,
                       batch_size=64,
                       epochs=5,
                       validation_split=0.2,
                       shuffle=True)
    return lstm


def modelB(x_train, y_train):
    lstm = Sequential()
    lstm.add(LSTM(84, return_sequences=True,
             activation='relu', input_shape=(10, 5)))
    lstm.add(Dropout(0.2))
    lstm.add(LSTM(42, activation='relu', return_sequences=False,))
    lstm.add(Dropout(0.2))
    lstm.add(Dense(64, activation='relu', kernel_regularizer='l1'))
    lstm.add(BatchNormalization())
    lstm.add(Dense(5, activation='sigmoid', kernel_regularizer='l1'))

    opt = keras.optimizers.Adam(
        learning_rate=0.0001, clipvalue=0.8, clipnorm=0.8)
    lstm.compile(optimizer=opt,
                 loss='mse',
                 metrics=['accuracy'])

    history = lstm.fit(x_train, y_train,
                       batch_size=64,
                       epochs=5,
                       validation_split=0.2,
                       shuffle=True)
    return lstm


def modelC(x_train, y_train):
    lstm = Sequential()
    lstm.add(LSTM(84, return_sequences=True,
             activation='tanh', input_shape=(10, 5)))
    lstm.add(Dropout(0.3))
    lstm.add(LSTM(42, activation='tanh', return_sequences=False,))
    lstm.add(Dropout(0.3))
    lstm.add(Dense(64, activation='tanh', kernel_regularizer='l1'))
    lstm.add(BatchNormalization())
    lstm.add(Dense(5, activation='sigmoid', kernel_regularizer='l1'))

    opt = keras.optimizers.Adam(
        learning_rate=0.0001)
    lstm.compile(optimizer=opt,
                 loss='mse',
                 metrics=['accuracy'])

    history = lstm.fit(x_train, y_train,
                       batch_size=64,
                       epochs=5,
                       validation_split=0.2,
                       shuffle=True)
    return lstm


def evaluate(model, x, y):
    print(model.evaluate(x, y))
    return model.evaluate(x, y)


def choose_best_model(models, x_test, y_test):
    best_model = 0
    best_score = -100
    for i in range(len(models)):
        loss = evaluate(models[i], x_test, y_test)[0]
        print("______________EVALUATING______________")
        accuracy = evaluate(models[i], x_test, y_test)[1]
        if (best_score > -2*loss + 2*accuracy):
            continue
        else:
            best_score = max(best_score, -2*loss + 2*accuracy)
            best_model = i
    return best_score


symbol = ' VIC121003'

data = show_table(symbol)

pre_data = preprocessing(data)

x_train, x_test, y_train, y_test = split(pre_data)

history = modelB(x_train, y_train)