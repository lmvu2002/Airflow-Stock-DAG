import requests
import finnhub
import json
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential, save_model
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from train_model import evaluate, show_table, preprocessing, split, modelB
# def show_table(symbol):
#     data = read_stock_data(symbol)
#     json_str = json.dumps(data, indent=4, sort_keys=True)
    
#     # table = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
#     return json_str

# def read_stock_data(symbol):
#     # Setup client
#     finnhub_client = finnhub.Client(api_key="cm3o8u9r01qsvtcqvj70cm3o8u9r01qsvtcqvj7g")

# # Stock candles
#     res = finnhub_client.company_basic_financials('AAPL', 'all')
#     print(res)
#     print("----------------------------------------------------------------")
#     return res
    
# print(show_table('AAPL'))

symbol = ' VIC121003'

data = show_table(symbol)

pre_data = preprocessing(data)

x_train, x_test, y_train, y_test = split(pre_data)

history = modelB(x_train, y_train)
