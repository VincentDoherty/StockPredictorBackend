import logging
import yfinance as yf
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import tensorflow as tf
from db_utils import get_stock_data, get_db_connection, save_model_to_db, load_model_from_db


def retrain_model(stock_symbol):
    existing_data = get_stock_data(stock_symbol)
    logging.debug(f"Existing data columns: {existing_data.columns}")
    last_date = existing_data.index[-1] if not existing_data.empty else '2020-01-01'
    new_data = yf.download(stock_symbol, start=last_date, end=datetime.now().strftime('%Y-%m-%d'), auto_adjust=False)
    logging.debug(f"New data columns: {new_data.columns}")

    if not new_data.empty:
        new_data.columns = new_data.columns.get_level_values(0)  # Flatten the MultiIndex
        new_data = new_data[['Close']]
        existing_data = pd.concat([existing_data, new_data])

    if existing_data.empty:
        return

    logging.debug(f"Final existing data columns: {existing_data.columns}")
    data = existing_data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    train_data = []
    target_data = []
    for i in range(365, len(scaled_data) - 30):
        train_data.append(scaled_data[i - 365:i, 0])
        target_data.append(scaled_data[i:i + 30, 0])
    train_data, target_data = np.array(train_data), np.array(target_data)

    train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], 1))

    model, last_updated = load_model_from_db(stock_symbol)
    if model is None or (datetime.now() - last_updated).days > 7:
        model = Sequential()
        model.add(LSTM(units=100, return_sequences=True, input_shape=(train_data.shape[1], 1)))
        model.add(Dropout(0.3))
        model.add(LSTM(units=100, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(units=100))
        model.add(Dropout(0.3))
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
        model.fit(train_data, target_data, epochs=10, batch_size=32, callbacks=[early_stopping])

        save_model_to_db(stock_symbol, model)


