import logging
import yfinance as yf
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from services.db_utils import get_stock_data, save_model_and_scaler, load_model_from_db, store_stock_data


def retrain_model(stock_symbol):
    existing_data = get_stock_data(stock_symbol)
    last_date = existing_data.index[-1] if not existing_data.empty else '2020-01-01'
    new_data = yf.download(stock_symbol, start=last_date, end=datetime.now().strftime('%Y-%m-%d'), auto_adjust=False)

    if not new_data.empty:
        new_data.columns = new_data.columns.get_level_values(0)
        new_data = new_data[['Close']]
        new_data.columns = ['close']  # FIX: normalize column BEFORE storing
        existing_data = pd.concat([existing_data, new_data])
        store_stock_data(stock_symbol, new_data)

    if existing_data.empty:
        logging.warning(f"No data available for stock symbol: {stock_symbol}")
        return False

    data = existing_data[['close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    if len(scaled_data) < 395:
        logging.warning(f"Not enough data to train the model for stock symbol: {stock_symbol}")
        return False

    train_data = []
    target_data = []
    for i in range(365, len(scaled_data) - 30):
        train_data.append(scaled_data[i - 365:i, 0])
        target_data.append(scaled_data[i:i + 30, 0])
    train_data, target_data = np.array(train_data), np.array(target_data)

    train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], 1))

    model, _, last_updated = load_model_from_db(stock_symbol)
    if model is None or (datetime.now() - last_updated).days > 7:
        model = Sequential()
        model.add(LSTM(units=100, return_sequences=True, input_shape=(train_data.shape[1], 1)))
        model.add(Dropout(0.3))
        model.add(LSTM(units=100, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(units=100))
        model.add(Dropout(0.3))
        model.add(Dense(units=30))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
        history = model.fit(train_data, target_data, epochs=10, batch_size=32, callbacks=[early_stopping], verbose=0)

        final_loss = float(history.history['loss'][-1]) if 'loss' in history.history else None

        save_model_and_scaler(stock_symbol, model, scaler, final_loss)
        return True

    return False
