from flask import request, jsonify
from flask_login import login_user, login_required, logout_user
from werkzeug.security import generate_password_hash
from user_db import get_user_by_id, get_user_by_username, create_user
import logging
import yfinance as yf
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from retrain_model import retrain_model, load_model_from_db, save_model_to_db
from db_utils import get_stock_data, store_stock_data, store_predictions

def register_routes(app, login_manager):

    @login_manager.user_loader
    def load_user(user_id):
        return get_user_by_id(user_id)

    @app.route('/api/register', methods=['POST'])
    def register():
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        if not username or not password:
            return jsonify({'error': 'Username and password are required'}), 400

        password_hash = generate_password_hash(password)
        try:
            create_user(username, password_hash)
            return jsonify({'message': 'User registered successfully'}), 201
        except Exception as e:
            return jsonify({'error': 'An error occurred during registration'}), 500

    @app.route('/api/login', methods=['POST'])
    def login():
        try:
            data = request.get_json()
            username = data['username']
            password = data['password']
            user = get_user_by_username(username)
            if user and user.check_password(password):
                login_user(user)
                return jsonify({'message': 'Login successful'}), 200
            return jsonify({'message': 'Invalid username or password'}), 401
        except Exception as e:
            logging.error(f"Error during login: {e}")
            return jsonify({'error': 'An error occurred during login'}), 500

    @app.route('/api/logout', methods=['POST'])
    @login_required
    def logout():
        logout_user()
        return jsonify({'message': 'Logged out successfully'}), 200

    @app.route('/api/stock', methods=['POST'])
    def stock():
        if request.method == 'OPTIONS':
            return '', 200
        try:
            stock_symbol = request.json.get('stock_symbol', 'GOOGL')
            logging.debug(f"Received stock symbol: {stock_symbol}")
            retrain_model(stock_symbol)

            existing_data = get_stock_data(stock_symbol)
            logging.debug(f"Existing data: {existing_data}")
            last_date = existing_data.index[-1] if not existing_data.empty else '2020-01-01'
            new_data = yf.download(stock_symbol, start=last_date, end=datetime.now().strftime('%Y-%m-%d'),
                                   auto_adjust=False)
            logging.debug(f"New data: {new_data}")
            if not new_data.empty:
                new_data = new_data[['Close']]
                store_stock_data(stock_symbol, new_data)
                existing_data = get_stock_data(stock_symbol)

            data = existing_data[['Close']]

            # Handle missing values
            data = data.dropna()

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            model, _ = load_model_from_db(stock_symbol)

            last_60_days = scaled_data[-60:]
            last_60_days = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))
            predicted_prices = model.predict(last_60_days)
            predicted_prices = scaler.inverse_transform(predicted_prices)

            future_dates = pd.date_range(start=data.index[-1], periods=31, inclusive='right')
            predicted_df = pd.DataFrame(predicted_prices[0], index=future_dates, columns=['Predicted Close'])
            store_predictions(stock_symbol, predicted_df)

            actual_prices = data['Close'].tolist()
            actual_dates = data.index.to_list()
            actual_dates = [date.strftime('%Y-%m-%d') for date in actual_dates]
            predicted_prices = predicted_df['Predicted Close'].tolist()
            predicted_dates = predicted_df.index.to_list()
            predicted_dates = [date.strftime('%Y-%m-%d') for date in predicted_dates]

            return jsonify({
                'stock_symbol': stock_symbol,
                'actual_prices': actual_prices,
                'actual_dates': actual_dates,
                'predicted_prices': predicted_prices,
                'predicted_dates': predicted_dates
            })
        except Exception as e:
            logging.error(f"Error in /stock endpoint: {e}", exc_info=True)
            return jsonify({'error': 'Internal Server Error'}), 500