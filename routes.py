from flask import request, jsonify
from flask_login import login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash
from user_db import get_user_by_id, get_user_by_username, create_user
import logging
import yfinance as yf
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
from retrain_model import retrain_model, load_model_from_db
from services.db_utils import get_stock_data, store_predictions, store_stock_data, get_db_connection


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


def stock_routes(app):
    @app.route('/api/stock', methods=['POST'])
    @login_required
    def stock():
        try:
            logging.debug(f"Request payload: {request.json}")
            stock_symbol = request.json.get('stock_symbol')
            if not stock_symbol:
                return jsonify({'error': 'Stock symbol is required'}), 400

            logging.debug(f"Received stock symbol: {stock_symbol}")
            new_data = yf.download(stock_symbol, start='2020-01-01', end=datetime.now().strftime('%Y-%m-%d'),
                                   auto_adjust=False)
            if new_data.empty:
                return jsonify({'error': 'No data available for the given stock symbol'}), 400

            if 'Close' in new_data.columns:
                new_data = new_data[['Close']].dropna()
                new_data.columns = ['close']
            else:
                return jsonify({'error': "'Close' column not found in the stock data"}), 400

            store_stock_data(stock_symbol, new_data)
            retrain_model(stock_symbol)

            existing_data = get_stock_data(stock_symbol)
            if existing_data.empty:
                return jsonify({'error': 'No data available for the given stock symbol'}), 400

            if 'close' not in existing_data.columns:
                return jsonify({'error': "'close' column not found in the stock data"}), 500

            data = existing_data[['close']].dropna()
            if data.empty:
                return jsonify({'error': 'Insufficient data for predictions'}), 400

            model, scaler, last_updated = load_model_from_db(stock_symbol)
            if model is None or scaler is None:
                return jsonify({'error': 'Model or scaler not found'}), 404

            scaled_data = scaler.transform(data)
            last_60_days = scaled_data[-60:]
            last_60_days = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))

            predicted_prices = model.predict(last_60_days)
            predicted_prices = scaler.inverse_transform(predicted_prices)

            future_dates = pd.date_range(start=data.index[-1], periods=31, inclusive='right')
            predicted_df = pd.DataFrame(predicted_prices[0], index=future_dates, columns=['Predicted Close'])
            store_predictions(stock_symbol, predicted_df)

            response = {
                'stock_symbol': stock_symbol,
                'actual_prices': data['close'].tolist(),
                'actual_dates': [date.strftime('%Y-%m-%d') for date in data.index],
                'predicted_prices': predicted_df['Predicted Close'].tolist(),
                'predicted_dates': [date.strftime('%Y-%m-%d') for date in predicted_df.index]
            }
            return jsonify(response), 200


        except Exception as e:

            logging.error(f"Error in /api/stock endpoint: {e}", exc_info=True)

            if app.config.get("TESTING", False):  # 👈 this allows test to see the traceback

                return jsonify({

                    'error': 'Internal Server Error',

                    'details': str(e)

                }), 500

            return jsonify({'error': 'Internal Server Error'}), 500


    @app.route('/api/stock/<string:symbol>/history', methods=['GET'])
    @login_required
    def get_stock_history(symbol):
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT date, close
                    FROM stock_data
                    WHERE stock_symbol = %s
                    ORDER BY date ASC
                    """,
                    (symbol.upper(),)
                )
                rows = cursor.fetchall()

            history = [
                {"date": row[0].strftime('%Y-%m-%d'), "close": round(row[1], 2)}
                for row in rows
            ]

            return jsonify({
                "symbol": symbol.upper(),
                "history": history
            }), 200

        except Exception as e:
            logging.error(f"Error fetching stock history for {symbol}: {e}", exc_info=True)
            return jsonify({'error': 'Failed to fetch stock history'}), 500
        finally:
            conn.close()

    @app.route('/api/stock/<string:symbol>/predictions', methods=['GET'])
    @login_required
    def get_stock_predictions(symbol):
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT date, predicted_close
                    FROM stock_predictions
                    WHERE stock_symbol = %s
                    ORDER BY date ASC
                    """,
                    (symbol.upper(),)
                )
                rows = cursor.fetchall()

            if not rows:
                return jsonify({'message': 'No predictions available'}), 404

            predictions = [
                {"date": row[0].strftime('%Y-%m-%d'), "price": round(row[1], 2)}
                for row in rows
            ]

            return jsonify({
                "symbol": symbol.upper(),
                "predicted": predictions
            }), 200

        except Exception as e:
            logging.error(f"Error fetching predictions for {symbol}: {e}", exc_info=True)
            return jsonify({'error': 'Failed to fetch predictions'}), 500
        finally:
            conn.close()

    @app.route('/api/stock/<string:symbol>/refresh', methods=['POST'])
    @login_required
    def refresh_stock_data(symbol):
        try:
            conn = get_db_connection()
            latest_date = None

            # Step 1: Find most recent date in DB
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT MAX(date) FROM stock_data WHERE stock_symbol = %s",
                    (symbol.upper(),)
                )
                result = cursor.fetchone()
                latest_date = result[0] if result and result[0] else datetime(2020, 1, 1)

            # Step 2: Download new data from yfinance
            start = latest_date.strftime('%Y-%m-%d')
            end = datetime.now().strftime('%Y-%m-%d')

            new_data = yf.download(symbol, start=start, end=end, auto_adjust=False)

            if new_data.empty or 'Close' not in new_data.columns:
                return jsonify({'error': 'No new data found for symbol'}), 400

            clean_data = new_data[['Close']].dropna()
            clean_data.columns = ['close']

            # Step 3: Store new data
            store_stock_data(symbol, clean_data)

            return jsonify({'message': f'Data for {symbol.upper()} refreshed successfully'}), 200

        except Exception as e:
            logging.error(f"Error in /api/stock/{symbol}/refresh: {e}", exc_info=True)
            return jsonify({'error': 'Failed to refresh stock data'}), 500
        finally:
            conn.close()

    @app.route('/api/portfolio/<int:portfolio_id>/metrics', methods=['GET'])
    @login_required
    def get_portfolio_metrics(portfolio_id):
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT stock_symbol, shares, purchase_date, purchase_price
                    FROM portfolio_stocks
                    WHERE portfolio_id = %s
                    """,
                    (portfolio_id,)
                )
                stocks = cursor.fetchall()
                if not stocks:
                    return jsonify({'error': 'Portfolio is empty'}), 404

                price_map = {}
                for symbol, shares, purchase_date, _ in stocks:
                    cursor.execute(
                        """
                        SELECT date, close
                        FROM stock_data
                        WHERE stock_symbol = %s
                          AND date >= %s
                        ORDER BY date ASC
                        """,
                        (symbol, purchase_date)
                    )
                    for date, close in cursor.fetchall():
                        if date not in price_map:
                            price_map[date] = 0
                        price_map[date] += close * shares

            # Build time series
            sorted_dates = sorted(price_map)
            values = [price_map[date] for date in sorted_dates]

            # Compute daily returns
            returns = pd.Series(values).pct_change().dropna()
            if returns.empty:
                return jsonify({'error': 'Not enough data for metrics'}), 400

            avg_daily_return = returns.mean()
            volatility = returns.std()
            sharpe_ratio = (avg_daily_return / volatility) * np.sqrt(252)
            total_return = (values[-1] - values[0]) / values[0]

            metrics = {
                'total_return': round(total_return * 100, 2),
                'avg_daily_return': round(avg_daily_return * 100, 4),
                'volatility': round(volatility * 100, 4),
                'sharpe_ratio': round(sharpe_ratio, 2)
            }

            return jsonify(metrics), 200

        except Exception as e:
            logging.error(f"Error computing metrics: {e}", exc_info=True)
            return jsonify({'error': 'Failed to compute metrics'}), 500
        finally:
            conn.close()


    @app.route('/api/portfolios/<int:portfolio_id>/rebalance', methods=['POST'])
    @login_required
    def rebalance_portfolio(portfolio_id):
        conn = get_db_connection()
        try:
            # 1. Load all holdings
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT stock_symbol, shares, purchase_date FROM portfolio_stocks WHERE portfolio_id = %s",
                    (portfolio_id,)
                )
                rows = cursor.fetchall()

            if not rows:
                return jsonify({'error': 'Portfolio is empty'}), 404

            symbols = [row[0] for row in rows]
            shares = {row[0]: row[1] for row in rows}
            start_date = min(row[2] for row in rows)

            # 2. Fetch price history
            price_map = {}
            with conn.cursor() as cursor:
                for sym in symbols:
                    cursor.execute(
                        """
                        SELECT date, close
                        FROM stock_data
                        WHERE stock_symbol = %s
                          AND date >= %s
                        ORDER BY date ASC
                        """,
                        (sym, start_date)
                    )
                    df = pd.DataFrame(cursor.fetchall(), columns=['date', 'price'])
                    df.set_index('date', inplace=True)
                    price_map[sym] = df

            df_prices = pd.concat(price_map.values(), axis=1)
            df_prices.columns = symbols
            df_prices = df_prices.dropna()

            if df_prices.empty:
                return jsonify({'error': 'Insufficient data for MPT'}), 400

            returns = df_prices.pct_change().dropna()

            # 3. Optimization: Maximize Sharpe Ratio
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            num_assets = len(symbols)
            bounds = tuple((0, 1) for _ in range(num_assets))
            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

            def neg_sharpe(weights):
                port_return = np.dot(weights, mean_returns)
                port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return -port_return / port_vol

            initial_weights = [1.0 / num_assets] * num_assets
            result = minimize(neg_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

            if not result.success:
                return jsonify({'error': 'Optimization failed'}), 500

            optimal_weights = result.x
            allocation = [
                {"symbol": symbols[i], "weight": round(optimal_weights[i] * 100, 2)}
                for i in range(num_assets)
            ]

            return jsonify({
                "message": "Optimal allocation computed using MPT",
                "allocation": allocation
            }), 200

        except Exception as e:
            logging.error(f"Error computing MPT rebalance: {e}", exc_info=True)
            return jsonify({'error': 'Failed to compute optimal portfolio'}), 500
        finally:
            conn.close()


def portfolio_routes(app):
    @app.route('/api/portfolios', methods=['GET', 'POST'])
    @login_required
    def portfolios():
        if request.method == 'GET':
            conn = get_db_connection()
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT id, name FROM portfolios WHERE user_id = %s", (current_user.id,))
                    portfolios = cursor.fetchall()
                    response = [{'id': row[0], 'name': row[1]} for row in portfolios]
                    return jsonify(response), 200
            except Exception as e:
                logging.error(f"Error fetching portfolios: {e}", exc_info=True)
                return jsonify({'error': 'Failed to fetch portfolios'}), 500
            finally:
                conn.close()

        elif request.method == 'POST':
            data = request.get_json()
            name = data.get('name')
            if not name:
                return jsonify({'error': 'Portfolio name is required'}), 400
            conn = get_db_connection()
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO portfolios (user_id, name) VALUES (%s, %s) RETURNING id",
                        (current_user.id, name)
                    )
                    portfolio_id = cursor.fetchone()[0]
                conn.commit()
                return jsonify({'message': 'Portfolio created successfully', 'portfolio_id': portfolio_id}), 201
            except Exception as e:
                logging.error(f"Error creating portfolio: {e}", exc_info=True)
                return jsonify({'error': 'Failed to create portfolio'}), 500
            finally:
                conn.close()
        return jsonify({'error': 'Invalid request method'}), 405

    @app.route('/api/portfolios/<int:portfolio_id>/stocks', methods=['POST'])
    @login_required
    def add_stock_to_portfolio(portfolio_id):
        data = request.get_json()
        stock_symbol = data.get('stock_symbol')
        purchase_price = data.get('purchase_price')
        purchase_date = data.get('purchase_date')
        shares = data.get('shares', 1.0)

        if not all([stock_symbol, purchase_price, purchase_date]) or shares <= 0:
            return jsonify({'error': 'Stock symbol, price, date, and shares > 0 required'}), 400

        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO portfolio_stocks (portfolio_id, stock_symbol, purchase_price, purchase_date, shares)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (portfolio_id, stock_symbol, purchase_price, purchase_date, shares)
                )
            conn.commit()
            return jsonify({'message': 'Stock added to portfolio successfully'}), 201
        except Exception as e:
            logging.error(f"Error adding stock to portfolio: {e}", exc_info=True)
            return jsonify({'error': 'Failed to add stock to portfolio'}), 500
        finally:
            conn.close()

    @app.route('/api/portfolios/<int:portfolio_id>', methods=['GET'])
    @login_required
    def get_portfolio(portfolio_id):
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT ps.id, ps.stock_symbol, ps.purchase_price, sd.close
                    FROM portfolio_stocks ps
                             LEFT JOIN (SELECT stock_symbol, close
                                        FROM stock_data
                                        WHERE date = (SELECT MAX(date)
                                                      FROM stock_data
                                                      WHERE stock_symbol = stock_data.stock_symbol)) sd
                                       ON ps.stock_symbol = sd.stock_symbol
                    WHERE ps.portfolio_id = %s
                    """,
                    (portfolio_id,)
                )
                stocks = cursor.fetchall()

            if not stocks:
                return jsonify([]), 200

            portfolio = []
            for stock in stocks:
                stock_id, stock_symbol, purchase_price, current_price = stock
                if current_price is None:
                    current_price = purchase_price
                profit_loss = current_price - purchase_price
                portfolio.append({
                    'id': stock_id,
                    'stock_symbol': stock_symbol,
                    'purchase_price': f"{purchase_price:.2f}",
                    'current_price': f"{current_price:.2f}",
                    'profit_loss': f"{profit_loss:.2f}"
                })

            sort_by = request.args.get('sort_by', 'profit_loss')
            reverse = request.args.get('order', 'desc') == 'desc'
            if sort_by not in ['stock_symbol', 'purchase_price', 'current_price', 'profit_loss']:
                return jsonify({'error': f'Invalid sort_by value: {sort_by}'}), 400
            portfolio = sorted(portfolio, key=lambda x: x[sort_by], reverse=reverse)

            return jsonify(portfolio), 200
        except Exception as e:
            logging.error(f"Error fetching portfolio: {e}", exc_info=True)
            return jsonify({'error': 'Failed to fetch portfolio'}), 500
        finally:
            conn.close()

    @app.route('/api/portfolios/<int:portfolio_id>', methods=['DELETE'])
    @login_required
    def delete_portfolio(portfolio_id):
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                # Confirm the portfolio belongs to the current user
                cursor.execute(
                    "SELECT id FROM portfolios WHERE id = %s AND user_id = %s",
                    (portfolio_id, current_user.id)
                )
                result = cursor.fetchone()
                if not result:
                    return jsonify({'error': 'Portfolio not found or unauthorized'}), 404

                # Delete the portfolio
                cursor.execute("DELETE FROM portfolios WHERE id = %s", (portfolio_id,))
            conn.commit()
            return jsonify({'message': 'Portfolio deleted successfully'}), 200
        except Exception as e:
            logging.error(f"Error deleting portfolio: {e}", exc_info=True)
            return jsonify({'error': 'Failed to delete portfolio'}), 500
        finally:
            conn.close()

    @app.route('/api/portfolios/<int:portfolio_id>/stocks/<int:stock_id>', methods=['DELETE'])
    @login_required
    def delete_stock_from_portfolio(portfolio_id, stock_id):
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                # Confirm the portfolio belongs to the current user
                cursor.execute(
                    "SELECT id FROM portfolios WHERE id = %s AND user_id = %s",
                    (portfolio_id, current_user.id)
                )
                if not cursor.fetchone():
                    return jsonify({'error': 'Portfolio not found or unauthorized'}), 404

                # Delete stock from portfolio
                cursor.execute(
                    "DELETE FROM portfolio_stocks WHERE id = %s AND portfolio_id = %s",
                    (stock_id, portfolio_id)
                )
                if cursor.rowcount == 0:
                    return jsonify({'error': 'Stock not found in portfolio'}), 404

            conn.commit()
            return jsonify({'message': 'Stock removed from portfolio successfully'}), 200
        except Exception as e:
            logging.error(f"Error deleting stock from portfolio: {e}", exc_info=True)
            return jsonify({'error': 'Failed to delete stock'}), 500
        finally:
            conn.close()

    @app.route('/api/portfolio/<int:portfolio_id>/allocation', methods=['GET'])
    @login_required
    def get_portfolio_allocation(portfolio_id):
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT ps.stock_symbol, ps.shares, sd.close
                    FROM portfolio_stocks ps
                             LEFT JOIN (SELECT stock_symbol, close
                                        FROM stock_data
                                        WHERE date = (SELECT MAX(date)
                                                      FROM stock_data
                                                      WHERE stock_symbol = stock_data.stock_symbol)) sd
                                       ON ps.stock_symbol = sd.stock_symbol
                    WHERE ps.portfolio_id = %s
                    """,
                    (portfolio_id,)
                )
                rows = cursor.fetchall()

            if not rows:
                return jsonify([]), 200

            values = []
            total = 0.0
            for stock_symbol, shares, close in rows:
                if close is None or shares is None:
                    continue
                value = shares * close
                values.append((stock_symbol, value))
                total += value

            if total == 0:
                return jsonify([]), 200

            allocation = [
                {"symbol": sym, "percent": round(val / total * 100, 2)}
                for sym, val in values
            ]

            return jsonify(allocation), 200

        except Exception as e:
            logging.error(f"Error in allocation endpoint: {e}", exc_info=True)
            return jsonify({'error': 'Failed to compute allocation'}), 500
        finally:
            conn.close()

    @app.route('/api/portfolio/<int:portfolio_id>/growth', methods=['GET'])
    @login_required
    def get_portfolio_growth(portfolio_id):
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                # Get all stocks in portfolio with purchase date and shares
                cursor.execute(
                    """
                    SELECT stock_symbol, shares, purchase_date
                    FROM portfolio_stocks
                    WHERE portfolio_id = %s
                    """,
                    (portfolio_id,)
                )
                portfolio = cursor.fetchall()

                if not portfolio:
                    return jsonify([]), 200

                # Fetch historical stock data
                stock_data = {}
                for stock_symbol, shares, purchase_date in portfolio:
                    cursor.execute(
                        """
                        SELECT date, close
                        FROM stock_data
                        WHERE stock_symbol = %s
                          AND date >= %s
                        ORDER BY date ASC
                        """,
                        (stock_symbol, purchase_date)
                    )
                    prices = cursor.fetchall()
                    for date, close in prices:
                        if date not in stock_data:
                            stock_data[date] = 0
                        stock_data[date] += close * shares

            # Prepare response
            sorted_data = sorted(stock_data.items())
            growth = [{"date": date.strftime('%Y-%m-%d'), "value": round(value, 2)} for date, value in sorted_data]

            return jsonify(growth), 200

        except Exception as e:
            logging.error(f"Error computing portfolio growth: {e}", exc_info=True)
            return jsonify({'error': 'Failed to compute portfolio growth'}), 500
        finally:
            conn.close()

    @app.route('/api/portfolios/<int:portfolio_id>/stocks/<int:stock_id>', methods=['PATCH'])
    @login_required
    def update_stock_in_portfolio(portfolio_id, stock_id):
        data = request.get_json()
        purchase_price = data.get('purchase_price')
        shares = data.get('shares')
        purchase_date = data.get('purchase_date')

        if not any([purchase_price, shares, purchase_date]):
            return jsonify({'error': 'At least one field must be provided to update'}), 400

        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                # Ensure the portfolio belongs to the user
                cursor.execute(
                    "SELECT id FROM portfolios WHERE id = %s AND user_id = %s",
                    (portfolio_id, current_user.id)
                )
                if not cursor.fetchone():
                    return jsonify({'error': 'Portfolio not found or unauthorized'}), 404

                # Build dynamic update
                updates = []
                values = []

                if purchase_price is not None:
                    updates.append("purchase_price = %s")
                    values.append(purchase_price)
                if shares is not None:
                    updates.append("shares = %s")
                    values.append(shares)
                if purchase_date is not None:
                    updates.append("purchase_date = %s")
                    values.append(purchase_date)

                values.extend([stock_id, portfolio_id])

                cursor.execute(
                    f"""
                    UPDATE portfolio_stocks
                    SET {', '.join(updates)}
                    WHERE id = %s AND portfolio_id = %s
                    """,
                    tuple(values)
                )

                if cursor.rowcount == 0:
                    return jsonify({'error': 'Stock not found in portfolio'}), 404

            conn.commit()
            return jsonify({'message': 'Stock updated successfully'}), 200
        except Exception as e:
            logging.error(f"Error updating stock in portfolio: {e}", exc_info=True)
            return jsonify({'error': 'Failed to update stock'}), 500
        finally:
            conn.close()


def risk_routes(app):
    @app.route('/api/risk-profile', methods=['POST'])
    @login_required
    def submit_risk_profile():
        data = request.get_json()
        score = data.get('score')
        if score is None or not (0 <= score <= 100):
            return jsonify({'error': 'Invalid risk score'}), 400

        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO user_risk_profiles (user_id, risk_score)
                    VALUES (%s, %s)
                    ON CONFLICT (user_id) DO UPDATE SET risk_score = EXCLUDED.risk_score
                    """,
                    (current_user.id, score)
                )
            conn.commit()
            return jsonify({'message': 'Risk profile saved'}), 200
        except Exception as e:
            logging.error(f"Error saving risk profile: {e}", exc_info=True)
            return jsonify({'error': 'Failed to save risk profile'}), 500
        finally:
            conn.close()

    @app.route('/api/risk-profile', methods=['GET'])
    @login_required
    def get_risk_profile():
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT risk_score FROM user_risk_profiles WHERE user_id = %s",
                    (current_user.id,)
                )
                result = cursor.fetchone()
                if result:
                    return jsonify({'score': result[0]}), 200
                return jsonify({'score': None}), 200
        except Exception as e:
            logging.error(f"Error fetching risk profile: {e}", exc_info=True)
            return jsonify({'error': 'Failed to fetch risk profile'}), 500
        finally:
            conn.close()

    @app.route('/api/recommendations', methods=['GET'])
    @login_required
    def get_recommendations():
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT risk_score FROM user_risk_profiles WHERE user_id = %s",
                    (current_user.id,)
                )
                result = cursor.fetchone()
                if not result:
                    return jsonify({'error': 'No risk profile found'}), 404

                score = result[0]

            if score <= 30:
                recs = ['JNJ', 'KO', 'VZ', 'TLT']
            elif score <= 70:
                recs = ['AAPL', 'MSFT', 'VOO', 'VTI']
            else:
                recs = ['TSLA', 'NVDA', 'ARKK', 'COIN']

            return jsonify({
                'risk_score': score,
                'recommendation': recs
            }), 200

        except Exception as e:
            logging.error(f"Error generating recommendations: {e}", exc_info=True)
            return jsonify({'error': 'Failed to generate recommendations'}), 500
        finally:
            conn.close()


def goal_routes(app):
    @app.route('/api/goals', methods=['POST'])
    @login_required
    def create_goal():
        data = request.get_json()
        name = data.get('name')
        amount = data.get('target_amount')
        date = data.get('target_date')

        if not name or not amount or not date:
            return jsonify({'error': 'Name, amount, and date are required'}), 400

        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO user_goals (user_id, name, target_amount, target_date)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (current_user.id, name, amount, date)
                )
            conn.commit()
            return jsonify({'message': 'Goal created'}), 201
        except Exception as e:
            logging.error(f"Error saving goal: {e}", exc_info=True)
            return jsonify({'error': 'Failed to save goal'}), 500
        finally:
            conn.close()

    @app.route('/api/goals', methods=['GET'])
    @login_required
    def get_goals():
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, name, target_amount, target_date
                    FROM user_goals
                    WHERE user_id = %s
                    ORDER BY target_date ASC
                    """,
                    (current_user.id,)
                )
                goals = [
                    {"id": row[0], "name": row[1], "target_amount": row[2], "target_date": row[3].strftime('%Y-%m-%d')}
                    for row in cursor.fetchall()
                ]
                return jsonify(goals), 200
        except Exception as e:
            logging.error(f"Error fetching goals: {e}", exc_info=True)
            return jsonify({'error': 'Failed to fetch goals'}), 500
        finally:
            conn.close()
