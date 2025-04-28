import logging
from datetime import datetime, timedelta
from decimal import Decimal
import numpy as np
import pandas as pd
import yfinance as yf
from flask import request, jsonify
from flask_login import login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash
from retrain_model import retrain_model, load_model_from_db
from services.db_utils import get_stock_data, store_predictions, store_stock_data, get_db_connection
from user_db import get_user_by_id, get_user_by_username, create_user
from Decorators.ownership import require_ownership

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

    @app.route('/api/check-session')
    def check_session():
        if current_user.is_authenticated:
            return jsonify({'authenticated': True}), 200
        else:
            return jsonify({'authenticated': False}), 200


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

            if app.config.get("TESTING", False):  #  this allows test to see the traceback

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
    @require_ownership('portfolios', url_param='portfolio_id')
    def add_stock_to_portfolio(portfolio_id):
        data = request.get_json()
        stock_symbol = data.get('stock_symbol')
        purchase_price = data.get('purchase_price')
        purchase_date = data.get('purchase_date')

        if not all([stock_symbol, purchase_price, purchase_date]):
            return jsonify({'error': 'Stock symbol, price, and date are required'}), 400

        try:
            start_date = datetime.strptime(purchase_date, "%Y-%m-%d")
            end_date = start_date + timedelta(days=1)
            history = yf.Ticker(stock_symbol).history(start=start_date, end=end_date)
            actual_price = history['Close'].iloc[0] if not history.empty else None
            if actual_price is None:
                return jsonify({'error': 'Unable to fetch share price from yfinance'}), 400

            shares = round(float(purchase_price) / float(actual_price), 4)

        except Exception as e:
            logging.error(f"Failed to fetch price from yfinance: {e}", exc_info=True)
            return jsonify({'error': 'Failed to fetch stock data'}), 500

        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO portfolio_stocks (portfolio_id, stock_symbol, purchase_price, purchase_date, shares)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (portfolio_id, stock_symbol.upper(), purchase_price, purchase_date, shares)
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
    @require_ownership('portfolios', url_param='portfolio_id')
    def get_portfolio(portfolio_id):
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT ps.id, ps.stock_symbol, ps.purchase_price, ps.purchase_date, ps.shares
                    FROM portfolio_stocks ps
                    WHERE ps.portfolio_id = %s
                    """,
                    (portfolio_id,)
                )
                stocks = cursor.fetchall()

            if not stocks:
                return jsonify([]), 200

            portfolio = []
            for stock in stocks:
                stock_id, stock_symbol, purchase_price, purchase_date, shares = stock
                purchase_price = float(purchase_price)
                shares = float(shares)

                try:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            """
                            SELECT close
                            FROM stock_data
                            WHERE stock_symbol = %s
                            ORDER BY date DESC
                            LIMIT 1
                            """,
                            (stock_symbol,)
                        )
                        row = cursor.fetchone()
                        current_price = float(row[0]) if row else None
                except Exception:
                    current_price = None

                # fallback to yfinance and cache if not found in DB
                if current_price is None:
                    try:
                        history = yf.Ticker(stock_symbol).history(period="1d")
                        current_price = float(history["Close"].iloc[-1]) if not history.empty else purchase_price

                        # cache into DB
                        with conn.cursor() as cursor:
                            cursor.execute(
                                """
                                INSERT INTO stock_data (stock_symbol, date, close)
                                VALUES (%s, CURRENT_DATE, %s)
                                ON CONFLICT DO NOTHING
                                """,
                                (stock_symbol, current_price)
                            )
                            conn.commit()

                    except Exception:
                        current_price = purchase_price

                total_value = current_price * shares
                profit_loss = total_value - purchase_price

                portfolio.append({
                    'id': stock_id,
                    'stock_symbol': stock_symbol,
                    'purchase_price': f"{purchase_price:.2f}",
                    'purchase_date': purchase_date.strftime('%Y-%m-%d'),
                    'shares': round(shares, 4),
                    'current_price': f"{current_price:.2f}",
                    'current_value': f"{total_value:.2f}",
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
    @require_ownership('portfolios', url_param='portfolio_id')
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

    # Grouped portfolio endpoint using DB-first price fallback to yfinance
    @app.route('/api/portfolios/<int:portfolio_id>/stocks/grouped', methods=['GET'])
    @login_required
    @require_ownership('portfolios', url_param='portfolio_id')
    def get_grouped_portfolio(portfolio_id):
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT ps.id, ps.stock_symbol, ps.purchase_price, ps.shares, ps.purchase_date
                    FROM portfolio_stocks ps
                    WHERE ps.portfolio_id = %s
                    """,
                    (portfolio_id,)
                )
                rows = cursor.fetchall()

            grouped = {}
            for stock_id, symbol, price, shares, date in rows:
                price = float(price)
                shares = float(shares)

                try:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            """
                            SELECT close
                            FROM stock_data
                            WHERE stock_symbol = %s
                            ORDER BY date DESC
                            LIMIT 1
                            """,
                            (symbol,)
                        )
                        row = cursor.fetchone()
                        current_price = float(row[0]) if row else None
                except Exception:
                    current_price = None

                if current_price is None:
                    try:
                        history = yf.Ticker(symbol).history(period="1d")
                        current_price = float(history['Close'].iloc[-1]) if not history.empty else price
                        with conn.cursor() as cursor:
                            cursor.execute(
                                """
                                INSERT INTO stock_data (stock_symbol, date, close)
                                VALUES (%s, CURRENT_DATE, %s)
                                ON CONFLICT DO NOTHING
                                """,
                                (symbol, current_price)
                            )
                            conn.commit()
                    except Exception:
                        current_price = price

                total_value = round(current_price * shares, 2)
                profit_loss = round(total_value - price, 2)  # Updated as per your request

                entry = {
                    'id': stock_id,
                    'purchase_price': price,
                    'purchase_date': date.strftime('%Y-%m-%d'),
                    'shares': shares,
                    'current_price': current_price,
                    'current_value': total_value,
                    'invested_value': price,
                    'profit_loss': profit_loss,
                }

                if symbol not in grouped:
                    grouped[symbol] = {
                        'symbol': symbol,
                        'total_shares': 0,
                        'total_invested': 0,
                        'total_current': 0,
                        'total_pl': 0,
                        'lots': []
                    }

                g = grouped[symbol]
                g['total_shares'] += entry['shares']
                g['total_invested'] += entry['invested_value']
                g['total_current'] += entry['current_value']
                g['total_pl'] += entry['profit_loss']
                g['lots'].append(entry)

            for g in grouped.values():
                g['total_invested'] = round(g['total_invested'], 2)
                g['total_current'] = round(g['total_current'], 2)
                g['total_pl'] = round(g['total_current'] - g['total_invested'], 2)

            return jsonify(list(grouped.values())), 200

        except Exception as e:
            logging.error(f"Error grouping portfolio stocks: {e}", exc_info=True)
            return jsonify({'error': 'Failed to group portfolio stocks'}), 500
        finally:
            conn.close()

    @app.route('/api/portfolios/<int:portfolio_id>/stocks/<int:stock_id>', methods=['DELETE'])
    @login_required
    @require_ownership('portfolios', url_param='portfolio_id')
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


    @app.route('/api/portfolios/<int:portfolio_id>/stocks/<int:stock_id>', methods=['PATCH'])
    @login_required
    @require_ownership('portfolios', url_param='portfolio_id')
    def update_stock_in_portfolio(portfolio_id, stock_id):
        data = request.get_json()
        purchase_price = data.get('purchase_price')
        purchase_date = data.get('purchase_date')
        stock_symbol = data.get('stock_symbol')

        if not all([purchase_price, purchase_date, stock_symbol]):
            return jsonify({'error': 'Stock symbol, price, and date are required to update'}), 400

        try:
            start_date = datetime.strptime(purchase_date, "%Y-%m-%d")
            end_date = start_date + timedelta(days=1)

            conn = get_db_connection()
            try:
                with conn.cursor() as cursor:
                    # Try fetching historical price from DB first
                    cursor.execute(
                        """
                        SELECT close
                        FROM stock_data
                        WHERE stock_symbol = %s
                          AND date = %s
                        LIMIT 1
                        """,
                        (stock_symbol, start_date)
                    )
                    row = cursor.fetchone()
                    actual_price = float(row[0]) if row else None

                    # If not found, fetch from yfinance and cache
                    if actual_price is None:
                        history = yf.Ticker(stock_symbol).history(start=start_date, end=end_date)
                        actual_price = float(history['Close'].iloc[0]) if not history.empty else None
                        if actual_price is None:
                            return jsonify({'error': 'Unable to fetch share price from yfinance'}), 400
                        cursor.execute(
                            """
                            INSERT INTO stock_data (stock_symbol, date, close)
                            VALUES (%s, %s, %s)
                            ON CONFLICT DO NOTHING
                            """,
                            (stock_symbol, start_date, actual_price)
                        )

                    shares = round(float(purchase_price) / actual_price, 4)

                    cursor.execute(
                        "SELECT id FROM portfolios WHERE id = %s AND user_id = %s",
                        (portfolio_id, current_user.id)
                    )
                    if not cursor.fetchone():
                        return jsonify({'error': 'Portfolio not found or unauthorized'}), 404

                    cursor.execute(
                        """
                        UPDATE portfolio_stocks
                        SET purchase_price = %s,
                            purchase_date  = %s,
                            shares         = %s
                        WHERE id = %s
                          AND portfolio_id = %s
                        """,
                        (purchase_price, purchase_date, shares, stock_id, portfolio_id)
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

        except Exception as e:
            logging.error(f"Failed to fetch or compute shares: {e}", exc_info=True)
            return jsonify({'error': 'Failed to compute shares for update'}), 500

    @app.route('/api/portfolio/<int:portfolio_id>/growth', methods=['GET'])
    @login_required
    @require_ownership('portfolios', url_param='portfolio_id')
    def get_portfolio_growth(portfolio_id):
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
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

            price_map = {}
            min_date = None

            for stock_symbol, shares, purchase_date in portfolio:
                start_date = purchase_date

                if min_date is None or start_date < min_date:
                    min_date = start_date

                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT date, close
                    FROM stock_data
                    WHERE stock_symbol = %s
                      AND date >= %s
                    ORDER BY date ASC
                    """,
                    (stock_symbol, start_date)
                )
                existing = dict(cursor.fetchall())

                bdays = pd.bdate_range(start=start_date, end=datetime.today().date())
                missing = [d.date() for d in bdays if d.date() not in existing]

                if missing:
                    try:
                        history = yf.Ticker(stock_symbol).history(
                            start=min(missing), end=max(missing) + pd.Timedelta(days=1)
                        )
                        for d in missing:
                            if d in history.index and pd.notna(history.loc[d, 'Close']):
                                close = float(history.loc[d, 'Close'])
                                existing[d] = close
                                try:
                                    with conn.cursor() as cursor:
                                        cursor.execute(
                                            """
                                            INSERT INTO stock_data (stock_symbol, date, close)
                                            VALUES (%s, %s, %s)
                                            ON CONFLICT DO NOTHING
                                            """,
                                            (stock_symbol, d, close)
                                        )
                                    conn.commit()
                                except Exception:
                                    pass
                    except Exception as e:
                        logging.warning(f"YFinance fetch failed for {stock_symbol}: {e}")

                for d, close in existing.items():
                    if d not in price_map:
                        price_map[d] = Decimal('0')
                    price_map[d] += Decimal(str(close)) * Decimal(str(shares))

            sorted_dates = sorted(price_map)
            growth = [{'date': d.strftime('%Y-%m-%d'), 'value': float(round(price_map[d], 2))} for d in sorted_dates]

            return jsonify(growth), 200

        except Exception as e:
            logging.error(f"Error computing portfolio growth: {e}", exc_info=True)
            return jsonify({'error': 'Failed to compute growth'}), 500
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

    @app.route('/api/goals/<int:goal_id>', methods=['PATCH'])
    @login_required
    @require_ownership('user_goals', url_param='goal_id')
    def update_goal(goal_id):
        data = request.get_json()
        name = data.get('name')
        amount = data.get('target_amount')
        date = data.get('target_date')

        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE user_goals
                    SET name          = %s,
                        target_amount = %s,
                        target_date   = %s
                    WHERE id = %s
                      AND user_id = %s
                    """,
                    (name, amount, date, goal_id, current_user.id)
                )
            conn.commit()
            return jsonify({'message': 'Goal updated successfully'}), 200
        except Exception as e:
            logging.error(f"Error updating goal: {e}", exc_info=True)
            return jsonify({'error': 'Failed to update goal'}), 500
        finally:
            conn.close()

    @app.route('/api/goals/<int:goal_id>', methods=['DELETE'])
    @login_required
    @require_ownership('user_goals', url_param='goal_id')
    def delete_goal(goal_id):
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE
                    FROM user_goals
                    WHERE id = %s
                      AND user_id = %s
                    """,
                    (goal_id, current_user.id)
                )
            conn.commit()
            return jsonify({'message': 'Goal deleted successfully'}), 200
        except Exception as e:
            logging.error(f"Error deleting goal: {e}", exc_info=True)
            return jsonify({'error': 'Failed to delete goal'}), 500
        finally:
            conn.close()


def feedback_routes(app):
    @app.route('/api/feedback', methods=['POST'])
    def submit_basic_feedback():
        data = request.get_json()
        comment = data.get('comment')

        if not comment:
            return jsonify({'error': 'Comment is required'}), 400

        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO feedback_basic (comment)
                    VALUES (%s)
                    """,
                    (comment,)
                )
            conn.commit()
            return jsonify({'message': 'Feedback submitted successfully'}), 201
        except Exception as e:
            return jsonify({'error': 'Failed to submit feedback'}), 500
        finally:
            conn.close()

    @app.route('/api/feedback/advanced', methods=['POST'])
    def submit_advanced_feedback():
        data = request.get_json()
        rating = data.get('rating')
        ease_of_use = data.get('ease_of_use')
        useful_features = data.get('useful_features')
        missing_features = data.get('missing_features')
        general_comments = data.get('general_comments')

        if not rating or not ease_of_use:
            return jsonify({'error': 'Rating and ease_of_use are required'}), 400

        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO feedback_advanced (rating, ease_of_use, useful_features, missing_features, general_comments)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (rating, ease_of_use, useful_features, missing_features, general_comments)
                )
            conn.commit()
            return jsonify({'message': 'Advanced feedback submitted successfully'}), 201
        except Exception as e:
            return jsonify({'error': 'Failed to submit advanced feedback'}), 500
        finally:
            conn.close()

    @app.route('/api/feedback/all', methods=['GET'])
    def get_all_feedback():
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id, comment FROM feedback_basic ORDER BY id DESC")
                basic = [{'id': row[0], 'comment': row[1]} for row in cursor.fetchall()]

                cursor.execute("""
                               SELECT id, rating, ease_of_use, useful_features, missing_features, general_comments
                               FROM feedback_advanced
                               ORDER BY id DESC
                               """)
                advanced = [
                    {
                        'id': row[0],
                        'rating': row[1],
                        'ease_of_use': row[2],
                        'useful_features': row[3],
                        'missing_features': row[4],
                        'general_comments': row[5]
                    } for row in cursor.fetchall()
                ]

            return jsonify({
                'basic_feedback': basic,
                'advanced_feedback': advanced
            }), 200

        except Exception as e:
            logging.error(f"Error fetching all feedback: {e}", exc_info=True)
            return jsonify({'error': 'Failed to fetch feedback'}), 500
        finally:
            conn.close()
