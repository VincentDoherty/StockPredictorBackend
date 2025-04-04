import pickle
from sqlalchemy import create_engine, text
import pandas as pd
from datetime import datetime

# Database connection
def get_db_connection():
    engine = create_engine('postgresql://postgres:vdonkeY800@localhost:5432/investmentdb')
    return engine

# Retrieve stock data from the database
def get_stock_data(stock_symbol):
    engine = get_db_connection()
    query = f"SELECT date, close FROM stock_data WHERE stock_symbol = '{stock_symbol}' ORDER BY date"
    stock_data = pd.read_sql(query, engine)
    return stock_data

# Save the trained model to the database
def store_stock_data(stock_symbol, data):
    engine = get_db_connection()
    with engine.connect() as conn:
        for date, row in data.iterrows():
            conn.execute(
                """
                INSERT INTO stock_data (stock_symbol, date, close)
                VALUES (%s, %s, %s)
                ON CONFLICT (stock_symbol, date)
                DO UPDATE SET close = EXCLUDED.close
                """,
                (stock_symbol, date, row['Close'].item())
            )

# Load the trained model from the database
def store_predictions(stock_symbol, predictions):
    engine = get_db_connection()
    with engine.connect() as conn:
        for date, price in predictions.iterrows():
            conn.execute(
                "INSERT INTO stock_predictions (stock_symbol, date, predicted_close) VALUES (%s, %s, %s) ON CONFLICT (stock_symbol, date) DO UPDATE SET predicted_close = EXCLUDED.predicted_close",
                (stock_symbol, date, float(price['Predicted Close']))
            )


# Load the trained model from the database
def load_model_from_db(stock_symbol):
    engine = get_db_connection()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT model, last_updated FROM stock_models WHERE stock_symbol = :stock_symbol"),
            {"stock_symbol": stock_symbol}
        ).fetchone()
        if result:
            model = pickle.loads(result[0])
            last_updated = result[1]
            return model, last_updated
    return None, None


# Save the trained model to the database
def save_model_to_db(stock_symbol, model):
    engine = get_db_connection()
    with engine.connect() as conn:
        try:
            model_binary = pickle.dumps(model)
            conn.execute(
                """
                INSERT INTO stock_models (stock_symbol, model, last_updated)
                VALUES (:stock_symbol, :model, :last_updated)
                ON CONFLICT (stock_symbol) DO UPDATE
                SET model = EXCLUDED.model, last_updated = EXCLUDED.last_updated
                """,
                {
                    "stock_symbol": stock_symbol,
                    "model": model_binary,
                    "last_updated": datetime.now()
                }
            )
        except Exception as e:
            print(f"Error saving model to the database: {e}")