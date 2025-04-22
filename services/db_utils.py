import os

import psycopg2
import pandas as pd
from datetime import datetime
from services.s3_utils import upload_to_s3, download_from_s3
from dotenv import load_dotenv
load_dotenv()
# --- Database connection ---
def get_db_connection():
    return psycopg2.connect(
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT')
    )

# --- Load historical stock prices ---
def get_stock_data(stock_symbol):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT date, close 
                FROM stock_data 
                WHERE stock_symbol = %s 
                ORDER BY date
                """,
                (stock_symbol,)
            )
            rows = cursor.fetchall()
            df = pd.DataFrame(rows, columns=['date', 'close'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
    finally:
        conn.close()

# --- Store stock price data ---
def store_stock_data(stock_symbol, data):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            for date, row in data.iterrows():
                cursor.execute(
                    """
                    INSERT INTO stock_data (stock_symbol, date, close)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (stock_symbol, date) DO UPDATE
                        SET close = EXCLUDED.close
                    """,
                    (stock_symbol, date, float(row['close']))
                )
        conn.commit()
    finally:
        conn.close()

# --- Store forecast results ---
def store_predictions(stock_symbol, predictions):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            for date, row in predictions.iterrows():
                cursor.execute(
                    """
                    INSERT INTO stock_predictions (stock_symbol, date, predicted_close)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (stock_symbol, date) DO UPDATE
                    SET predicted_close = EXCLUDED.predicted_close
                    """,
                    (stock_symbol, date, float(row['Predicted Close']))
                )
        conn.commit()
    finally:
        conn.close()

# --- Save model metadata to database ---
def save_model_metadata(stock_symbol, model_url, scaler_url, loss):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT model_version FROM stock_models WHERE stock_symbol = %s",
                (stock_symbol,)
            )
            result = cursor.fetchone()
            version = result[0] + 1 if result else 1

            cursor.execute(
                """
                INSERT INTO stock_models (stock_symbol, model_url, scaler_url, last_updated, model_version, training_loss)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (stock_symbol) DO UPDATE SET
                    model_url = EXCLUDED.model_url,
                    scaler_url = EXCLUDED.scaler_url,
                    last_updated = EXCLUDED.last_updated,
                    model_version = EXCLUDED.model_version,
                    training_loss = EXCLUDED.training_loss
                """,
                (stock_symbol, model_url, scaler_url, datetime.now(), version, loss)
            )
        conn.commit()
    finally:
        conn.close()

# --- Save model & scaler ---
def save_model_and_scaler(stock_symbol, model, scaler, loss):
    model_url = upload_to_s3(model, f"models/{stock_symbol}.pkl")
    scaler_url = upload_to_s3(scaler, f"scalers/{stock_symbol}.pkl")
    save_model_metadata(stock_symbol, model_url, scaler_url, loss)

# --- Load model & metadata ---
def load_model_from_db(stock_symbol):
    conn = get_db_connection()
    model = scaler = last_updated = None
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT model_url, scaler_url, last_updated FROM stock_models WHERE stock_symbol = %s",
                (stock_symbol,)
            )
            result = cursor.fetchone()
            if not result:
                return None, None, None
            model_url, scaler_url, last_updated = result
            model_key = model_url.split(f".com/")[1]
            scaler_key = scaler_url.split(f".com/")[1]
            model = download_from_s3(model_key)
            scaler = download_from_s3(scaler_key)
    finally:
        conn.close()
    return model, scaler, last_updated