import os

import psycopg2
from dotenv import load_dotenv
load_dotenv()
def create_tables():
    commands = [
        """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(255) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            risk_tolerance VARCHAR(50),
            investment_preferences TEXT,
            is_admin BOOLEAN DEFAULT FALSE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS portfolio_stocks (
            id SERIAL PRIMARY KEY,
            portfolio_id INTEGER REFERENCES portfolios(id),
            stock_symbol VARCHAR(10) NOT NULL,
            shares FLOAT NOT NULL DEFAULT 1.0,
            purchase_price NUMERIC,
            purchase_date DATE NOT NULL DEFAULT CURRENT_DATE
        )
        """,
        """
        CREATE TABLE if not exists portfolios (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id),
        name VARCHAR(255) NOT NULL
        );
        """
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id SERIAL PRIMARY KEY,
            portfolio_id INTEGER REFERENCES portfolios(id),
            stock_symbol VARCHAR(10) NOT NULL,
            transaction_type VARCHAR(4) CHECK (transaction_type IN ('buy', 'sell')),
            quantity INTEGER NOT NULL,
            price NUMERIC NOT NULL,
            date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS stock_data (
            stock_symbol VARCHAR(10) NOT NULL,
            date DATE NOT NULL,
            close NUMERIC NOT NULL,
            PRIMARY KEY (stock_symbol, date)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS stock_predictions (
            stock_symbol VARCHAR(10) NOT NULL,
            date DATE NOT NULL,
            predicted_close NUMERIC NOT NULL,
            PRIMARY KEY (stock_symbol, date)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS notifications (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            message TEXT NOT NULL,
            date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS feedback_basic (
        id SERIAL PRIMARY KEY,
        comment TEXT NOT NULL,
        submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
        """,
        """
        CREATE TABLE IF NOT EXISTS  feedback_advanced (
        id SERIAL PRIMARY KEY,
        rating INTEGER CHECK (rating >= 1 AND rating <= 5),
        ease_of_use INTEGER CHECK (ease_of_use BETWEEN 1 AND 5),
        useful_features TEXT,
        missing_features TEXT,
        general_comments TEXT,
        submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS stock_models (
            stock_symbol VARCHAR(10) PRIMARY KEY,
            model_url TEXT NOT NULL,
            last_updated TIMESTAMP NOT NULL,
            scaler_url TEXT,
            model_version INTEGER DEFAULT 1,
            training_loss FLOAT

        )
        """,
        """
        CREATE TABLE IF NOT EXISTS user_risk_profiles (
        user_id INTEGER PRIMARY KEY REFERENCES users(id),
        risk_score INTEGER NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS user_goals (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id),
        name TEXT NOT NULL,
        target_amount FLOAT NOT NULL,
        target_date DATE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ); """
    ]

    conn = None
    try:
        conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT')
        )
        cursor = conn.cursor()
        for command in commands:
            cursor.execute(command)
        cursor.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

if __name__ == '__main__':
    create_tables()