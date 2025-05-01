import pytest
import numpy as np
from services.db_utils import load_model_from_db, get_stock_data
from retrain_model import retrain_model
from sklearn.metrics import mean_squared_error, mean_absolute_error


@pytest.mark.integration
def test_retrain_and_load_pipeline():
    stock_symbol = "MSFT"

    success = retrain_model(stock_symbol)
    assert success is True, "Model retraining failed."

    model, scaler, last_updated = load_model_from_db(stock_symbol)
    assert model is not None, "Model not loaded from DB."
    assert scaler is not None, "Scaler not loaded from DB."
    assert last_updated is not None, "Model last_updated not retrieved."

    data = get_stock_data(stock_symbol)
    assert not data.empty, "No stock data found for symbol."
    assert 'close' in data.columns, "Missing 'close' column in stock data."

    # Prepare data
    scaled_data = scaler.transform(data[['close']])
    X = []
    y = []
    for i in range(365, len(scaled_data) - 30):
        X.append(scaled_data[i - 365:i, 0])
        y.append(scaled_data[i:i + 30, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    preds = model.predict(X, verbose=0)
    preds_inv = scaler.inverse_transform(preds)
    y_inv = scaler.inverse_transform(y)

    # Flatten for comparison
    preds_flat = preds_inv.flatten()
    y_flat = y_inv.flatten()

    mse = mean_squared_error(y_flat, preds_flat)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_flat, preds_flat)

    print(f"\nModel and scaler loaded successfully for: {stock_symbol}")
    print(f"Data shape: {data.shape}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
