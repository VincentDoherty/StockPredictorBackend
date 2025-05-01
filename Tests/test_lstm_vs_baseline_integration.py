import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from services.db_utils import load_model_from_db, get_stock_data
from retrain_model import retrain_model

@pytest.mark.integration
def test_lstm_vs_baseline_prediction():
    stock_symbol = "CL"

    assert retrain_model(stock_symbol), "Retraining failed"

    model, scaler, _ = load_model_from_db(stock_symbol)
    assert model and scaler, "Model or scaler not loaded"

    data = get_stock_data(stock_symbol)
    assert not data.empty and 'close' in data.columns, "No valid stock data"

    # Prepare time series
    scaled = scaler.transform(data[['close']])
    X, y = [], []
    for i in range(365, len(scaled) - 30):
        X.append(scaled[i - 365:i, 0])
        y.append(scaled[i:i + 30, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Predict with LSTM
    y_pred_scaled = model.predict(X, verbose=0)
    y_pred_lstm = scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = scaler.inverse_transform(y).flatten()

    # Baseline: yesterday's price
    y_baseline = np.array(data['close'].values[365:-30], dtype=np.float32)
    y_actual = np.array(data['close'].values[366:-29], dtype=np.float32)

    mae_baseline = mean_absolute_error(y_actual, y_baseline)
    mse_baseline = mean_squared_error(y_actual, y_baseline)
    rmse_baseline = np.sqrt(mse_baseline)

    mae_lstm = mean_absolute_error(y_true, y_pred_lstm)
    mse_lstm = mean_squared_error(y_true, y_pred_lstm)
    rmse_lstm = np.sqrt(mse_lstm)

    print(f"\n[REAL] MAE (LSTM): {mae_lstm:.4f}")
    print(f"[REAL] RMSE (LSTM): {rmse_lstm:.4f}")
    print(f"[BASELINE] MAE: {mae_baseline:.4f}")
    print(f"[BASELINE] RMSE: {rmse_baseline:.4f}")

    # Optional checks
    assert mae_lstm <= mae_baseline, "LSTM should beat baseline MAE"
    assert rmse_lstm <= rmse_baseline, "LSTM should beat baseline RMSE"