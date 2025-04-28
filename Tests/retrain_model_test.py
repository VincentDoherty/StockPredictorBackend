import pytest
from services.db_utils import load_model_from_db, get_stock_data
from retrain_model import retrain_model

@pytest.mark.integration
def test_retrain_and_load_pipeline():
    stock_symbol = "NVDA"

    # Step 1: Retrain the model and store it
    success = retrain_model(stock_symbol)
    assert success is True, "Model retraining failed."

    # Step 2: Load the model and scaler
    model, scaler, last_updated = load_model_from_db(stock_symbol)
    assert model is not None, "Model not loaded from DB."
    assert scaler is not None, "Scaler not loaded from DB."
    assert last_updated is not None, "Model last_updated not retrieved."

    # Step 3: Fetch the stock data and verify shape
    data = get_stock_data(stock_symbol)
    assert not data.empty, "No stock data found for symbol."
    assert 'close' in data.columns, "Missing 'close' column in stock data."

    print("\nModel and scaler loaded successfully for:", stock_symbol)
    print("Data shape:", data.shape)
