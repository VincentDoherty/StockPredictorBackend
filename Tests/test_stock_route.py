import pytest
from flask import json
from app import app as flask_app
from services.db_utils import get_db_connection

@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        # login first to authorize
        login_resp = client.post('/api/login', json={"username": "testuser", "password": "testpassword"})
        assert login_resp.status_code == 200
        yield client

@pytest.fixture
def seed_stock(client):
    response = client.post('/api/stock', json={'stock_symbol': 'MSFT'})
    assert response.status_code == 200
    return 'MSFT'

def test_get_stock_history(client, seed_stock):
    response = client.get(f'/api/stock/{seed_stock}/history')
    assert response.status_code == 200
    data = response.get_json()
    assert 'symbol' in data and data['symbol'] == seed_stock
    assert 'history' in data and isinstance(data['history'], list)

def test_get_stock_predictions(client, seed_stock):
    response = client.get(f'/api/stock/{seed_stock}/predictions')
    assert response.status_code == 200
    data = response.get_json()
    assert 'symbol' in data and data['symbol'] == seed_stock
    assert 'predicted' in data and isinstance(data['predicted'], list)

def test_refresh_stock_data(client, seed_stock):
    response = client.post(f'/api/stock/{seed_stock}/refresh')
    assert response.status_code == 200
    assert 'message' in response.get_json()

def test_portfolio_metrics(client):
    # create new portfolio
    create_port = client.post('/api/portfolios', json={'name': 'Metrics Portfolio'})
    portfolio_id = create_port.get_json()['portfolio_id']

    payload = {
        'stock_symbol': 'MSFT',
        'purchase_price': 200.0,
        'purchase_date': '2024-01-01',
    }
    client.post(f'/api/portfolios/{portfolio_id}/stocks', json=payload)

    response = client.get(f'/api/portfolio/{portfolio_id}/metrics')
    assert response.status_code == 200
    metrics = response.get_json()
    assert 'total_return' in metrics
    assert 'sharpe_ratio' in metrics

def test_rebalance_portfolio(client):
    create_port = client.post('/api/portfolios', json={'name': 'Rebalance Portfolio'})
    portfolio_id = create_port.get_json()['portfolio_id']

    for symbol in ['AAPL', 'MSFT']:
        payload = {
            'stock_symbol': symbol,
            'purchase_price': 150.0,
            'purchase_date': '2024-01-01',
        }
        client.post(f'/api/portfolios/{portfolio_id}/stocks', json=payload)

    response = client.post(f'/api/portfolios/{portfolio_id}/rebalance')
    assert response.status_code == 200
    rebalance = response.get_json()
    assert 'allocation' in rebalance
