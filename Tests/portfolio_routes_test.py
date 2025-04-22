import pytest
from flask import json
from app import app as flask_app
from services.db_utils import get_db_connection


@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        response = client.post('/api/login', json={
            'username': 'testuser',
            'password': 'testpassword'
        })
        assert response.status_code == 200
        yield client


@pytest.fixture
def create_portfolio(client):
    response = client.post('/api/portfolios', json={'name': 'Test Portfolio'})
    assert response.status_code == 201
    return response.get_json()['portfolio_id']

def test_get_portfolios(client):
    response = client.get('/api/portfolios')
    assert response.status_code == 200
    assert isinstance(response.get_json(), list)

def test_create_portfolio(client):
    response = client.post('/api/portfolios', json={'name': 'New Portfolio'})
    assert response.status_code == 201
    data = response.get_json()
    assert 'portfolio_id' in data

def test_add_stock_to_portfolio(client, create_portfolio):
    payload = {
        'stock_symbol': 'AAPL',
        'purchase_price': 100.0,
        'purchase_date': '2024-01-01',
        'shares': 5
    }
    response = client.post(f'/api/portfolios/{create_portfolio}/stocks', json=payload)
    assert response.status_code == 201

def test_get_portfolio_sorted(client, create_portfolio):
    response = client.get(f'/api/portfolios/{create_portfolio}?sort_by=profit_loss&order=desc')
    assert response.status_code == 200
    assert isinstance(response.get_json(), list)

def test_delete_portfolio(client, create_portfolio):
    response = client.delete(f'/api/portfolios/{create_portfolio}')
    assert response.status_code == 200

def test_delete_stock_from_portfolio(client, create_portfolio):
    payload = {
        'stock_symbol': 'AAPL',
        'purchase_price': 100.0,
        'purchase_date': '2024-01-01',
        'shares': 5
    }
    add_response = client.post(f'/api/portfolios/{create_portfolio}/stocks', json=payload)
    assert add_response.status_code == 201

    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT id FROM portfolio_stocks WHERE portfolio_id = %s", (create_portfolio,))
        stock_id = cursor.fetchone()[0]
    conn.close()

    delete_response = client.delete(f'/api/portfolios/{create_portfolio}/stocks/{stock_id}')
    assert delete_response.status_code == 200

def test_portfolio_allocation(client, create_portfolio):
    payload = {
        'stock_symbol': 'AAPL',
        'purchase_price': 100.0,
        'purchase_date': '2024-01-01',
        'shares': 5
    }
    client.post(f'/api/portfolios/{create_portfolio}/stocks', json=payload)
    response = client.get(f'/api/portfolio/{create_portfolio}/allocation')
    assert response.status_code == 200
    assert isinstance(response.get_json(), list)

def test_portfolio_growth(client, create_portfolio):
    payload = {
        'stock_symbol': 'AAPL',
        'purchase_price': 100.0,
        'purchase_date': '2024-01-01',
        'shares': 5
    }
    client.post(f'/api/portfolios/{create_portfolio}/stocks', json=payload)
    response = client.get(f'/api/portfolio/{create_portfolio}/growth')
    assert response.status_code == 200
    assert isinstance(response.get_json(), list)

def test_patch_portfolio_stock(client, create_portfolio):
    payload = {
        'stock_symbol': 'AAPL',
        'purchase_price': 100.0,
        'purchase_date': '2024-01-01',
        'shares': 5
    }
    client.post(f'/api/portfolios/{create_portfolio}/stocks', json=payload)

    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT id FROM portfolio_stocks WHERE portfolio_id = %s", (create_portfolio,))
        stock_id = cursor.fetchone()[0]
    conn.close()

    patch_data = {
        'purchase_price': 150.0
    }
    response = client.patch(f'/api/portfolios/{create_portfolio}/stocks/{stock_id}', json=patch_data)
    assert response.status_code == 200
    assert 'updated' in response.get_json()['message']
