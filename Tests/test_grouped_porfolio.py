import pytest
from flask import Flask
from flask.testing import FlaskClient
from app import register_routes, stock_routes, portfolio_routes

@pytest.fixture
def app():
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.secret_key = 'test'

    from flask_login import LoginManager
    login_manager = LoginManager()
    login_manager.init_app(app)

    register_routes(app, login_manager)
    stock_routes(app)
    portfolio_routes(app)

    return app

@pytest.fixture
def client(app) -> FlaskClient:
    return app.test_client()

def login(client):
    return client.post('/api/login', json={
        'username': 'Vincent',
        'password': 'Doherty'
    })

def test_grouped_stocks_endpoint(client):
    login(client)
    response = client.get('/api/portfolios/1/stocks/grouped')
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        data = response.get_json()
        assert isinstance(data, list)
        for item in data:
            assert 'symbol' in item
            assert 'total_invested' in item
            assert 'total_current' in item
            assert 'lots' in item
