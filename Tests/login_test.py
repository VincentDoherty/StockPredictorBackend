import pytest
from flask import json
from app import app as flask_app
from services.db_utils import get_db_connection

@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client

def test_register_new_user(client):
    response = client.post('/api/register', json={
        'username': 'testuser1',
        'password': 'securepass'
    })
    assert response.status_code == 201
    assert 'User registered successfully' in response.get_json()['message']

def test_register_duplicate_user(client):
    client.post('/api/register', json={
        'username': 'testuser2',
        'password': 'securepass'
    })
    response = client.post('/api/register', json={
        'username': 'testuser2',
        'password': 'anotherpass'
    })
    assert response.status_code == 500

def test_login_valid_user(client):
    client.post('/api/register', json={
        'username': 'testuser3',
        'password': 'securepass'
    })
    response = client.post('/api/login', json={
        'username': 'testuser3',
        'password': 'securepass'
    })
    assert response.status_code == 200
    assert 'Login successful' in response.get_json()['message']

def test_login_invalid_user(client):
    response = client.post('/api/login', json={
        'username': 'nonexistent',
        'password': 'wrongpass'
    })
    assert response.status_code == 401

def test_session_cookie_persists(client):
    client.post('/api/register', json={
        'username': 'testuser4',
        'password': 'securepass'
    })
    login_resp = client.post('/api/login', json={
        'username': 'testuser4',
        'password': 'securepass'
    })
    assert login_resp.status_code == 200
    with client.session_transaction() as sess:
        assert '_user_id' in sess

def test_logout(client):
    client.post('/api/register', json={
        'username': 'testuser5',
        'password': 'securepass'
    })
    client.post('/api/login', json={
        'username': 'testuser5',
        'password': 'securepass'
    })
    response = client.post('/api/logout')
    assert response.status_code == 200
    assert 'Logged out successfully' in response.get_json()['message']

def test_protected_route_requires_login(client):
    response = client.get('/api/portfolios')  # Example protected route
    assert response.status_code in [302, 401]
