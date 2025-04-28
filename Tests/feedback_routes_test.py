import pytest
from application import app as flask_app

@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        with client.session_transaction() as sess:
            sess['_user_id'] = '25'  # Simulate logged-in user
        yield client

def test_submit_basic_feedback(client):
    response = client.post('/api/feedback', json={"comment": "Great app!"})
    assert response.status_code == 201
    assert response.get_json()['message'] == 'Feedback submitted successfully'

def test_submit_basic_feedback_missing_comment(client):
    response = client.post('/api/feedback', json={})
    assert response.status_code == 400

def test_submit_advanced_feedback(client):
    payload = {
        "rating": 5,
        "ease_of_use": 4,
        "useful_features": "Prediction model",
        "missing_features": "Real-time alerts",
        "general_comments": "Very helpful!"
    }
    response = client.post('/api/feedback/advanced', json=payload)
    assert response.status_code == 201
    assert response.get_json()['message'] == 'Advanced feedback submitted successfully'

def test_submit_advanced_feedback_missing_required(client):
    payload = {
        "useful_features": "Graphs",
        "missing_features": "Backtesting",
        "general_comments": "Needs work"
    }
    response = client.post('/api/feedback/advanced', json=payload)
    assert response.status_code == 400
