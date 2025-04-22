import pytest
from app import app as flask_app
from services.db_utils import get_db_connection
from unittest.mock import patch

class MockAdminUser:
    def __init__(self):
        self.id = 1
        self.is_authenticated = True
        self.is_admin = True

    def get_id(self):
        return str(self.id)

@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        with client.session_transaction() as sess:
            sess['_user_id'] = '1'
        yield client

@patch('user_db.get_user_by_id', return_value=MockAdminUser())
def test_get_all_basic_feedback(mock_user, client):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("INSERT INTO feedback_basic (comment) VALUES (%s)", ("Test comment",))
        conn.commit()
    finally:
        conn.close()

    response = client.get('/api/feedback/all')
    assert response.status_code == 200
    data = response.get_json()
    assert 'basic_feedback' in data
    assert any('comment' in item for item in data['basic_feedback'])

@patch('user_db.get_user_by_id', return_value=MockAdminUser())
def test_get_all_feedback(mock_user, client):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("INSERT INTO feedback_basic (comment) VALUES (%s)", ("Test comment",))
            cursor.execute("""
                INSERT INTO feedback_advanced (rating, ease_of_use, useful_features, missing_features, general_comments)
                VALUES (%s, %s, %s, %s, %s)
            """, (5, 4, "Graphs", "Notifications", "Great UX"))
        conn.commit()
    finally:
        conn.close()

    response = client.get('/api/feedback/all')
    assert response.status_code == 200
    data = response.get_json()
    assert 'basic_feedback' in data
    assert 'advanced_feedback' in data
