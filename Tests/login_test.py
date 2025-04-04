import unittest
from werkzeug.security import generate_password_hash
from app import app
from db_utils import get_db_connection

class AuthTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

        # Create a test user in the database
        self.username = 'testuser'
        self.password = 'testpassword'
        self.create_test_user()

    def tearDown(self):
        # Remove the test user from the database
        self.delete_test_user()

    def create_test_user(self):
        password_hash = generate_password_hash(self.password)
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (%s, %s) ON CONFLICT (username) DO NOTHING",
            (self.username, password_hash)
        )
        conn.commit()
        cursor.close()
        conn.close()

    def delete_test_user(self, username=None):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE username = %s", (username or self.username,))
        conn.commit()
        cursor.close()
        conn.close()

    def test_register(self):
        response = self.app.post('/api/register', json={
            'username': 'newuser',
            'password': 'newpassword'
        })
        self.assertEqual(response.status_code, 201)
        self.assertIn('User registered successfully', response.get_data(as_text=True))
        self.delete_test_user(username='newuser')

    def test_login(self):
        response = self.app.post('/api/login', json={
            'username': self.username,
            'password': self.password
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('Login successful', response.get_data(as_text=True))

    def test_logout(self):
        # First, log in the user
        self.app.post('/api/login', json={
            'username': self.username,
            'password': self.password
        })
        # Then, log out the user
        response = self.app.post('/api/logout')
        self.assertEqual(response.status_code, 200)
        self.assertIn('Logged out successfully', response.get_data(as_text=True))

if __name__ == '__main__':
    unittest.main()