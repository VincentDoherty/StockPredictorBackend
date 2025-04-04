import unittest
from app import app

class StocksApiTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_post_stock(self):
        response = self.app.post('/api/stock', json={'stock_symbol': 'GOOGL'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('application/json', response.content_type)
        data = response.get_json()
        self.assertIsInstance(data, dict)
        self.assertIn('stock_symbol', data)
        self.assertIn('actual_prices', data)
        self.assertIn('predicted_prices', data)

if __name__ == '__main__':
    unittest.main()