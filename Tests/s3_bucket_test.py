import unittest
from services.db_utils import download_model_from_s3


class TestS3BucketDownload(unittest.TestCase):
    def test_download_model_from_s3(self):
        # Replace with a valid S3 URL for testing
        s3_url = "https://stock-prediction-models-fyp.s3.eu-north-1.amazonaws.com/models/AAPL.pkl"

        # Attempt to download the model
        model = download_model_from_s3(s3_url)

        # Assert that the model is not None
        self.assertIsNotNone(model, "Model should be successfully downloaded from S3.")


if __name__ == '__main__':
    unittest.main()