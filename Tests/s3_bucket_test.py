# test_s3_utils.py

import pytest
import pickle
import tempfile
import os
from unittest import mock
from services.s3_utils import download_from_s3

@pytest.fixture
def dummy_pickle_file():
    dummy_obj = {'key': 'value'}
    fd, temp_path = tempfile.mkstemp()
    with os.fdopen(fd, 'wb') as f:
        pickle.dump(dummy_obj, f)
    yield temp_path, dummy_obj
    os.remove(temp_path)

def test_download_from_s3(dummy_pickle_file):
    temp_file_path, expected_obj = dummy_pickle_file

    with mock.patch('s3_utils.s3_client.download_file') as mock_download:
        def copy_dummy_file(Bucket, Key, Filename):
            os.replace(temp_file_path, Filename)

        mock_download.side_effect = copy_dummy_file

        result_obj = download_from_s3('dummy-key')

        assert result_obj == expected_obj
