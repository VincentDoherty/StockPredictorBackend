# s3_utils.py

import pickle
import boto3
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

# --- S3 Setup ---
S3_BUCKET = os.getenv('S3_BUCKET')
S3_REGION = 'eu-north-1'

# Corrected keys to match .env
S3_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
S3_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

s3_client = boto3.client(
    's3',
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    region_name=S3_REGION
)

def upload_to_s3(obj, key):
    fd, temp_path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, 'wb') as f:
            pickle.dump(obj, f)
        s3_client.upload_file(temp_path, S3_BUCKET, key)
    finally:
        os.remove(temp_path)
    return f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{key}"

def download_from_s3(key):
    fd, temp_path = tempfile.mkstemp()
    os.close(fd)
    s3_client.download_file(S3_BUCKET, key, temp_path)
    with open(temp_path, 'rb') as f:
        obj = pickle.load(f)
    os.remove(temp_path)
    return obj
