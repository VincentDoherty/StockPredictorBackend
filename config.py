import os

class Config:
    SECRET_KEY = os.urandom(24)
    DB_NAME = 'investmentdb'
    DB_USER = 'postgres'
    DB_PASSWORD = 'vdonkeY800'
    DB_HOST = 'localhost'
    DB_PORT = '5432'