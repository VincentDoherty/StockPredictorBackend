import logging

from flask import Flask
from flask_login import LoginManager

from routes import register_routes

app = Flask(__name__)
app.config.from_object('config.Config')

# logging
logging.basicConfig(filename='app.log', level=logging.DEBUG)

# Login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

register_routes(app, login_manager)

if __name__ == '__main__':
    app.run(debug=True)