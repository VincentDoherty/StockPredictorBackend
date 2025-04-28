import logging
from flask import Flask
from flask_login import LoginManager
from routes import register_routes, stock_routes, portfolio_routes, risk_routes, goal_routes, feedback_routes
from flask_cors import CORS
import os

# Create Flask app
application = Flask(__name__)
application.config.from_object('config.Config')

# logging
logging.basicConfig(filename='app.log', level=logging.DEBUG)

# CORS Setup
frontend_origin = os.getenv('FRONTEND_URL')
CORS(application, resources={r"/*": {"origins": frontend_origin}}, supports_credentials=True)

# Login manager
login_manager = LoginManager()
login_manager.init_app(application)
login_manager.login_view = 'login'

# Register routes
register_routes(application, login_manager)
stock_routes(application)
portfolio_routes(application)
risk_routes(application)
goal_routes(application)
feedback_routes(application)

# No `app.run()` needed for Elastic Beanstalk
