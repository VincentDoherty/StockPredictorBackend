import logging
from dotenv import load_dotenv
from flask import Flask
from flask_login import LoginManager
from routes import register_routes, stock_routes, portfolio_routes, risk_routes, goal_routes, feedback_routes
from flask_cors import CORS
import os

load_dotenv()

# Create Flask app
app = Flask(__name__)
app.config.from_object('config.Config')

# logging
logging.basicConfig(filename='app.log', level=logging.DEBUG)

# CORS Setup
frontend_origin = os.getenv('FRONTEND_URL')
CORS(app, resources={r"/*": {"origins": frontend_origin}}, supports_credentials=True)

# Login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Register routes
register_routes(app, login_manager)
stock_routes(app)
portfolio_routes(app)
risk_routes(app)
goal_routes(app)
feedback_routes(app)

# No `app.run()` needed for Elastic Beanstalk
