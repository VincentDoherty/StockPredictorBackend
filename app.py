import logging
from flask import Flask
from flask_login import LoginManager
from routes import register_routes, stock_routes, portfolio_routes, risk_routes, goal_routes, feedback_routes
from flask_cors import CORS

app = Flask(__name__)
app.config.from_object('config.Config')
# logging
logging.basicConfig(filename='app.log', level=logging.DEBUG)
# Enable CORS with credentials and specific origin
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}}, supports_credentials=True)
# Login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

register_routes(app, login_manager)
stock_routes(app)
portfolio_routes(app)
risk_routes(app)
goal_routes(app)
feedback_routes(app)

if __name__ == '__main__':
    app.run(debug=True)