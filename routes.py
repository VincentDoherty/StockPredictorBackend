from flask import request, jsonify
from flask_login import login_user, login_required, logout_user
from werkzeug.security import generate_password_hash
from user_db import get_user_by_id, get_user_by_username, create_user
import logging

def register_routes(app, login_manager):

    @login_manager.user_loader
    def load_user(user_id):
        return get_user_by_id(user_id)

    @app.route('/api/register', methods=['POST'])
    def register():
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        if not username or not password:
            return jsonify({'error': 'Username and password are required'}), 400

        password_hash = generate_password_hash(password)
        try:
            create_user(username, password_hash)
            return jsonify({'message': 'User registered successfully'}), 201
        except Exception as e:
            logging.error(f"Error during registration: {e}")
            return jsonify({'error': 'An error occurred during registration'}), 500
    @app.route('/api/login', methods=['POST'])
    def login():
        try:
            data = request.get_json()
            username = data['username']
            password = data['password']
            user = get_user_by_username(username)
            if user and user.check_password(password):
                login_user(user)
                return jsonify({'message': 'Login successful'}), 200
            return jsonify({'message': 'Invalid username or password'}), 401
        except Exception as e:
            logging.error(f"Error during login: {e}")
            return jsonify({'error': 'An error occurred during login'}), 500

    @app.route('/api/logout', methods=['POST'])
    @login_required
    def logout():
        logout_user()
        return jsonify({'message': 'Logged out successfully'}), 200