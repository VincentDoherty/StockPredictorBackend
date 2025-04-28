
from functools import wraps
from flask import request, jsonify
from flask_login import current_user
from services.db_utils import get_db_connection
import logging

def require_ownership(table_name: str, id_field: str = 'id', url_param: str = 'portfolio_id'):
    """
    Decorator to ensure the current user owns the object being accessed.

    Args:
        table_name (str): Name of the table to query.
        id_field (str): Field name in DB to match the ID. Defaults to 'id'.
        url_param (str): Name of the URL parameter to pull the ID from. Defaults to 'portfolio_id'.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            object_id = kwargs.get(url_param)
            if object_id is None:
                return jsonify({'error': f'Missing required URL parameter: {url_param}'}), 400

            conn = get_db_connection()
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        f"SELECT {id_field} FROM {table_name} WHERE {id_field} = %s AND user_id = %s",
                        (object_id, current_user.id)
                    )
                    result = cursor.fetchone()
                    if not result:
                        return jsonify({'error': 'Unauthorized access'}), 403
            except Exception as e:
                logging.error(f"Ownership check failed for {table_name}: {e}", exc_info=True)
                return jsonify({'error': 'Internal Server Error during ownership check'}), 500
            finally:
                conn.close()

                return f(*args, **kwargs)

        return decorated_function
    return decorator
