from services.db_utils import get_db_connection
from user import User

def get_user_by_id(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, password_hash FROM users WHERE id = %s", (user_id,))
    user_data = cursor.fetchone()
    cursor.close()
    conn.close()
    if user_data:
        return User(user_data[0], user_data[1], user_data[2])
    return None

def get_user_by_username(username):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, password_hash FROM users WHERE username = %s", (username,))
    user_data = cursor.fetchone()
    cursor.close()
    conn.close()
    if user_data:
        return User(user_data[0], user_data[1], user_data[2])
    return None

def create_user(username, password_hash):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
        (username, password_hash)
    )
    conn.commit()
    cursor.close()
    conn.close()