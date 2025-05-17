#ex_mysql_connector

import mysql.connector
from dotenv import load_dotenv
import os

# ✅ .env 파일 로드
load_dotenv()

def get_mysql_connection():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE")
    )

def fetch_user_mbti_and_hobby(cursor, user_id: int):
    query = "SELECT mbti, hobby FROM Users WHERE id = %s"
    cursor.execute(query, (user_id,))
    result = cursor.fetchone()
    if result:
        return result[0], result[1]
    else:
        return None