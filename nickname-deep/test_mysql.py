import pymysql
import os
from dotenv import load_dotenv

load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

print(f"[DEBUG] {MYSQL_HOST=}, {MYSQL_USER=}, {MYSQL_PASSWORD=}, {MYSQL_DATABASE=}")

try:
    print("[DEBUG] pymysql.connect 호출 전")
    conn = pymysql.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE
    )
    print("[DEBUG] 연결 성공")
    conn.close()
except Exception as e:
    print(f"[ERROR] 연결 실패: {e}")
