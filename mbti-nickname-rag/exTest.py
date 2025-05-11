# test_fetch_users.py

import os
import sys
from pathlib import Path
import mysql.connector
from dotenv import load_dotenv

# ── 1) mbti-nickname-rag 모듈 경로를 PYTHONPATH에 추가 ─────────────────
BASE_DIR   = Path(__file__).resolve().parent
MODULE_DIR = BASE_DIR / "mbti-nickname-rag"
sys.path.append(str(MODULE_DIR))

# ── 2) fetch_target_users 임포트 ───────────────────────────────────────
from Test import fetch_target_users  # Test.py 안에 fetch_target_users 정의되어 있어야 합니다

# ── 3) DB 연결
load_dotenv()
conn = mysql.connector.connect(
    host=os.getenv("MYSQL_HOST"),
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    database=os.getenv("MYSQL_DB")
)
cursor = conn.cursor(dictionary=True)

# ── 4) 함수 호출 & 결과 확인
users = fetch_target_users(cursor)
print(f"✅ 총 대상 사용자 수: {len(users)}명")
print("샘플 최대 5명:")
for u in users[:5]:
    # u는 dict: {'user_id':..., 'group_id':..., 'mbti':..., 'hobby':..., 'week':...}
    print(u)

# ── 5) 정리
cursor.close()
conn.close()
