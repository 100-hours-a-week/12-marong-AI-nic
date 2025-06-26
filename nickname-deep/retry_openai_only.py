import json
from modules.openai_fallback import retry_failed_nicknames_with_openai
import pymysql
import os
from dotenv import load_dotenv
import time

print("=== retry_openai_only.py 시작 ===")
print("현재 경로:", os.getcwd())
print("실패 조합 파일 존재:", os.path.exists('failed_combinations.json'))

# 환경 변수 로드 (DB 정보 등)
load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "marong")

print("DB 연결 정보:")
print("  HOST:", MYSQL_HOST)
print("  USER:", MYSQL_USER)
print("  PASSWORD:", MYSQL_PASSWORD)
print("  NAME:", MYSQL_DATABASE)

try:
    print("DB 연결 시도...")
    conn = pymysql.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        charset='utf8mb4'
    )
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    print("[info] DB 연결 성공")
except Exception as e:
    print("[DB 연결 실패]", e)
    import traceback
    traceback.print_exc()
    print("10초 후 종료됩니다.")
    time.sleep(10)
    exit(1)

# 실패 조합 불러오기
try:
    with open('failed_combinations.json', 'r', encoding='utf-8') as f:
        failed_combinations = json.load(f)
    print("불러온 실패 조합 개수:", len(failed_combinations))
    if failed_combinations:
        print("첫 번째 실패 조합:", failed_combinations[0])
except Exception as e:
    print("[실패 조합 파일 읽기 실패]", e)
    exit(1)

if not failed_combinations:
    print("실패 조합이 없습니다. 종료합니다.")
    exit(0)

# OpenAI로만 재시도
try:
    print("[info] OpenAI 폴백 함수 진입")
    result = retry_failed_nicknames_with_openai(failed_combinations, cursor)
    print(f"\n[OpenAI Only] 최종 결과: {result}")
except Exception as e:
    import traceback
    print("[OpenAI 폴백 중 오류]", e)
    traceback.print_exc()
finally:
    try:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    except:
        pass