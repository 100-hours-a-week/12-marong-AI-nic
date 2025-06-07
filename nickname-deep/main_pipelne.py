import mysql.connector
from dotenv import load_dotenv
import os
from datetime import datetime

from modules.mysql_loader import fetch_user_mbti_and_hobby
from modules.user_vector import create_user_vector
from modules.candidate_selector import load_candidate_vectors, select_top_candidate
from modules.keyword_extractor import load_traits, get_mbti_keywords, get_hobby_keywords
from modules.prompt_builder import build_prompt
from modules.llm_caller import call_llm
from modules.result_parser import parse_nickname_response
from modules.mysql_saver import fetch_user_group_id, save_anonymous_name
from utils.week_index import GetWeekIndex

# ✅ 환경 변수 로딩 (API 키 등)
load_dotenv()

# ✅ MySQL 연결
conn = mysql.connector.connect(
    host=os.getenv("MYSQL_HOST"),
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    database=os.getenv("MYSQL_DATABASE"),
)
cursor = conn.cursor(dictionary=True)

# ✅ 기준일 및 오늘 날짜 기반 주차 계산
base_date = datetime(2025, 1, 6)
today = datetime.today()
week = GetWeekIndex(today, base_date).get()

# ✅ 처리 대상 사용자 리스트 (예시: Users 테이블에서 전체 가져오기)
cursor.execute("SELECT id FROM Users")
user_ids = [row["id"] for row in cursor.fetchall()]

# ✅ 후보 벡터 및 traits 로드
candidates = load_candidate_vectors()
traits = load_traits()

for user_id in user_ids:
    scores, hobby = fetch_user_mbti_and_hobby(cursor, user_id)
    if not scores or not hobby:
        print(f"⚠️ user_id={user_id} → MBTI or Hobby 누락 → 스킵")
        continue

    user_vector = create_user_vector(scores, hobby)
    mbti, matched_hobby = select_top_candidate(user_vector, candidates)

    mbti_keywords = get_mbti_keywords(mbti, scores, traits)
    hobby_keywords = get_hobby_keywords(matched_hobby, traits)
    prompt = build_prompt(mbti_keywords, hobby_keywords)
    llm_output = call_llm(prompt)
    nicknames = parse_nickname_response(llm_output)

    if not nicknames:
        print(f"⚠️ user_id={user_id} → 별명 생성 실패")
        continue

    final_nickname = nicknames[0]  # 기본적으로 첫 번째 추천 사용
    group_id = fetch_user_group_id(cursor, user_id)

    if not group_id:
        print(f"⚠️ user_id={user_id} → group_id 없음")
        continue

    save_anonymous_name(cursor, user_id, group_id, week, final_nickname)
    print(f"✅ user_id={user_id} → 저장된 별명: {final_nickname}")

conn.commit()
cursor.close()
conn.close()
print("🎉 전체 nickname 생성 완료")
