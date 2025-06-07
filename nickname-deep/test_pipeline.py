# test_pipeline.py (중복 회피 + 무작위 선택 적용)

from modules.mysql_loader import fetch_user_mbti_and_hobby
from modules.user_vector import create_user_vector
from modules.candidate_selector import load_candidate_vectors, select_top_candidate
from modules.keyword_extractor import load_traits, get_mbti_keywords, get_hobby_keywords
from modules.prompt_builder import build_prompt
from modules.mysql_saver import fetch_user_group_id, save_anonymous_name
from modules.nickname_generator import generate_unique_nickname
from utils.week_index import GetWeekIndex

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import mysql.connector
from datetime import datetime
import os

# ✅ 환경 변수 로드
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ✅ Hugging Face 모델 로드 (transformers 기반)
model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=HF_TOKEN)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def call_llm(prompt: str) -> str:
    result = generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
    return result[0]["generated_text"]

try:
    # ✅ MySQL 연결
    conn = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE"),
    )
    cursor = conn.cursor(dictionary=True)
    print("[info] MySQL 연결 성공")

    # ✅ 사용자 전체 조회
    cursor.execute("SELECT id FROM Users")
    user_ids = [row["id"] for row in cursor.fetchall()]

    # ✅ 후보 벡터 및 traits 로드
    candidates = load_candidate_vectors("data/candidate_vectors.pkl")
    traits = load_traits("data/mbti_traits_with_binary_vector.json", "data/hobby_traits.json")

    for user_id in user_ids:
        print(f"\n============================")
        print(f"[info] user_id={user_id} 처리 시작")

        # ✅ Step 1: 성향 + 취미 조회
        scores, hobby = fetch_user_mbti_and_hobby(cursor, user_id)
        if not scores or not hobby:
            print(f"[warn] user_id={user_id} → 성향 또는 취미 정보 누락")
            continue

        print("[step1] 사용자 성향 및 취미 로드 완료")
        print("MBTI 점수:", scores)
        print("취미:", hobby)

        # ✅ Step 2: 사용자 벡터 생성
        user_vector = create_user_vector(scores, hobby)
        print("[step2] 사용자 벡터 생성 완료:", user_vector)

        # ✅ Step 3: 유사도 기반 조합 추천
        mbti, matched_hobby = select_top_candidate(user_vector, candidates)
        print(f"[step3] 추천 조합 → MBTI={mbti}, Hobby={matched_hobby}")

        # ✅ Step 4: 키워드 추출
        mbti_keywords = get_mbti_keywords(mbti, scores, traits)
        hobby_keywords = get_hobby_keywords(matched_hobby, traits)
        print("[step4] 키워드 추출 완료")
        print("MBTI 키워드:", mbti_keywords)
        print("Hobby 키워드:", hobby_keywords)

        # ✅ Step 5: 별명 생성 (중복 회피 포함)
        try:
            final_nickname = generate_unique_nickname(cursor, mbti_keywords, hobby_keywords)
            print("[step5] 유일 별명 생성 완료:", final_nickname)
        except Exception as e:
            print(f"[step5] 별명 생성 실패: {e}")
            continue

        # ✅ Step 6: 그룹 ID 조회
        group_id = fetch_user_group_id(cursor, user_id)
        print("[step6] 사용자 그룹 ID:", group_id)

        # ✅ Step 7: 별명 저장
        week = GetWeekIndex(datetime.today(), datetime(2025, 1, 6)).get()
        save_anonymous_name(cursor, user_id, group_id, week, final_nickname)
        conn.commit()
        print(f"[step7] 별명 저장 완료: {final_nickname}")

except mysql.connector.Error as db_err:
    print(f"[error] MySQL 연결 또는 쿼리 오류: {db_err}")

except Exception as e:
    print(f"[error] 처리 중 오류 발생: {e}")

finally:
    try:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    except:
        pass
