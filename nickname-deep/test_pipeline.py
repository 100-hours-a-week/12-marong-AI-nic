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
from datetime import datetime
import pymysql  
import os
import sys
from collections import defaultdict

# ✅ 환경 변수 로드
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ✅ MySQL 환경 변수 로드 및 디버깅
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

print(f"[debug] MYSQL_HOST={MYSQL_HOST}, USER={MYSQL_USER}, DB={MYSQL_DATABASE}")

if not all([MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE]):
    raise EnvironmentError("환경 변수(.env)가 일부 누락되었습니다. .env 파일을 확인하세요.")

# ✅ Hugging Face 모델 로드 (transformers 기반)
model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_id, token=HF_TOKEN)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def call_llm(prompt: str) -> str:
    result = generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
    return result[0]["generated_text"]

try:
    print("[debug] MySQL 연결 시도 중...")
    # ✅ MySQL 연결 (pymysql 버전)
    try:
        conn = pymysql.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            charset="utf8mb4"
        )
        print("[debug] MySQL 연결 성공")
    except pymysql.MySQLError as err:
        print(f"[error] MySQL 연결 실패: {err}")
        print(f"[debug] 연결 정보: host={MYSQL_HOST}, user={MYSQL_USER}, database={MYSQL_DATABASE}")
        sys.exit(1)

    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        print("[debug] 커서 생성 성공")
    except pymysql.MySQLError as err:
        print(f"[error] 커서 생성 실패: {err}")
        sys.exit(1)

    # ✅ 그룹별 사용자 정보 수집
    try:
        print("[debug] 그룹별 사용자 정보 조회 시도...")
        cursor.execute("""
            SELECT u.id as user_id, g.id as group_id 
            FROM Users u 
            JOIN UserGroups ug ON u.id = ug.user_id
            JOIN `Groups` g ON ug.group_id = g.id 
            WHERE g.id IS NOT NULL
            ORDER BY g.id, u.id
        """)
        user_group_mapping = cursor.fetchall()
        print(f"[debug] 조회된 사용자-그룹 매핑 수: {len(user_group_mapping)}")
    except pymysql.MySQLError as err:
        print(f"[error] 그룹별 사용자 조회 실패: {err}")
        sys.exit(1)

    # ✅ 후보 벡터 및 traits 로드
    try:
        print("[debug] 후보 벡터 로드 시도...")
        candidates = load_candidate_vectors("data/candidate_vectors.pkl")
        print("[debug] 후보 벡터 로드 성공")
    except Exception as e:
        print(f"[error] 후보 벡터 로드 실패: {e}")
        sys.exit(1)

    try:
        print("[debug] traits 로드 시도...")
        traits = load_traits("data/mbti_traits_with_binary_vector.json", "data/hobby_traits.json")
        print("[debug] traits 로드 성공")
    except Exception as e:
        print(f"[error] traits 로드 실패: {e}")
        sys.exit(1)

    # 사용자별로 처리
    current_group_id = None
    for mapping in user_group_mapping:
        user_id = mapping['user_id']
        group_id = mapping['group_id']
        
        # 그룹이 바뀌면 로그 출력
        if current_group_id != group_id:
            print(f"\n============================")
            print(f"[info] 그룹 {group_id} 처리 시작")
            current_group_id = group_id

        print(f"[info] user_id={user_id} 처리 시작")

        # ✅ Step 1: 성향 + 취미 조회
        try:
            scores, hobby = fetch_user_mbti_and_hobby(cursor, user_id)
            if not scores or not hobby or not isinstance(hobby, str) or hobby.strip() == "":
                print(f"[warn] user_id={user_id} → 성향 또는 취미 정보 누락 → 건너뜀")
                continue
        except Exception as e:
            print(f"[error] user_id={user_id} → 성향/취미 조회 오류: {e}")
            continue

        print("[step1] 사용자 성향 및 취미 로드 완료")
        print("MBTI 점수:", scores)
        print("취미:", hobby)

        # ✅ Step 2: 사용자 벡터 생성
        try:
            user_vector = create_user_vector(scores, hobby)
            print("[step2] 사용자 벡터 생성 완료:", user_vector)
        except Exception as e:
            print(f"[error] 사용자 벡터 생성 오류: {e}")
            continue

        # ✅ Step 3: 유사도 기반 조합 추천
        try:
            mbti, matched_hobby = select_top_candidate(user_vector, candidates)
            print(f"[step3] 추천 조합 → MBTI={mbti}, Hobby={matched_hobby}")
        except Exception as e:
            print(f"[error] 유사도 기반 추천 오류: {e}")
            continue

        # ✅ Step 4: 키워드 추출
        try:
            mbti_keywords = get_mbti_keywords(mbti, scores, traits)
            hobby_keywords = get_hobby_keywords(matched_hobby, traits)
            print("[step4] 키워드 추출 완료")
            print("MBTI 키워드:", mbti_keywords)
            print("Hobby 키워드:", hobby_keywords)
        except Exception as e:
            print(f"[error] 키워드 추출 오류: {e}")
            continue

        # ✅ Step 5: 별명 생성
        try:
            final_nickname = generate_unique_nickname(cursor, mbti_keywords, hobby_keywords, group_id)
            print("[step5] 유일 별명 생성 완료:", final_nickname)
        except Exception as e:
            print(f"[step5] 별명 생성 실패: {e}")
            continue

        # ✅ Step 6: 별명 저장
        try:
            week = GetWeekIndex(datetime.today(), datetime(2025, 1, 6)).get()
            save_anonymous_name(cursor, user_id, group_id, week, final_nickname)
            conn.commit()
            print(f"[step6] 별명 저장 완료: {final_nickname}")
        except Exception as e:
            print(f"[error] 별명 저장 실패: {e}")
            continue

        conn.commit()
        print(f"[step6] 그룹 {group_id} 별명 저장 완료: {final_nickname} (성공: {success_count}/{len(user_ids)})")

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

print("\n🎉 전체 nickname 생성 완료")
