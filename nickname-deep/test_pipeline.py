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
import json

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
        
        # 누락된 데이터 분석
        print("\n[debug] 누락된 데이터 분석 중...")
        cursor.execute("""
            SELECT 
                u.id as user_id,
                CASE WHEN sm.id IS NULL THEN 'MBTI 없음' ELSE 'MBTI 있음' END as mbti_status,
                CASE WHEN sh.id IS NULL THEN '취미 없음' ELSE '취미 있음' END as hobby_status
            FROM Users u 
            JOIN UserGroups ug ON u.id = ug.user_id
            JOIN `Groups` g ON ug.group_id = g.id 
            LEFT JOIN SurveyMBTI sm ON u.id = sm.user_id
            LEFT JOIN SurveyHobby sh ON u.id = sh.user_id
            WHERE g.id IS NOT NULL
            ORDER BY u.id
        """)
        missing_data = cursor.fetchall()
        
        mbti_missing = sum(1 for row in missing_data if row['mbti_status'] == 'MBTI 없음')
        hobby_missing = sum(1 for row in missing_data if row['hobby_status'] == '취미 없음')
        both_missing = sum(1 for row in missing_data if row['mbti_status'] == 'MBTI 없음' and row['hobby_status'] == '취미 없음')
        
        print(f"[debug] MBTI 정보 누락: {mbti_missing}명")
        print(f"[debug] 취미 정보 누락: {hobby_missing}명")
        print(f"[debug] 둘 다 누락: {both_missing}명")
        
    except pymysql.MySQLError as err:
        print(f"[error] 그룹별 사용자 조회 실패: {err}")
        sys.exit(1)

    # ✅ 후보 벡터 및 traits 로드
    try:
        print("[debug] 후보 벡터 로드 시도...")
        candidates = load_candidate_vectors("data/candidate_vectors01.pkl")
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
    group_success_count = 0  # 그룹별 성공 카운트 초기화
    total_users_in_group = 0  # 그룹별 전체 사용자 수 초기화
    
    # 전체 통계 추가
    total_users = len(user_group_mapping)
    total_skipped = 0
    skipped_reasons = {
        "mbti_hobby_missing": 0,
        "vector_creation_error": 0,
        "candidate_selection_error": 0,
        "keyword_extraction_error": 0,
        "nickname_generation_error": 0,
        "save_error": 0
    }
    
    # 그룹별 실패 user_id 저장
    skipped_user_ids = []

    # 실패한 닉네임 조합을 저장할 리스트
    # 각 항목은 {'user_id': ..., 'mbti_keywords': [...], 'hobby_keywords': [...], 'error': ...} 형태로 저장됨
    failed_combinations = []

    print(f"[debug] 전체 처리 대상 사용자 수: {total_users}")

    for mapping in user_group_mapping:
        user_id = mapping['user_id']
        group_id = mapping['group_id']
        
        # 그룹이 바뀌면 로그 출력 및 카운터 초기화
        if current_group_id != group_id:
            if current_group_id is not None:
                print(f"\n[info] 그룹 {current_group_id} 처리 완료 (성공: {group_success_count}/{total_users_in_group})")
                if skipped_user_ids:
                    print(f"[info] 그룹 {current_group_id}에서 실패한 user_id 목록: {skipped_user_ids}")
            print(f"\n==========================================")
            print(f"[info] 그룹 {group_id} 처리 시작")
            print(f"==========================================")
            current_group_id = group_id
            group_success_count = 0
            total_users_in_group = 0
            skipped_user_ids = []

        total_users_in_group += 1
        print(f"\n[info] user_id={user_id} 처리 시작")
        print("----- 사용자 처리 시작 -----")

        # ✅ Step 1: 성향 + 취미 조회
        try:
            scores, hobby = fetch_user_mbti_and_hobby(cursor, user_id)
            if not scores or not hobby or not isinstance(hobby, str) or hobby.strip() == "":
                print(f"[warn] user_id={user_id} → 성향 또는 취미 정보 누락 → 건너뜀")
                total_skipped += 1
                skipped_reasons["mbti_hobby_missing"] += 1
                skipped_user_ids.append(user_id)
                continue
        except Exception as e:
            print(f"[error] user_id={user_id} → 성향/취미 조회 오류: {e}")
            total_skipped += 1
            skipped_reasons["mbti_hobby_missing"] += 1
            skipped_user_ids.append(user_id)
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
            total_skipped += 1
            skipped_reasons["vector_creation_error"] += 1
            skipped_user_ids.append(user_id)
            continue

        # ✅ Step 3: 유사도 기반 조합 추천
        try:
            mbti, matched_hobby = select_top_candidate(user_vector, candidates)
            print(f"[step3] 추천 조합 → MBTI={mbti}, Hobby={matched_hobby}")
        except Exception as e:
            print(f"[error] 유사도 기반 추천 오류: {e}")
            total_skipped += 1
            skipped_reasons["candidate_selection_error"] += 1
            skipped_user_ids.append(user_id)
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
            total_skipped += 1
            skipped_reasons["keyword_extraction_error"] += 1
            skipped_user_ids.append(user_id)
            continue

        # ✅ Step 5: 별명 생성
        try:
            # 별명 생성 시 입력 키워드 로그
            print(f"[debug] 별명 생성 입력 키워드: MBTI={mbti_keywords}, Hobby={hobby_keywords}")
            final_nickname = generate_unique_nickname(cursor, mbti_keywords, hobby_keywords, group_id)
            print("[step5] 유일 별명 생성 완료:", final_nickname)
        except Exception as e:
            print(f"[step5] 별명 생성 실패: {e}")
            print(f"[debug] 실패 user_id: {user_id}")
            print(f"[debug] 입력 키워드: MBTI={mbti_keywords}, Hobby={hobby_keywords}")
            # 실패한 조합을 리스트에 저장
            failed_combinations.append({
                'user_id': user_id,
                'mbti_keywords': mbti_keywords,
                'hobby_keywords': hobby_keywords,
                'error': str(e)
            })
            total_skipped += 1
            skipped_reasons["nickname_generation_error"] += 1
            skipped_user_ids.append(user_id)
            continue

        # ✅ Step 6: 별명 저장
        try:
            week = GetWeekIndex(datetime.today(), datetime(2025, 1, 6)).get()
            save_anonymous_name(cursor, user_id, group_id, week, final_nickname)
            conn.commit()
            group_success_count += 1  # 성공 시 카운트 증가
            print(f"[step6] 별명 저장 완료: {final_nickname}")
        except Exception as e:
            print(f"[error] 별명 저장 실패: {e}")
            total_skipped += 1
            skipped_reasons["save_error"] += 1
            skipped_user_ids.append(user_id)
            continue
        print("----- 사용자 처리 완료 -----")

    # 마지막 그룹의 처리 결과 출력
    if current_group_id is not None:
        print(f"\n[info] 그룹 {current_group_id} 처리 완료 (성공: {group_success_count}/{total_users_in_group})")
        if skipped_user_ids:
            print(f"[info] 그룹 {current_group_id}에서 실패한 user_id 목록: {skipped_user_ids}")

    # 전체 통계 출력
    print(f"\n============================")
    print(f"[통계] 전체 처리 결과")
    print(f"전체 사용자 수: {total_users}")
    print(f"성공한 사용자 수: {total_users - total_skipped}")
    print(f"건너뛴 사용자 수: {total_skipped}")
    print(f"성공률: {((total_users - total_skipped) / total_users * 100):.1f}%")
    print(f"\n건너뛴 이유별 통계:")
    for reason, count in skipped_reasons.items():
        if count > 0:
            print(f"  - {reason}: {count}명")

    # ★ 닉네임 생성 실패 조합을 한 번에 출력 ★
    # 실패한 user_id, MBTI 키워드, 취미 키워드, 에러 메시지를 모두 출력
    if failed_combinations:
        print("\n[닉네임 생성 실패 조합 목록]")
        for item in failed_combinations:
            print(f"user_id: {item['user_id']}, MBTI: {item['mbti_keywords']}, 취미: {item['hobby_keywords']}, 에러: {item['error']}")
        
        # 실패 조합을 JSON 파일로 저장
        try:
            with open('failed_combinations.json', 'w', encoding='utf-8') as f:
                json.dump(failed_combinations, f, ensure_ascii=False, indent=2)
            print("\n[info] 실패 조합이 'failed_combinations.json' 파일로 저장되었습니다.")
        except Exception as e:
            print(f"[error] 실패 조합 저장 실패: {e}")
        
        # OpenAI 폴백으로 실패한 닉네임 재시도
        print("\n" + "="*60)
        print("🔄 OpenAI API로 실패한 닉네임 재시도 시작")
        print("="*60)
        
        try:
            from modules.openai_fallback import retry_failed_nicknames_with_openai
            openai_result = retry_failed_nicknames_with_openai(failed_combinations, cursor)
            
            # 최종 통계 업데이트
            total_success_with_openai = total_users - total_skipped + openai_result["success"]
            final_failed = openai_result["failed"]
            
            print(f"\n🎯 최종 결과 (OpenAI 폴백 포함)")
            print(f"총 사용자 수: {total_users}")
            print(f"기존 성공: {total_users - total_skipped}")
            print(f"OpenAI로 추가 성공: {openai_result['success']}")
            print(f"최종 성공: {total_success_with_openai}")
            print(f"최종 실패: {final_failed}")
            print(f"최종 성공률: {total_success_with_openai/total_users*100:.1f}%")
            
        except ImportError:
            print("[OpenAI] openai 모듈이 설치되지 않았습니다. 'pip install openai'로 설치하세요.")
        except Exception as e:
            print(f"[OpenAI] 폴백 처리 중 오류 발생: {e}")
        
    else:
        print("\n[닉네임 생성 실패 조합 없음]")

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
