# nickname_generator.py (유지)

from modules.llm_caller import call_llm
from modules.result_parser import parse_nickname_response
import random
import time
from modules.prompt_builder import build_prompt
from datetime import datetime, timedelta

# 전역 캐시 변수
_group_nickname_cache = {}  # group_id를 키로 하는 딕셔너리
_last_cache_update = None
_CACHE_TTL = timedelta(minutes=5)  # 캐시 유효 시간

def _should_refresh_cache() -> bool:
    """캐시를 갱신해야 하는지 확인"""
    global _last_cache_update
    if _last_cache_update is None:
        return True
    return datetime.now() - _last_cache_update > _CACHE_TTL

def fetch_group_used_nicknames(cursor, group_id) -> set:
    """특정 그룹에서 사용된 닉네임을 가져와 캐시를 갱신"""
    global _group_nickname_cache, _last_cache_update
    
    if _should_refresh_cache():
        # 모든 그룹의 닉네임을 한 번에 가져옴
        cursor.execute("""
            SELECT g.id as group_id, an.anonymous_name 
            FROM `Groups` g 
            JOIN AnonymousNames an ON g.id = an.group_id
            WHERE an.week = (SELECT MAX(week) FROM AnonymousNames)
        """)
        rows = cursor.fetchall()
        
        # 그룹별로 닉네임을 분류
        _group_nickname_cache = {}
        for row in rows:
            group_id = row['group_id']
            if group_id not in _group_nickname_cache:
                _group_nickname_cache[group_id] = set()
            _group_nickname_cache[group_id].add(row['anonymous_name'])
        
        _last_cache_update = datetime.now()
        print(f"[info] 그룹별 닉네임 캐시 갱신 완료: {len(_group_nickname_cache)}개 그룹")
    
    # 요청된 그룹의 닉네임 반환 (없으면 빈 set 반환)
    return _group_nickname_cache.get(group_id, set())

def normalize_nickname(name):
    # 띄어쓰기 제거, 소문자 변환 등 필요시 추가
    return name.replace(" ", "").lower()

def generate_unique_nickname(cursor, mbti_keywords, hobby_keywords, group_id, max_retries=8) -> str:
    used_nicknames = fetch_group_used_nicknames(cursor, group_id)
    print(f"[debug] 그룹 {group_id}의 현재 사용 중인 닉네임 수: {len(used_nicknames)}")

    # 중복 체크를 위한 닉네임 정규화
    normalized_used = set(normalize_nickname(n) for n in used_nicknames)

    for attempt in range(max_retries):
        # 매 시도마다 MBTI와 Hobby 키워드에서 각각 4개씩 랜덤 선택
        selected_mbti_keywords = random.sample(mbti_keywords, min(4, len(mbti_keywords)))
        selected_hobby_keywords = random.sample(hobby_keywords, min(4, len(hobby_keywords)))
        
        print(f"[debug] 시도 {attempt+1}/{max_retries} - 선택된 키워드")
        print(f"MBTI 키워드: {selected_mbti_keywords}")
        print(f"Hobby 키워드: {selected_hobby_keywords}")

        prompt = build_prompt(selected_mbti_keywords, selected_hobby_keywords)

        try:
            response = call_llm(prompt)
            candidates = parse_nickname_response(response)
            print(f"[debug] 생성된 닉네임 후보: {candidates}")
            
            if not candidates:
                print(f"[warn] LLM이 닉네임 후보를 생성하지 못했습니다. (시도 {attempt+1}/{max_retries})")
                continue
                
            random.shuffle(candidates)

            for name in candidates:
                if normalize_nickname(name) not in normalized_used:
                    print(f"[info] 선택된 닉네임: {name}")
                    return name
                else:
                    print(f"[debug] 그룹 내 중복된 닉네임 발견: {name}")
                    
            print(f"[warn] 모든 후보가 그룹 내에서 중복입니다. (시도 {attempt+1}/{max_retries})")
            
        except Exception as e:
            print(f"[warn] LLM 응답 오류 (재시도 {attempt+1}/{max_retries}): {e}")

        time.sleep(1)

    raise RuntimeError(f"그룹 {group_id}에서 유일한 별명 생성 실패: 중복 회피 불가")
