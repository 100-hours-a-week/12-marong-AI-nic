# nickname_generator.py (유지)

from modules.llm_caller import call_llm
from modules.result_parser import parse_nickname_response
import random
import time
from modules.prompt_builder import build_prompt
from datetime import datetime, timedelta

# 전역 캐시 변수
_nickname_cache = set()
_last_cache_update = None
_CACHE_TTL = timedelta(minutes=5)  # 캐시 유효 시간

def _should_refresh_cache() -> bool:
    """캐시를 갱신해야 하는지 확인"""
    global _last_cache_update
    if _last_cache_update is None:
        return True
    return datetime.now() - _last_cache_update > _CACHE_TTL

def fetch_all_used_nicknames(cursor) -> set:
    """DB에서 사용된 닉네임을 가져와 캐시를 갱신"""
    global _nickname_cache, _last_cache_update
    
    if _should_refresh_cache():
        cursor.execute("SELECT anonymous_name FROM AnonymousNames")
        rows = cursor.fetchall()
        _nickname_cache = set(row["anonymous_name"] for row in rows)
        _last_cache_update = datetime.now()
        print(f"[info] 닉네임 캐시 갱신 완료: {len(_nickname_cache)}개")
    
    return _nickname_cache

def generate_unique_nickname(cursor, mbti_keywords, hobby_keywords, max_retries=5) -> str:
    used_nicknames = fetch_all_used_nicknames(cursor)
    print(f"[debug] 현재 사용 중인 닉네임 수: {len(used_nicknames)}")

    for attempt in range(max_retries):
        prompt = build_prompt(mbti_keywords, hobby_keywords)

        try:
            response = call_llm(prompt)
            candidates = parse_nickname_response(response)
            print(f"[debug] 생성된 닉네임 후보: {candidates}")
            
            if not candidates:
                print(f"[warn] LLM이 닉네임 후보를 생성하지 못했습니다. (시도 {attempt+1}/{max_retries})")
                continue
                
            random.shuffle(candidates)

            for name in candidates:
                if name not in used_nicknames:
                    print(f"[info] 선택된 닉네임: {name}")
                    return name
                else:
                    print(f"[debug] 중복된 닉네임 발견: {name}")
                    
            print(f"[warn] 모든 후보가 중복입니다. (시도 {attempt+1}/{max_retries})")
            
        except Exception as e:
            print(f"[warn] LLM 응답 오류 (재시도 {attempt+1}/{max_retries}): {e}")

        time.sleep(1)

    raise RuntimeError("유일한 별명 생성 실패: 중복 회피 불가")
