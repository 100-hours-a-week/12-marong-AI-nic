# nickname_generator.py (유지)

from modules.llm_caller import call_llm
from modules.result_parser import parse_nickname_response
import random
import time
from modules.prompt_builder import build_prompt

def fetch_all_used_nicknames(cursor) -> set:
    cursor.execute("SELECT anonymous_name FROM AnonymousNames")
    rows = cursor.fetchall()
    return set(row["anonymous_name"] for row in rows)

def generate_unique_nickname(cursor, mbti_keywords, hobby_keywords, max_retries=5) -> str:
    used_nicknames = fetch_all_used_nicknames(cursor)

    for attempt in range(max_retries):
        prompt = build_prompt(mbti_keywords, hobby_keywords)

        try:
            response = call_llm(prompt)
            candidates = parse_nickname_response(response)
            random.shuffle(candidates)

            for name in candidates:
                if name not in used_nicknames:
                    return name
        except Exception as e:
            print(f"[warn] LLM 응답 오류 (재시도 {attempt+1}/{max_retries}): {e}")

        time.sleep(1)

    raise RuntimeError("유일한 별명 생성 실패: 중복 회피 불가")
