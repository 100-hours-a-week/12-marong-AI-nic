# result_parser.py

import re
from typing import List

# ✅ 번호 기반 별명 추출기 (10자 이하 한글만 필터링)
def parse_nickname_response(text: str) -> List[str]:
    # 패턴 수정: **별명** 형식을 고려하고, 뒤에 오는 설명은 무시
    pattern = r"\d+\.\s*\*\*([가-힣\s]+?)\*\*\s*(?:[:(].*)?$"

    matches = []
    # LLM 응답을 줄 단위로 분리하여 각 줄에서 별명을 찾음
    for line in text.splitlines():
        match = re.search(pattern, line)
        if match:
            name = match.group(1).strip()
            # ✅ 조건: 8자 이하, 한글/공백만
            if len(name) <= 8 and re.fullmatch(r"[가-힣\s]+", name):
                matches.append(name)

    # 중복 제거
    return list(dict.fromkeys(matches))


