# result_parser.py

import re
from typing import List

# ✅ 번호 기반 별명 추출기 (10자 이하 한글만 필터링)
def parse_nickname_response(text: str) -> List[str]:
    pattern = r"\d+\.\s*(.+?)\n(?=\d+\.|$)"
    matches = re.findall(pattern, text + "\n")
    seen = set()
    filtered = []

    for m in matches:
        name = m.strip()
        # ✅ 조건: 10자 이하, 한글/공백만, 중복 제거
        if (
            len(name) <= 10 and
            re.fullmatch(r"[가-힣\s]+", name) and
            name not in seen
        ):
            seen.add(name)
            filtered.append(name)

    return filtered


