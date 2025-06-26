# result_parser.py

import re
from typing import List

# ✅ 번호 기반 별명 추출기 (8자 이하, 한글 우선)
def parse_nickname_response(text: str) -> List[str]:
    # 패턴 수정: **별명** 형식을 고려하고, 뒤에 오는 설명은 무시
    pattern = r"\d+\.\s*\*\*([가-힣a-zA-Z\s]+?)\*\*\s*(?:[:(].*)?$"

    matches = []
    # LLM 응답을 줄 단위로 분리하여 각 줄에서 별명을 찾음
    for line in text.splitlines():
        match = re.search(pattern, line)
        if match:
            name = match.group(1).strip()
            # ✅ 조건: 8자 이하, 한글/영어/공백, '별명'과 '취미' 단어 제외
            if (len(name) <= 10 and 
                re.fullmatch(r"[가-힣a-zA-Z\s]+", name) and 
                '별명' not in name and 
                '취미' not in name):
                matches.append(name)

    # 중복 제거
    unique_matches = list(dict.fromkeys(matches))
    
    # 한글 닉네임을 우선 정렬
    korean_matches = [name for name in unique_matches if re.search(r'[가-힣]', name)]
    other_matches = [name for name in unique_matches if not re.search(r'[가-힣]', name)]
    
    return korean_matches + other_matches


