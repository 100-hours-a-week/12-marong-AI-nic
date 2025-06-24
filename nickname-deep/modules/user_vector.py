# user_vector.py

from typing import Dict, List

# ✅ 취미 리스트 순서 고정 (one-hot 용)
HOBBY_LIST = [
    "운동", "독서", "음악", "여행", "게임", "요리", "OTT시청", "문화생활",
    "코딩", "애니메이션", "집안일", "밴드", "취침", "사진 찍기", "노래하기", "자전거",
    "연기", "잠자기", "웹툰", "야구 시청", "수영", "명상"
]

# ✅ 사용자 점수 + 취미를 26차원 벡터로 변환
def create_user_vector(scores: Dict[str, int], hobby_name: str) -> List[float]:
    # ① MBTI 점수 4차원: [ei, sn, tf, jp]
    mbti_vector = [
        scores["ei_score"],
        scores["sn_score"],
        scores["tf_score"],
        scores["jp_score"]
    ]

    # ② 취미 one-hot 22차원
    hobby_vector = [1 if h == hobby_name else 0 for h in HOBBY_LIST]

    # ③ 결합 → 총 26차원 벡터
    return mbti_vector + hobby_vector


