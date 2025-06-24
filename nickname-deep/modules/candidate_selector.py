#candidate_selector.py

import pickle
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import os

# ✅ 후보 벡터 로드 함수 (경로 수정)
def load_candidate_vectors(path: str = "nickname-deep/data/candidate_vectors01.pkl") -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 후보 벡터 파일이 존재하지 않습니다: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

# ✅ 유사도 기반 Top-1 조합 선택기
def select_top_candidate(user_vector: List[float], candidates: List[Dict]) -> Tuple[str, str]:
    candidate_matrix = np.array([c["vector"] for c in candidates])  # (128, 12)
    user_vector_np = np.array(user_vector).reshape(1, -1)            # (1, 12)

    similarities = cosine_similarity(user_vector_np, candidate_matrix)[0]  # (128,)
    top_index = int(np.argmax(similarities))

    top_candidate = candidates[top_index]
    return top_candidate["mbti"], top_candidate["hobby"]


