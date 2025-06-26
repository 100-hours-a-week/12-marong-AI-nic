#keyword_extractor.py

import json
from typing import List, Dict
import os
import random

# ✅ traits 로딩 함수
def load_traits(
    mbti_path="nickname-deep/data/mbti_traits_with_binary_vector.json",
    hobby_path="nickname-deep/data/hobby_traits.json"
) -> Dict:
    if not os.path.exists(mbti_path):
        raise FileNotFoundError(f"MBTI traits 파일이 존재하지 않습니다: {mbti_path}")
    if not os.path.exists(hobby_path):
        raise FileNotFoundError(f"Hobby traits 파일이 존재하지 않습니다: {hobby_path}")

    with open(mbti_path, encoding="utf-8") as f:
        mbti_data = json.load(f)
    with open(hobby_path, encoding="utf-8") as f:
        hobby_data = json.load(f)
    return {"mbti": mbti_data, "hobby": hobby_data}

# ✅ MBTI 키워드 추출기
def get_mbti_keywords(mbti: str, scores: Dict[str, int], traits: Dict) -> List[str]:
    result = []
    mbti_traits = traits["mbti"][mbti]
    
    # 각 스케일별로 키워드 수집
    for scale in ["ei_score", "sn_score", "tf_score", "jp_score"]:
        value = scores[scale]
        for rng, words in mbti_traits[scale].items():
            low, high = map(int, rng.split("-"))
            if low <= value <= high:
                result.extend(words)
                break
    
    # 기본 키워드 추가
    result.extend(mbti_traits.get("base_keywords", []))
    
    # 결과를 무작위로 섞기
    random.shuffle(result)
    
    return result

# ✅ Hobby 키워드 추출기
def get_hobby_keywords(hobby_name: str, traits: Dict) -> List[str]:
    return traits["hobby"][hobby_name]["subtraits"]
