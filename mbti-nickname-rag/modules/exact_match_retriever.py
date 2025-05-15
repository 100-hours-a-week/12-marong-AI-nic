# modules/exact_match_retriever.py

import pandas as pd
from typing import List

class ExactMatchRetriever:
    def __init__(self, mbti_csv_path: str, hobby_csv_path: str, top_k: int = 5):
        # MBTI traits 로드 (기존대로)
        self.mbti_df = pd.read_csv(mbti_csv_path, encoding="utf-8")
        
        # 취미 가중치 CSV 로드
        # Pandas 1.3+라면 on_bad_lines='skip' 사용
        hobby_df = pd.read_csv(
            hobby_csv_path,
            encoding="utf-8-sig",
            on_bad_lines="skip",   # 잘못된 라인은 건너뜁니다
            engine="python"        # python 엔진으로 읽어야 on_bad_lines가 동작
        )
        
        # 컬럼명 통일
        hobby_df = hobby_df.rename(columns={
            "subhobby": "Hobby",
            "trait":    "Subtrait",
        })
        # Weight 컬럼이 없다면 기본 1로
        if "Weight" not in hobby_df.columns:
            hobby_df["Weight"] = 1
        
        self.hobby_df = hobby_df
        self.top_k = top_k

    def get_mbti_traits(self, mbti: str) -> List[str]:
        return (
            self.mbti_df[self.mbti_df["MBTI"] == mbti]
            .sort_values(by="Weight", ascending=False)
            .head(self.top_k)["Trait"]
            .tolist()
        )

    def get_hobby_traits(self, hobby: str) -> List[str]:
        return (
            self.hobby_df[self.hobby_df["Hobby"] == hobby]
            .sort_values(by="Weight", ascending=False)
            .head(self.top_k)["Subtrait"]
            .tolist()
        )
