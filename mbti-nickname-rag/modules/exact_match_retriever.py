# modules/exact_match_retriever.py
import pandas as pd
from typing import List

class ExactMatchRetriever:
    def __init__(self, mbti_csv_path: str, hobby_csv_path: str, top_k: int = 5):
        self.mbti_df = pd.read_csv(mbti_csv_path)
        self.hobby_df = pd.read_csv(hobby_csv_path)
        self.top_k = top_k

    def get_mbti_traits(self, mbti: str) -> List[str]:
        return (
            self.mbti_df[self.mbti_df["MBTI"] == mbti]
            .sort_values(by="Weight", ascending=False)["Trait"]
            .head(self.top_k)
            .tolist()
        )

    def get_hobby_traits(self, hobby: str) -> List[str]:
        return (
            self.hobby_df[self.hobby_df["Hobby"] == hobby]
            .sort_values(by="Weight", ascending=False)["Subtrait"]
            .head(self.top_k)
            .tolist()
        )
