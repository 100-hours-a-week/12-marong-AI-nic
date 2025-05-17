from langchain.schema import BaseRetriever
import random
from typing import List

class NicknameRetriever:
    def __init__(self, grouped_nicknames: dict, memory):
        self.grouped_nicknames = grouped_nicknames
        self.memory = memory

    def get_nickname(self, mbti: str, hobby: str, user_id: int, group_id: int) -> str:
        """MBTI, 취미 기반 별명 추천 (중복 방지 포함)"""
        candidates = self.grouped_nicknames.get((mbti, hobby), [])
        available = [n for n in candidates if not self.memory.is_used(user_id, group_id)]

        if not available:
            return "⚠️ 사용 가능한 별명이 없습니다."

        nickname = random.choice(available)
        self.memory.mark_used(user_id, group_id, nickname)
        return nickname

    def _get_relevant_documents(self, query: str) -> List:
        raise NotImplementedError("이 리트리버는 직접 get_nickname으로 사용하세요.")
