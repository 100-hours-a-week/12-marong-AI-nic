from langchain.schema import BaseRetriever, Document
from pydantic import PrivateAttr
from typing import List
import random

class NicknameRetriever(BaseRetriever):
    _grouped_nicknames: dict = PrivateAttr()
    _memory: any = PrivateAttr()

    def __init__(self, grouped_nicknames: dict, memory):
        super().__init__()
        self._grouped_nicknames = grouped_nicknames
        self._memory = memory

    def get_nickname(self, mbti: str, hobby: str, user_id: int, group_id: int, week: int) -> str:
        """MBTI와 취미 조합 기반 별명 생성 (주차 기반 중복 방지 포함)"""
        candidates = self._grouped_nicknames.get((mbti, hobby), [])
        available = [n for n in candidates if not self._memory.is_used(user_id, group_id, week)]

        if not available:
            return "⚠️ 사용 가능한 별명이 없습니다."

        nickname = random.choice(available)
        self._memory.mark_used(user_id, group_id, nickname, week)
        return nickname

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        LangChain 체인 호환을 위한 dummy 함수.
        현재는 get_nickname()으로만 사용.
        """
        raise NotImplementedError("이 리트리버는 직접 get_nickname()으로 사용하세요.")
