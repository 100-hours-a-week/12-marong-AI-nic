# modules/retriever.py

from langchain.schema import Document
from typing import List

class HybridRetriever:
    """
    MBTI 리트리버와 Hobby 리트리버를 결합한 하이브리드 리트리버입니다.
    """
    def __init__(
        self,
        mbti_retriever,
        hobby_retriever,
        k1: int = 3,
        k2: int = 3,
    ):
        # 개별 리트리버와 반환할 개수 설정
        self.mbti_retriever = mbti_retriever
        self.hobby_retriever = hobby_retriever
        self.k1 = k1
        self.k2 = k2

    def get_relevant_documents(self, query: str) -> List[Document]:
        # MBTI 및 Hobby 리트리버에서 각각 문서를 가져와 k1, k2 개수만큼 결합
        mbti_docs = self.mbti_retriever.get_relevant_documents(query)
        hobby_docs = self.hobby_retriever.get_relevant_documents(query)
        return mbti_docs[: self.k1] + hobby_docs[: self.k2]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        mbti_docs = await self.mbti_retriever.aget_relevant_documents(query)
        hobby_docs = await self.hobby_retriever.aget_relevant_documents(query)
        return mbti_docs[: self.k1] + hobby_docs[: self.k2]
