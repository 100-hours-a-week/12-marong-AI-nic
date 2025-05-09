# modules/retriever.py

from langchain.schema import BaseRetriever, Document
from typing import List

class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        mbti_retriever: BaseRetriever,
        hobby_retriever: BaseRetriever,
        k1: int = 3,
        k2: int = 3,
    ):
        super().__init__()                # ← 반드시 호출!
        self.mbti_retriever = mbti_retriever
        self.hobby_retriever = hobby_retriever
        self.k1 = k1
        self.k2 = k2

    def get_relevant_documents(self, query: str) -> List[Document]:
        mbti_docs  = self.mbti_retriever.get_relevant_documents(query)
        hobby_docs = self.hobby_retriever.get_relevant_documents(query)
        return mbti_docs[:self.k1] + hobby_docs[:self.k2]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        mbti_docs  = await self.mbti_retriever.aget_relevant_documents(query)
        hobby_docs = await self.hobby_retriever.aget_relevant_documents(query)
        return mbti_docs[:self.k1] + hobby_docs[:self.k2]
