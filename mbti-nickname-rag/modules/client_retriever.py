# modules/client_retriever.py

from typing import List, Callable
from langchain.schema import BaseRetriever, Document

class ChromaClientRetriever(BaseRetriever):
    """
    Chroma 컬렉션을 직접 호출해 벡터 검색 후
    langchain.schema.Document 리스트로 반환합니다.
    """
    collection: any
    embedding_fn: Callable[[str], List[float]]
    k: int

    class Config:
        extra = "allow"  # Pydantic이 모르는 필드를 허용

    def __init__(
        self,
        collection,
        embedding_fn: Callable[[str], List[float]],
        k: int = 5
    ):
        # 필수 필드 초기화
        super().__init__(collection=collection, embedding_fn=embedding_fn, k=k)
        self.collection = collection
        self.embedding_fn = embedding_fn
        self.k = k

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # 동기 검색
        embedding = self.embedding_fn(query)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=self.k,
            include=["metadatas", "documents"],
        )
        docs = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]
        return docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        # 비동기 검색 (필요 시)
        embedding = self.embedding_fn(query)
        results = await self.collection.aquery(
            query_embeddings=[embedding],
            n_results=self.k,
            include=["metadatas", "documents"],
        )
        docs = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]
        return docs

    @classmethod
    def create(
        cls,
        collection,
        embedding_fn: Callable[[str], List[float]],
        k: int = 5
    ) -> "ChromaClientRetriever":
        return cls(collection=collection, embedding_fn=embedding_fn, k=k)
