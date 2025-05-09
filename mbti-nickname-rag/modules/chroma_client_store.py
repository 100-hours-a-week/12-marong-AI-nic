# ── modules/chroma_client_store.py ────────────────────────
from chromadb import PersistentClient


def load_chroma_collections(persist_dir: str):
    """
    ChromaClient 처리 파일
    - 지정된 디렉토리를 기본으로 ChromaClient 체계 로드
    - collection_name과 연결하여 각 필드 설정
    """
    client = PersistentClient(path=persist_dir)

    mbti_col = client.get_or_create_collection(name="mbti_traits")
    hobby_col = client.get_or_create_collection(name="hobby_subtraits")

    return client, mbti_col, hobby_col
