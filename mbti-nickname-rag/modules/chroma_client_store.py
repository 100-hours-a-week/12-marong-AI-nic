# modules/chroma_client_store.py

import os
from pathlib import Path
import chromadb

def load_chroma_collections(persist_dir: str):
    """
    레거시 호환용: PersistentClient로 로컬 DB에 연결하고
    mbti_traits, hobby_subtraits 컬렉션을 반환합니다.
    """
    from chromadb import PersistentClient

    # 디스크 기반 Chroma DB에 연결
    client    = PersistentClient(path=persist_dir)
    mbti_col  = client.get_or_create_collection(name="mbti_traits")
    hobby_col = client.get_or_create_collection(name="hobby_subtraits")
    return client, mbti_col, hobby_col

