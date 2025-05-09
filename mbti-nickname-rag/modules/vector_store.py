# ── modules/vector_store.py ───────────────────────────
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def create_vector_stores(mbti_chunks, hobby_chunks, persist_dir="chroma_db"):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    mbti_vs = Chroma.from_documents(
        mbti_chunks,
        embedding,
        persist_directory=persist_dir,
        collection_name="mbti_traits"
    )
    mbti_vs.persist()

    hobby_vs = Chroma.from_documents(
        hobby_chunks,
        embedding,
        persist_directory=persist_dir,
        collection_name="hobby_subtraits"
    )
    hobby_vs.persist()

    return mbti_vs, hobby_vs
