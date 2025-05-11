# ingest_data.py

import yaml
from pathlib import Path
import pandas as pd
from modules.chroma_client_store import load_chroma_collections

# ── 설정 파일 및 경로 설정 ───────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "configs" / "default.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

MBTI_CSV     = BASE_DIR / config["data"]["mbti_csv_path"]
HOBBY_CSV    = BASE_DIR / config["data"]["hobby_csv_path"]
CHROMA_DIR   = BASE_DIR / config["data"]["chroma_dir"]

# ── MBTI 벡터 삽입 ────────────────────────────────────────
def ingest_mbti_traits(csv_path: Path = None):
    """
    MBTI traits CSV를 읽어 mbti_traits 컬렉션에 벡터와 메타데이터를 추가합니다.
    """
    persist_dir = str(CHROMA_DIR)
    client, mbti_col, _ = load_chroma_collections(persist_dir)
    path = Path(csv_path) if csv_path else MBTI_CSV
    df = pd.read_csv(path)
    docs, metas, ids = [], [], []
    for idx, row in df.iterrows():
        docs.append(row["Trait"])
        metas.append({
            "MBTI":   row["MBTI"],
            "trait":  row["Trait"],
            "weight": float(row["Weight"])
        })
        ids.append(f"{row['MBTI']}_{idx}")
    mbti_col.add(documents=docs, metadatas=metas, ids=ids)
    print("✅ MBTI 삽입 완료")

# ── Hobby 벡터 삽입 ──────────────────────────────────────
def ingest_hobby_subtraits(csv_path: Path = None):
    """
    Hobby subtraits CSV를 읽어 hobby_subtraits 컬렉션에 벡터와 메타데이터를 추가합니다.
    """
    persist_dir = str(CHROMA_DIR)
    client, _, hobby_col = load_chroma_collections(persist_dir)
    path = Path(csv_path) if csv_path else HOBBY_CSV
    df = pd.read_csv(path)
    docs, metas, ids = [], [], []
    for idx, row in df.iterrows():
        docs.append(row["Hobby Subtrait"])
        metas.append({
            "hobby":    row["Hobby Subtrait"].split("(")[0],
            "subtrait": row["Hobby Subtrait"],
            "weight":   float(row["Weight"])
        })
        ids.append(f"{row['Hobby Subtrait']}_{idx}")
    hobby_col.add(documents=docs, metadatas=metas, ids=ids)
    print("✅ Hobby 삽입 완료")

# ── 직접 실행 시 삽입 작업 ─────────────────────────────────
if __name__ == "__main__":
    ingest_mbti_traits()
    ingest_hobby_subtraits()
