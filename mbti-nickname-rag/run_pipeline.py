import os
import yaml
from dotenv import load_dotenv
from huggingface_hub import login
import mysql.connector

from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

from modules.chroma_client_store import load_chroma_collections
from modules.client_retriever import ChromaClientRetriever
from modules.llm_wrapper import LLMWrapper
from modules.utils import reorder_nicknames

# ── 1. 환경 설정 ───────────────────────────────────────────
load_dotenv()
login(token=os.getenv("HF_TOKEN"))

with open("configs/default.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# ── 2. 벡터 DB 로드 (ChromaClient 기반) ─────────────────────
client, mbti_col, hobby_col = load_chroma_collections(persist_dir=config["data"]["chroma_dir"])
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

mbti_retriever = ChromaClientRetriever.create(mbti_col, embedding_model.embed_query, k=config["retrieval"]["k_mbti"])
hobby_retriever = ChromaClientRetriever.create(hobby_col, embedding_model.embed_query, k=config["retrieval"]["k_hobby"])

# ── 3. LLM + QA 체인 구성 ───────────────────────────────────
llm = LLMWrapper(model_id=config["model"]["huggingface_model_id"]).pipeline

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="아래 내용을 참고해서 질문에 답해주세요:\n{context}\n\n질문: {question}"
)

qa_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt,
    document_variable_name="context"
)

# 병합 리트리버 (MBTI + Hobby 순서 검색)
class HybridRetriever:
    def __init__(self, r1, r2):
        self.r1 = r1
        self.r2 = r2

    def get_relevant_documents(self, query):
        return self.r1.get_relevant_documents(query) + self.r2.get_relevant_documents(query)

retriever = HybridRetriever(mbti_retriever, hobby_retriever)

# ── 4. MySQL 연결 ─────────────────────────────────────────
conn = mysql.connector.connect(
    host=os.getenv("MYSQL_HOST"),
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    database=os.getenv("MYSQL_DB")
)
cursor = conn.cursor(dictionary=True)

# ── 5. 추천 대상 사용자 조회 ───────────────────────────────
cursor.execute("""
SELECT u.id AS user_id, g.id AS group_id, sm.mbti, sh.hobby_name AS hobby, ug.week
FROM Users u
JOIN UserGroups ug ON u.id = ug.user_id
JOIN `Groups` g ON ug.group_id = g.id
JOIN SurveyMBTI sm ON u.id = sm.user_id
JOIN SurveyHobby sh ON u.id = sh.user_id
LEFT JOIN AnonymousNames an 
  ON u.id = an.user_id AND g.id = an.group_id AND ug.week = an.week
WHERE an.id IS NULL
""")
users = cursor.fetchall()

# ── 6. 추천 및 저장 ───────────────────────────────────────
insert_query = """
INSERT INTO AnonymousNames (user_id, group_id, anonymous_name, week)
VALUES (%s, %s, %s, %s)
"""

for user in users:
    uid = user["user_id"]
    gid = user["group_id"]
    mbti = user["mbti"]
    hobby = user["hobby"]
    week = user["week"]

    prompt = f"MBTI가 {mbti}이고 Hobby가 {hobby}인 사람에게 별명 5개 추천해줘"

    # ✅ 과거 닉네임 조회
    cursor.execute("SELECT anonymous_name FROM AnonymousNames WHERE user_id = %s", (uid,))
    past_nicknames = [row["anonymous_name"] for row in cursor.fetchall()]

    try:
        # ✅ LLM 호출 방식 단순화
        result = qa_chain.run(prompt)

        # ✅ 닉네임 정리
        reordered = reorder_nicknames(result, past_nicknames)
        if reordered:
            final_nick = reordered[0]
            cursor.execute(insert_query, (uid, gid, final_nick, week))
            print(f"[✓] {uid}-{gid} (week {week}): {final_nick}")
        else:
            print(f"[!] {uid}-{gid}: 추천 결과가 없음 또는 중복")
    except Exception as e:
        print(f"[Error] {uid}-{gid}: {e}")

# ── 7. 종료 ───────────────────────────────────────────────
conn.commit()
cursor.close()
conn.close()
print("✅ 모든 사용자에 대해 닉네임 추천이 완료되었습니다.")
