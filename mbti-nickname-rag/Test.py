"""
MBTI 및 취미 정보 기반 별명 생성 파이프라인 (정확 매칭 기반)
"""
import os
import yaml
import logging
from dotenv import load_dotenv
from huggingface_hub import login
import mysql.connector
from typing import List, Dict, Any

from langchain.schema import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

from modules.llm_wrapper import HyperclovaxChat
from modules.utils import reorder_nicknames
from modules.exact_match_retriever import ExactMatchRetriever

from pathlib import Path

# ── 로깅 설정 ───────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("nickname_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── 1. 환경 설정 ─────────────────────────────────────────
def setup_environment() -> Dict[str, Any]:
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN이 .env 파일에 없습니다.")
        raise ValueError("HF_TOKEN 환경 변수가 필요합니다.")
    login(token=hf_token)
    logger.info("HuggingFace 로그인 완료")

    BASE_DIR = Path(__file__).resolve().parent
    config_path = BASE_DIR / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # 절대 경로로 변환
    config["data"]["mbti_csv_path"]  = str(BASE_DIR / config["data"]["mbti_csv_path"])
    config["data"]["hobby_csv_path"] = str(BASE_DIR / config["data"]["hobby_csv_path"])
    return config

# ── 2. MySQL 연결 ────────────────────────────────────────
def connect_to_mysql():
    try:
        conn = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DB")
        )
        logger.info("MySQL 데이터베이스 연결 성공")
        return conn
    except mysql.connector.Error as err:
        logger.error(f"MySQL 연결 실패: {err}")
        raise

# ── 3. 정확 매칭 리트리버 로드 ─────────────────────────────
def load_exact_match_retriever(config: Dict[str, Any]) -> ExactMatchRetriever:
    try:
        retriever = ExactMatchRetriever(
            mbti_csv_path=config["data"]["mbti_csv_path"],
            hobby_csv_path=config["data"]["hobby_csv_path"],
            top_k=config["retrieval"]["k_mbti"]
        )
        logger.info("정확 매칭 리트리버 로드 완료")
        return retriever
    except Exception as e:
        logger.error(f"정확 매칭 리트리버 로드 실패: {e}")
        raise

# ── 4. LLM + QA 체인 설정 ─────────────────────────────────
def setup_llm_chain(config: Dict[str, Any]):
    try:
        chat_model = HyperclovaxChat(model_id=config["model"]["huggingface_model_id"])
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "role: system\n"
                "content: \"- AI 언어모델의 이름은 \\\"CLOVA X\\\"이고, 오늘은 2025-04-24입니다.\"\n\n"
                "role: user\n"
                "content: |\n"
                "  당신은 익명 별명 추천 전문가입니다.\n"
                "  아래 특성을 참고하여, 별명 5개를 번호 형식으로 추천해주세요.\n\n"
                "  === MBTI 특징 ===\n"
                "  {context}\n\n"
                "  질문: {question}\n"
            )
        )
        qa_chain = create_stuff_documents_chain(
            llm=chat_model,
            prompt=prompt,
            document_variable_name="context"
        )
        logger.info("LLM 및 QA 체인 설정 완료")
        return qa_chain
    except Exception as e:
        logger.error(f"LLM 체인 설정 실패: {e}")
        raise

# ── 5. 사용자 조회 ─────────────────────────────────────────
def fetch_target_users(cursor) -> List[Dict[str, Any]]:
    query = """
    SELECT u.id AS user_id, g.id AS group_id,
           sm.mbti, sh.hobby_name AS hobby, ug.week
    FROM Users u
    JOIN UserGroups ug ON u.id = ug.user_id
    JOIN `Groups` g ON ug.group_id = g.id
    JOIN SurveyMBTI sm ON u.id = sm.user_id
    JOIN SurveyHobby sh ON u.id = sh.user_id
    LEFT JOIN AnonymousNames an
      ON u.id=an.user_id AND g.id=an.group_id AND ug.week=an.week
    WHERE an.id IS NULL
    """
    cursor.execute(query)
    users = cursor.fetchall()
    logger.info(f"별명 생성 대상 사용자 {len(users)}명 조회 완료")
    return users

# ── 6. 별명 생성 및 저장 ───────────────────────────────────
def generate_and_save_nicknames(
    cursor, conn, users, qa_chain, retriever: ExactMatchRetriever
):
    insert_q = """
    INSERT INTO AnonymousNames(user_id, group_id, anonymous_name, week) VALUES (%s,%s,%s,%s)
    """
    used = set()
    for user in users:
        uid, gid, mbti, hobby, week = user.values()
        past = [r[0] for r in cursor.execute(
            "SELECT anonymous_name FROM AnonymousNames WHERE user_id=%s", (uid,)
        ) or []]
        mbti_traits = retriever.get_mbti_traits(mbti)
        hobby_traits = retriever.get_hobby_traits(hobby)
        context_list = mbti_traits + hobby_traits
        docs = [Document(page_content=t) for t in context_list]
        question = f"MBTI가 {mbti}이고 취미가 {hobby}인 사람에게 어울리는 별명 5개를 추천해주세요"
        result = qa_chain.invoke({"context": docs, "question": question})
        raw = result.get("text") if isinstance(result, dict) else result
        candidates = reorder_nicknames(str(raw), past)
        final = next((n for n in candidates if n not in used), None)
        if final:
            cursor.execute(insert_q, (uid, gid, final, week))
            used.add(final)
            conn.commit()
            logger.info(f"✅ 사용자{uid}별명:'{final}' 저장")
    logger.info("🌟 파이프라인 완료 🌟")

# ── 메인 ─────────────────────────────────────────────────
def main():
    config = setup_environment()
    conn = connect_to_mysql()
    uid, gid, mbti, hobby, week = user  # user is tuple with these values
    retriever = load_exact_match_retriever(config)
    qa = setup_llm_chain(config)
    users = fetch_target_users(cur)
    if users:
        generate_and_save_nicknames(cur, conn, users, qa, retriever)
    cur.close(); conn.close()

if __name__ == "__main__":
    main()
