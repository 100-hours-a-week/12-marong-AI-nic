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
            input_variables=["context", "mbti", "hobby"],
            template=(
                "role: system\n"
                "content: \"- 당신은 익명 별명을 추천해주는 전문가입니다. "
                "- AI 언어모델의 이름은 'CLOVA X'입니다. "
                "- 오늘은 2025-04-24입니다.\"\n\n"

                "role: user\n"
                "content: |\n"
                "  아래 특성을 가진 사람에게 어울리는 한국어 별명을 5개 추천해주세요.\n"
                "  각 별명은 간결하고 기억하기 쉬운 단어로 구성되어야 하며, "
                "번호 매기기 형식(1. ~)으로 출력해주세요.\n\n"
                "  === 특성 목록 ===\n"
                "  {context}\n\n"
                "  MBTI 유형: {mbti}\n"
                "  취미: {hobby}\n"
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
    INSERT INTO AnonymousNames(user_id, group_id, anonymous_name, week)
    VALUES (%s, %s, %s, %s)
    """
    used_names = set()

    for user in users:
        # user는 dict 형태: dictionary=True 설정 필요
        uid = user["user_id"]
        gid = user["group_id"]
        mbti = user["mbti"]
        hobby = user["hobby"]
        week = user["week"]

        # 과거 별명 가져오기
        cursor.execute(
            "SELECT anonymous_name FROM AnonymousNames WHERE user_id = %s",
            (uid,)
        )
        past_rows = cursor.fetchall()
        past = [row["anonymous_name"] for row in past_rows]

        # 정확 매칭 기반 traits
        mbti_traits = retriever.get_mbti_traits(mbti)
        hobby_traits = retriever.get_hobby_traits(hobby)

        # 여기에 로그 추가
        logger.debug(f"[{uid}] MBTI traits: {mbti_traits}")
        logger.debug(f"[{uid}] Hobby traits: {hobby_traits}")

        
        context_list = mbti_traits + hobby_traits
        docs = [Document(page_content=trait) for trait in context_list]

       # LLM 질의
        result = qa_chain.invoke({
            "context": docs,
          "mbti": mbti,
            "hobby": hobby
        })
        raw = result.get("text") if isinstance(result, dict) else result
        logger.debug(f"LLM 응답 원문: {raw}")

        # 별명 후보 추출
        candidates = reorder_nicknames(str(raw), past)

        # 길이 제한 및 중복 제거
        valid_candidates = [n for n in candidates if n not in used_names and len(n) <= 30]
        final_nickname = valid_candidates[0] if valid_candidates else None

        # 최종 저장
        if final_nickname:
            cursor.execute(insert_q, (uid, gid, final_nickname, week))
            conn.commit()
            used_names.add(final_nickname)
            logger.info(f"[{uid}] '{final_nickname}' 저장 완료")
        else:
            logger.warning(f"[{uid}] 유효한 별명 없음 (중복 또는 길이 초과)")

    
 # ── 7 & 8. 메인 함수 ────────────────────────────────────────
def main():
    """별명 생성 파이프라인 메인 함수"""
    try:
        # 1. 환경 설정
        logger.info("==== 별명 생성 파이프라인 시작 ====")
        config = setup_environment()

        # 2. MySQL 연결
        conn = connect_to_mysql()
        cursor = conn.cursor(dictionary=True)

        # 3. 정확 매칭 리트리버 로드
        retriever = load_exact_match_retriever(config)

        # 4. LLM + QA 체인 구성
        qa_chain = setup_llm_chain(config)

        # 5. 별명 생성 대상 사용자 조회
        users = fetch_target_users(cursor)

        if not users:
            logger.info("처리할 사용자가 없습니다.")
        else:
            # 6. 별명 생성 및 저장
            generate_and_save_nicknames(cursor, conn, users, qa_chain, retriever)

        # 7. 커넥션 종료 및 정리
        cursor.close()
        conn.close()
        logger.info("MySQL 연결 종료")

        # 8. 종료
        logger.info("==== 별명 생성 파이프라인 완료 ====")

    except Exception as e:
        logger.critical(f"파이프라인 실행 중 오류 발생: {str(e)}", exc_info=True)
        return 1

    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
