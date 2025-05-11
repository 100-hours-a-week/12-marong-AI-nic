"""
MBTI 및 취미 정보 기반 별명 생성 파이프라인
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
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

from modules.chroma_client_store import load_chroma_collections
from modules.client_retriever import ChromaClientRetriever
from modules.llm_wrapper import HyperclovaxChat
from modules.utils import reorder_nicknames
from modules.retriever import HybridRetriever

from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG, # INFO , DEBUGE
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("nickname_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── 1. 환경 설정 ───────────────────────────────────────────
def setup_environment():
    """환경 설정 및 구성 파일 로드"""
    load_dotenv()
    
    # HuggingFace 로그인
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN이 .env 파일에 없습니다.")
        raise ValueError("HF_TOKEN 환경 변수가 필요합니다.")
    
    login(token=hf_token)
    logger.info("HuggingFace 로그인 완료")
    
    # 설정 파일 로드
    # 1) 프로젝트 루트 계산 (Test.py 기준)
    BASE_DIR = Path(__file__).resolve().parent

    # 2) YAML 로드
    config_path = BASE_DIR / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 3) config 에 있는 상대경로를 절대경로로 변환
    #    이렇게 하면 어디서 실행하든 올바른 파일을 찾습니다.
    config["data"]["mbti_csv_path"]  = str(BASE_DIR / config["data"]["mbti_csv_path"])
    config["data"]["hobby_csv_path"] = str(BASE_DIR / config["data"]["hobby_csv_path"])
    config["data"]["chroma_dir"]     = str(BASE_DIR / config["data"]["chroma_dir"])
    
    return config

# ── 2. MySQL 연결 ─────────────────────────────────────────
def connect_to_mysql():
    """MySQL 데이터베이스 연결"""
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

# ── 3. 벡터 DB 로드 ───────────────────────────────────────
def load_vector_db(config: Dict[str, Any]):
    """Chroma 벡터 데이터베이스 로드"""
    persist_dir = config["data"]["chroma_dir"]
    try:
        # Chroma 컬렉션 로드
        client, mbti_col, hobby_col = load_chroma_collections(persist_dir)
        logger.info(f"Chroma 컬렉션 로드 완료: {persist_dir}")
        
        # 임베딩 모델 초기화
        embedding_model = HuggingFaceEmbeddings(
            model_name=config["model"]["embedding_model"]
        )
        
        # Retriever 생성
        mbti_retriever = ChromaClientRetriever.create(
            mbti_col,
            embedding_model.embed_query,
            k=config["retrieval"]["k_mbti"]
        )
        hobby_retriever = ChromaClientRetriever.create(
            hobby_col,
            embedding_model.embed_query,
            k=config["retrieval"]["k_hobby"]
        )
        
        # 하이브리드 리트리버 생성 후 반환
        return HybridRetriever(
            mbti_retriever=mbti_retriever,
            hobby_retriever=hobby_retriever,
            k1=config["retrieval"]["k_mbti"],
            k2=config["retrieval"]["k_hobby"]
        )
        
    except Exception as e:
        logger.error(f"벡터 DB 로드 실패: {e}")
        raise


# ── 4. LLM + QA 체인 구성 (HyperCLOVAX Chat 양식) ─────────────────────────────────
from typing import Dict, Any
from modules.llm_wrapper import HyperclovaxChat
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
import logging

logger = logging.getLogger(__name__)

def setup_llm_chain(config: Dict[str, Any]):
    """HyperCLOVAX Chat 기반 LLM + QA 체인 설정"""
    try:
        # 1) HyperCLOVA Chat 모델 래퍼
        chat_model = HyperclovaxChat(
            model_id=config["model"]["huggingface_model_id"]
        )
        
        # 2) PromptTemplate 정의
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                # tool_list 역할 (현재는 빈 리스트)
                "role: tool_list\n"
                "content: \"\"\n\n"
                
                # system 역할
                "role: system\n"
                "content: \"- AI 언어모델의 이름은 \\\"CLOVA X\\\" 이며 네이버에서 만들었다.\\n"
                "- 오늘은 2025년 04월 24일(목)이다.\"\n\n"
                
                # user 역할: 문서(context)와 질문(question)
                "role: user\n"
                "content: |\n"
                "  당신은 익명 별명 추천 전문가입니다.\n"
                "  아래 내용을 참고해서 답해주세요:\n\n"
                "  === 참고 정보 ===\n"
                "  {context}\n\n"
                "  질문: {question}\n\n"
                "  위 정보를 바탕으로, 사용자에게 어울리는 별명 5개를\n"
                "  번호 매기기 형식으로 추천하세요.\n"
                "  예: 1. 별명A 2. 별명B ...\n"
                "  별명은 한국어로 짧고 기억하기 쉽게 만들어주세요."
            )
        )
        
        # 3) QA 체인 생성
        #    HyperCLOVAX Chat 모델을 llm 파라미터로 넘깁니다.
        qa_chain = create_stuff_documents_chain(
            llm=chat_model,
            prompt=prompt,
            document_variable_name="context"
        )
        
        logger.info("LLM 및 QA 체인 설정 완료 (HyperCLOVA Chat 양식)")
        return qa_chain
        
    except Exception as e:
        logger.error(f"LLM 체인 설정 실패: {e}")
        raise


# ── 5. 별명 생성 대상 사용자 조회 ──────────────────────────
def fetch_target_users(cursor):
    """별명 생성이 필요한 사용자 목록 조회"""
    try:
        query = """
        SELECT u.id AS user_id, g.id AS group_id,
               sm.mbti, sh.hobby_name AS hobby, ug.week
        FROM Users u
        JOIN UserGroups ug ON u.id = ug.user_id
        JOIN `Groups` g ON ug.group_id = g.id
        JOIN SurveyMBTI sm ON u.id = sm.user_id
        JOIN SurveyHobby sh ON u.id = sh.user_id
        LEFT JOIN AnonymousNames an
          ON u.id = an.user_id AND g.id = an.group_id AND ug.week = an.week
        WHERE an.id IS NULL
        """
        
        cursor.execute(query)
        users = cursor.fetchall()
        logger.info(f"별명 생성 대상 사용자 {len(users)}명 조회 완료")
        return users
        
    except mysql.connector.Error as err:
        logger.error(f"사용자 조회 실패: {err}")
        raise

# ── 6. 별명 생성 및 저장 ───────────────────────────────────
def generate_and_save_nicknames(
    cursor,
    conn,
    users: List[Dict],
    qa_chain,
    retriever
):
    """사용자별 별명 생성 및 DB 저장"""
    insert_query = """
    INSERT INTO AnonymousNames 
    (user_id, group_id, anonymous_name, week) 
    VALUES (%s, %s, %s, %s)
    """
    success_count = 0
    error_count = 0

    for user in users:
        uid   = user["user_id"]
        gid   = user["group_id"]
        mbti  = user["mbti"]
        hobby = user["hobby"]
        week  = user["week"]

        try:
            # 1) 과거 닉네임 조회
            cursor.execute(
                "SELECT anonymous_name FROM AnonymousNames WHERE user_id = %s",
                (uid,)
            )
            past_nicknames = [r["anonymous_name"] for r in cursor.fetchall()]

            # 2) 쿼리 생성
            query = f"MBTI가 {mbti}이고 취미가 {hobby}인 사람에게 어울리는 별명 5개를 추천해주세요"

            # 3) 관련 문서 검색 → Document 객체 리스트로 변환
            raw_docs = retriever.get_relevant_documents(query)
            docs: List[Document] = []
            for d in raw_docs:
                if isinstance(d, Document):
                    docs.append(d)
                else:
                    docs.append(Document(page_content=str(d)))
            logger.debug(f"▶ context docs: {[doc.page_content for doc in docs]}")

            # 4) LLM 체인 호출
            result = qa_chain.invoke({
                "context": docs,
                "question": query
            })
            # invoke() 반환값이 dict인지 str인지 분기 처리
            if isinstance(result, dict):
                # 'text' 또는 'answer' 키 중 있는 쪽을 사용
                raw_output = result.get("text") or result.get("answer") or ""
            else:
                raw_output = result
            logger.debug(f"▶ raw_output type: {type(raw_output)}, value: {raw_output!r}")

            # 5) 별명 정제
            nickname_text = str(raw_output)
            nicknames = reorder_nicknames(nickname_text, past_nicknames)

            if nicknames:
                final_nick = nicknames[0]
                cursor.execute(insert_query, (uid, gid, final_nick, week))
                logger.info(f"✅ 사용자 {uid}, 그룹 {gid}, 주차 {week}: '{final_nick}' 저장 완료")
                success_count += 1

                if success_count % 50 == 0:
                    conn.commit()
                    logger.info(f"중간 커밋: {success_count}명 처리")
            else:
                logger.warning(f"⚠️ 사용자 {uid}, 그룹 {gid}: 별명 추천 결과가 비어있음")
                error_count += 1

        except Exception as e:
            logger.error(f"[ERROR] 사용자 {uid}, 그룹 {gid} 에러: {e}")
            error_count += 1

    # 최종 커밋 및 요약 로깅
    conn.commit()
    logger.info(f"✅ 처리 완료: 성공 {success_count}명, 실패 {error_count}명")
    return success_count, error_count


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
        
        # 3. 벡터 DB 로드
        retriever = load_vector_db(config)
        
        # 4. LLM + QA 체인 구성
        qa_chain = setup_llm_chain(config)
        
        # 5. 별명 생성 대상 사용자 조회
        users = fetch_target_users(cursor)
        
        if not users:
            logger.info("처리할 사용자가 없습니다.")
        else:
            # 6. 별명 생성 및 저장
            success_count, error_count = generate_and_save_nicknames(
                cursor, conn, users, qa_chain, retriever
            )
            
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