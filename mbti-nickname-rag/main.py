#main.py

from run_pipeline import (
    setup_environment,
    connect_to_mysql,
    load_exact_match_retriever,
    setup_llm_chain,
    fetch_target_users,
    generate_and_save_nicknames,
    logger
)

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
