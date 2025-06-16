# mysql_loader.py
# import mysql.connector
from typing import Tuple, Dict, Optional

# ✅ 사용자 점수 및 취미 조회 함수
def fetch_user_mbti_and_hobby(cursor, user_id: int) -> Tuple[Optional[Dict[str, int]], Optional[str]]:
    # ① SurveyMBTI 최신 1개 조회
    cursor.execute(
        """
        SELECT ei_score, sn_score, tf_score, jp_score
        FROM SurveyMBTI
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (user_id,)
    )
    mbti_row = cursor.fetchone()

    if not mbti_row:
        return None, None

    scores = {
        "ei_score": mbti_row["ei_score"],
        "sn_score": mbti_row["sn_score"],
        "tf_score": mbti_row["tf_score"],
        "jp_score": mbti_row["jp_score"]
    }

    # ② SurveyHobby 최신 1개 조회
    cursor.execute(
        """
        SELECT hobby_name
        FROM SurveyHobby
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (user_id,)
    )
    hobby_row = cursor.fetchone()
    hobby_name = hobby_row["hobby_name"] if hobby_row else None

    return scores, hobby_name
