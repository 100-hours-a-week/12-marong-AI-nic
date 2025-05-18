import os
import sys
from datetime import datetime

# 🔧 모듈 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "Exmodules")))

from Exmodules.ex_nickname_data_loader import load_grouped_nicknames
from Exmodules.ex_nickname_memory import NicknameMemory
from Exmodules.ex_nickname_retriever import NicknameRetriever
from Exmodules.ex_mysql_connector import get_mysql_connection
from Exmodules.ex_week_index import GetWeekIndex  # ✅ 주차 계산기

def main():
    # ✅ 주차 계산
    base_date = datetime(2025, 1, 6)
    today = datetime.today()
    week_index = GetWeekIndex(today, base_date).get()
    print(f"📆 오늘은 기준일로부터 {week_index}주차입니다.\n")

    # ✅ 1. CSV 로딩
    csv_path = os.path.join(os.path.dirname(__file__), "Exmodules", "Exdata", "Exdata.csv")
    grouped_nicknames = load_grouped_nicknames(csv_path)

    # ✅ 2. DB 연결
    conn = get_mysql_connection()
    cursor = conn.cursor()

    # ✅ 3. 전체 사용자 조회
    query = """
    SELECT 
        u.id AS user_id,
        ug.group_id,
        sm.mbti,
        sh.hobby_name AS hobby
    FROM Users u
    JOIN UserGroups ug ON u.id = ug.user_id
    LEFT JOIN SurveyMBTI sm ON u.id = sm.user_id
    LEFT JOIN SurveyHobby sh ON u.id = sh.user_id
    """
    cursor.execute(query)
    targets = cursor.fetchall()

    # ✅ 4. 닉네임 생성 및 저장 (user_id 기준 덮어쓰기)
    memory = NicknameMemory(cursor)
    retriever = NicknameRetriever(grouped_nicknames=grouped_nicknames, memory=memory)

    for user_id, group_id, mbti, hobby in targets:
        if not mbti or not hobby:
            print(f"[error] [user_id={user_id}] MBTI 또는 취미 누락 → 생략")
            continue

        nickname = retriever.get_nickname(mbti, hobby, user_id, group_id, week_index)
        print(f"✅ [user_id={user_id}, group={group_id}] → {nickname}")

        if "⚠️ 사용 가능한 별명이 없습니다." in nickname:
            print(f"[skip] [user_id={user_id}] 별명 생성 실패 → 저장 생략")
            continue

        # ✅ INSERT with user_id 기준 덮어쓰기 + week 갱신
        insert_query = """
        INSERT INTO AnonymousNames (user_id, group_id, anonymous_name, week)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE 
            anonymous_name = VALUES(anonymous_name),
            group_id = VALUES(group_id),
            week = VALUES(week)
        """
        cursor.execute(insert_query, (user_id, group_id, nickname, week_index))

    # ✅ 종료 처리
    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
