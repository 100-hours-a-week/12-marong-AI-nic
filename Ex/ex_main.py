# ex_main.py

import os
import sys

# 🔧 모듈 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "Exmodules")))

from Exmodules.ex_nickname_data_loader import load_grouped_nicknames
from Exmodules.ex_nickname_memory import NicknameMemory
from Exmodules.ex_nickname_retriever import NicknameRetriever
from Exmodules.ex_mysql_connector import get_mysql_connection


# ✅ 1. CSV 경로 수정 (정확한 실제 경로 반영)
csv_path = os.path.join(os.path.dirname(__file__), "Exmodules", "Exdata", "Exdata.csv")
grouped_nicknames = load_grouped_nicknames(csv_path)

# ✅ 2. DB 연결
conn = get_mysql_connection()
cursor = conn.cursor()

# ✅ 3. 별명 미지정 대상 사용자 조회
query = """
SELECT 
  u.id AS user_id,
  ug.group_id,
  ug.week,
  sm.mbti,
  sh.hobby_name AS hobby
FROM Users u
JOIN UserGroups ug ON u.id = ug.user_id
LEFT JOIN SurveyMBTI sm ON u.id = sm.user_id
LEFT JOIN SurveyHobby sh ON u.id = sh.user_id
WHERE NOT EXISTS (
    SELECT 1 FROM AnonymousNames an 
    WHERE an.user_id = u.id AND an.group_id = ug.group_id AND an.week = ug.week
);
"""

cursor.execute(query)
targets = cursor.fetchall()

# ✅ 4. 메모리 및 리트리버 초기화
memory = NicknameMemory(cursor)
retriever = NicknameRetriever(grouped_nicknames, memory)

# ✅ 5. 별명 생성 및 저장
for user_id, group_id, week, mbti, hobby in targets:
    if not mbti or not hobby:
        print(f"[errero] [user_id={user_id}] MBTI 또는 취미 누락 → 생략")
        continue

    nickname = retriever.get_nickname(mbti, hobby, user_id, group_id, week)
    print(f"✅ [user_id={user_id}, group={group_id}, week={week}] → {nickname}")

# ✅ 6. 종료 처리
conn.commit()
cursor.close()
conn.close()
