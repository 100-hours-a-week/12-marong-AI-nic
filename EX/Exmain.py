# EXmain.py

import os
from Exmodules.ex_mysql_connector import get_mysql_connection, fetch_user_mbti_and_hobby
from Exmodules.ex_nickname_data_loader import load_grouped_nicknames
from Exmodules.ex_nickname_memory import NicknameMemory
from Exmodules.ex_nickname_retriever import NicknameRetriever

# CSV 경로
csv_path = os.path.join(os.path.dirname(__file__), "Exdata.csv")
grouped_nicknames = load_grouped_nicknames(csv_path)

# DB 연결
conn = get_mysql_connection()
cursor = conn.cursor()

# 테스트용 사용자 ID
user_id = 1
user_info = fetch_user_mbti_and_hobby(cursor, user_id)
if not user_info:
    raise ValueError("사용자 정보를 찾을 수 없습니다.")
mbti, hobby = user_info

# 리트리버 실행
memory = NicknameMemory(cursor)
retriever = NicknameRetriever(grouped_nicknames, memory)
nickname = retriever.get_nickname(mbti, hobby, user_id)

print(f"[{mbti} / {hobby}] → 추천된 별명: {nickname}")

# 마무리
conn.commit()
cursor.close()
conn.close()
