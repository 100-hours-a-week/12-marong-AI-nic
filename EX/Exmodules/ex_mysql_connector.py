import mysql.connector

# ✅ 본인의 실제 DB 정보로 수정
def get_mysql_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",              # 🔁 실제 DB 사용자명
        password="3114",          # 🔁 실제 비밀번호
        database="marong"         # 🔁 실제 DB 이름
    )

# (옵션) 이건 지금은 사용 안 하지만 남겨도 됩니다
def fetch_user_mbti_and_hobby(cursor, user_id: int):
    query = "SELECT mbti, hobby FROM Users WHERE id = %s"
    cursor.execute(query, (user_id,))
    result = cursor.fetchone()
    if result:
        return result[0], result[1]
    else:
        return None

