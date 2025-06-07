import mysql.connector
from typing import Optional

# ✅ 추천된 별명을 AnonymousNames 테이블에 저장
def save_anonymous_name(
    cursor,
    user_id: int,
    group_id: int,
    week: int,
    anonymous_name: str
) -> None:
    query = """
        INSERT INTO AnonymousNames (user_id, group_id, week, anonymous_name)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE anonymous_name = VALUES(anonymous_name)
    """
    cursor.execute(query, (user_id, group_id, week, anonymous_name))

# ✅ 사용자 그룹 ID 조회 함수
def fetch_user_group_id(cursor, user_id: int) -> Optional[int]:
    cursor.execute(
        "SELECT group_id FROM UserGroups WHERE user_id = %s LIMIT 1",
        (user_id,)
    )
    row = cursor.fetchone()
    return row["group_id"] if row else None
