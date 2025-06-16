# import mysql.connector
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
def fetch_user_group_id(cursor, user_id: int) -> int:
    cursor.execute(
        "SELECT group_id FROM UserGroups WHERE user_id = %s LIMIT 1",
        (user_id,)
    )
    row = cursor.fetchone()
    
    # 그룹이 없는 경우 기본 그룹(1) 반환
    if not row:
        print(f"[warn] user_id={user_id}의 그룹 정보가 없습니다. 기본 그룹(1)을 사용합니다.")
        return 1
        
    return row["group_id"]
