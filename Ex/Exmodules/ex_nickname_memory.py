class NicknameMemory:
    def __init__(self, cursor):
        self.cursor = cursor

    def is_used(self, user_id: int, group_id: int) -> bool:
        """해당 user/group 조합의 별명이 이미 존재하는지 확인"""
        self.cursor.execute(
            """
            SELECT COUNT(*) FROM AnonymousNames 
            WHERE user_id = %s AND group_id = %s
            """,
            (user_id, group_id)
        )
        return self.cursor.fetchone()[0] > 0

    def mark_used(self, user_id: int, group_id: int, nickname: str):
        """별명을 AnonymousNames에 저장"""
        self.cursor.execute(
            """
            INSERT INTO AnonymousNames (user_id, group_id, anonymous_name, week)
            VALUES (%s, %s, %s,NULL)
            """,
            (user_id, group_id, nickname)
        )
