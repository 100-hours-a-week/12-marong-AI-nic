from datetime import datetime

# ✅ 주차 계산기 클래스
class GetWeekIndex:
    def __init__(self, target_date: datetime, base_date: datetime):
        self.target_date = target_date
        self.base_date = base_date

    def get(self) -> int:
        delta_days = (self.target_date - self.base_date).days
        return (delta_days // 7) + 1 if delta_days >= 0 else 0

# ✅ 사용 예시
if __name__ == "__main__":
    base_date = datetime(2025,1,1)  # 기준일: 1주차 시작일
    today = datetime.today()
    index = GetWeekIndex(today, base_date).get()
    print(f"📅 오늘은 기준일로부터 {index}주차입니다.")  