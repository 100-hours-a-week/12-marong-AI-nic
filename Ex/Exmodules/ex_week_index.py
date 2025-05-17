from datetime import datetime

class GetWeekIndex:
    def __init__(self, target_date: datetime, base_date: datetime):
        self.target_date = target_date
        self.base_date = base_date

    def get(self) -> int:
        delta_days = (self.target_date - self.base_date).days
        return (delta_days // 7) + 1 if delta_days >= 0 else 0
