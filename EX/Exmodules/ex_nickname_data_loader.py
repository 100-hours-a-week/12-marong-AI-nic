# ex_nickname_data_loader.py
import pandas as pd

def load_grouped_nicknames(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    df['별명'] = df['별명'].str.replace(r'^\s*\d+\.\s*', '', regex=True)
    df = df.drop_duplicates()
    return df.groupby(['MBTI', 'Hobby'])['별명'].apply(list).to_dict()
