# openai_fallback.py

import openai
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional
import time
import random

# 환경 변수 로딩
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_nickname_with_openai(mbti_keywords: List[str], hobby_keywords: List[str], group_id: int, used_nicknames: set) -> Optional[str]:
    """
    OpenAI API를 사용하여 닉네임을 생성합니다.
    
    Args:
        mbti_keywords: MBTI 키워드 리스트
        hobby_keywords: 취미 키워드 리스트
        group_id: 그룹 ID
        used_nicknames: 이미 사용된 닉네임들
    
    Returns:
        생성된 닉네임 또는 None (실패 시)
    """
    
    # 프롬프트 구성
    prompt = f"""
당신은 창의적인 별명을 만드는 전문가입니다. 아래의 성격(MBTI)과 취미 키워드를 참고해, 
동물이나 귀여운 캐릭터와 조합된 참신한 한글 별명을 만들어주세요.

[중요 규칙]
1. 동물이나 귀여운 상상 캐릭터를 반드시 포함해야 합니다.
2. 동물이 포함되지 않은 닉네임은 절대 생성하지 마세요.
3. 의성어, 의태어, 감성어를 활용해 개성 있게 만드세요.
4. 별명은 한글로만, 공백 없이 8자 이내로 작성하세요.
5. 반드시 **별명** 형식으로 출력하세요.

[MBTI 키워드]: {', '.join(mbti_keywords[:8])}
[취미 키워드]: {', '.join(hobby_keywords[:4])}

[창의적인 조합 방법 - 동물 필수 포함]
1. 동물/캐릭터 + 성격 특징 (예: **활발한토끼**, **조용한부엉이**)
2. 의성어 + 동물/캐릭터 (예: **뽀글곰이**, **깡총토끼**)
3. 감성어 + 동물/캐릭터 (예: **달콩다람이**, **귀여운강아지**)
4. 취미 + 동물/캐릭터 (예: **달리기강아지**, **독서부엉이**)
5. 성격 + 취미 + 동물/캐릭터 (예: **활발달리기고양이**, **조용독서거북이**)

[사용 가능한 동물 키워드 예시]
토끼류: 토끼, 토깽이, 깡총이, 버니, 래빗
강아지류: 강아지, 멍멍이, 퍼피, 도그
고양이류: 고양이, 냥이, 냥냥이, 캣
조류: 펭귄, 부엉이, 나비, 까마귀, 앵무새, 독수리, 매
포유류: 코끼리, 사자, 호랑이, 원숭이, 말, 양, 소
해양생물: 고래, 돌고래, 물범, 게, 새우, 조개
곤충류: 나비, 벌, 잠자리, 무당벌레
기타: 개구리, 거북이, 달팽이, 지렁이, 해파리

[금지사항]
- 동물이 포함되지 않은 닉네임 (예: 성격, 키워드, 홍길동 등)
- 일반적인 사람 이름이나 성격만 표현한 닉네임

위 규칙과 조합 방법을 참고하여, 주어진 키워드에 맞는 창의적인 별명 8개를 만들어주세요. 
각 별명은 반드시 동물을 포함하고 **별명** 형식으로 출력해주세요.
"""

    try:
        # OpenAI API 호출
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # 또는 "gpt-4" 사용 가능
            messages=[
                {"role": "system", "content": "당신은 창의적인 별명을 만드는 전문가입니다. 동물을 포함한 참신한 한글 별명을 생성해주세요."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.8,
            top_p=0.9
        )
        
        # 응답에서 닉네임 추출
        content = response.choices[0].message.content
        print(f"[OpenAI] 응답: {content}")
        
        # 닉네임 파싱 (기존 result_parser와 동일한 로직)
        import re
        pattern = r"\d+\.\s*\*\*([가-힣a-zA-Z\s]+?)\*\*\s*(?:[:(].*)?$"
        matches = []
        
        for line in content.splitlines():
            match = re.search(pattern, line)
            if match:
                name = match.group(1).strip()
                if (len(name) <= 10 and 
                    re.fullmatch(r"[가-힣a-zA-Z\s]+", name) and 
                    '별명' not in name and 
                    '취미' not in name):
                    matches.append(name)
        
        # 중복 제거
        unique_matches = list(dict.fromkeys(matches))
        
        # 사용되지 않은 닉네임 찾기
        for name in unique_matches:
            if name not in used_nicknames:
                print(f"[OpenAI] 성공적으로 닉네임 생성: {name}")
                return name
        
        print(f"[OpenAI] 모든 후보가 이미 사용됨: {unique_matches}")
        return None
        
    except Exception as e:
        print(f"[OpenAI] API 호출 실패: {e}")
        return None

def retry_failed_nicknames_with_openai(failed_combinations: List[Dict], cursor) -> Dict:
    """
    실패한 닉네임 조합들을 OpenAI API로 재시도합니다.
    
    Args:
        failed_combinations: 실패한 조합 리스트
        cursor: 데이터베이스 커서
    
    Returns:
        성공/실패 통계
    """
    
    if not openai.api_key:
        print("[OpenAI] OpenAI API 키가 설정되지 않았습니다.")
        return {"success": 0, "failed": len(failed_combinations)}
    
    print(f"\n[OpenAI] {len(failed_combinations)}개의 실패한 닉네임을 OpenAI로 재시도합니다...")
    
    success_count = 0
    failed_count = 0
    
    for i, item in enumerate(failed_combinations, 1):
        user_id = item['user_id']
        mbti_keywords = item['mbti_keywords']
        hobby_keywords = item['hobby_keywords']
        
        print(f"\n[OpenAI] 재시도 {i}/{len(failed_combinations)} - user_id: {user_id}")
        
        # 그룹 ID 조회 (UserGroups 테이블에서 조회)
        cursor.execute("SELECT group_id FROM UserGroups WHERE user_id = %s LIMIT 1", (user_id,))
        result = cursor.fetchone()
        if not result:
            print(f"[OpenAI] user_id {user_id}의 그룹 정보를 찾을 수 없습니다.")
            failed_count += 1
            continue
        group_id = result['group_id']
        
        # 그룹에서 사용된 닉네임 조회
        cursor.execute("""
            SELECT anonymous_name FROM AnonymousNames 
            WHERE group_id = %s AND week = (SELECT MAX(week) FROM AnonymousNames)
        """, (group_id,))
        used_nicknames = {row['anonymous_name'] for row in cursor.fetchall()}
        
        # OpenAI로 닉네임 생성
        nickname = generate_nickname_with_openai(mbti_keywords, hobby_keywords, group_id, used_nicknames)
        
        if nickname:
            try:
                # 닉네임 저장
                from utils.week_index import GetWeekIndex
                from datetime import datetime, date
                week = GetWeekIndex(datetime.today(), datetime(2025, 1, 6)).get()
                
                cursor.execute("""
                    INSERT INTO AnonymousNames (user_id, group_id, week, anonymous_name) 
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE anonymous_name = VALUES(anonymous_name)
                """, (user_id, group_id, week, nickname))
                
                cursor.connection.commit()
                print(f"[OpenAI] 닉네임 저장 성공: {nickname}")
                success_count += 1
                
            except Exception as e:
                print(f"[OpenAI] 닉네임 저장 실패: {e}")
                failed_count += 1
        else:
            print(f"[OpenAI] 닉네임 생성 실패")
            failed_count += 1
        
        # API 호출 간격 조절
        time.sleep(1)
    
    print(f"\n[OpenAI] 재시도 완료 - 성공: {success_count}, 실패: {failed_count}")
    return {"success": success_count, "failed": failed_count} 