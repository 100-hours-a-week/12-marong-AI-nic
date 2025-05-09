# ── modules/utils.py ──────────────────────────────────────

def reorder_nicknames(generated: str, past_nicks: list[str]) -> list[str]:
    # 응답 문자열을 줄바꿈, 숫자, 쉼표 기준으로 분리
    raw = generated.replace(',', '\n').split('\n')
    nicknames = [n.strip("1234567890.:- \t") for n in raw if n.strip()]

    # 중복 제거 및 정렬
    seen = set()
    unique_nicknames = []
    for nick in nicknames:
        if nick and nick not in seen:
            seen.add(nick)
            unique_nicknames.append(nick)

    # 과거 닉네임은 맨 뒤로 보내기
    new_first = [n for n in unique_nicknames if n not in past_nicks]
    old_later = [n for n in unique_nicknames if n in past_nicks]
    return new_first + old_later
