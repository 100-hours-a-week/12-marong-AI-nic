# ── modules/utils.py ──────────────────────────────────────

def reorder_nicknames(generated: str, past_nicks: list[str]) -> list[str]:
    """
    응답 문자열에서 별명만 추출하고,
    과거 별명은 뒤로 보내며,
    "assistant", "별명:" 등의 불필요한 토큰을 필터링합니다.
    """
    # 번호, 쉼표를 줄바꿈으로 통일하고 분리
    raw = generated.replace(',', '\n').split('\n')

    # 필터링할 불용어 목록
    blacklist = {"assistant", "tool_list", "system", "user", "별명", "별명:"}

    nicknames = []
    for line in raw:
        token = line.strip()
        if not token:
            continue
        # 순번 제거: 숫자, 점, 콜론, 하이픈, 공백
        clean = token.strip("1234567890.:- \t")
        # 소문자로 검사해 불용어 제거
        low = clean.lower()
        if any(b in low for b in blacklist):
            continue
        nicknames.append(clean)

    # 중복 제거
    seen = set()
    unique_nicknames = []
    for nick in nicknames:
        if nick and nick not in seen:
            seen.add(nick)
            unique_nicknames.append(nick)

    # 과거 별명은 맨 뒤로 보내기
    new_first = [n for n in unique_nicknames if n not in past_nicks]
    old_later = [n for n in unique_nicknames if n in past_nicks]
    return new_first + old_later
