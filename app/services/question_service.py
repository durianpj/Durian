import re

# =========================
# 질문 판단 서비스
# =========================
# 사용자의 질문이 어떤 유형인지 판단하는 함수들을 모아두는 파일이다.
# main.py가 너무 길어지는 것을 막기 위해 분리했다.


def is_self_question(question: str) -> bool:
    """
    질문이 본인 정보 조회인지 판단한다.

    예:
    - 내 부서 알려줘
    - 나의 직급 알려줘
    - 본인 연봉 알려줘
    - 나는 어느 부서야?
    """

    # 질문이 비어 있으면 본인 질문이 아님
    if not question:
        return False

    # 본인 조회로 판단할 키워드 목록
    self_keywords = ["내", "나의", "본인", "나는", "난"]

    # 질문 안에 본인 조회 키워드가 하나라도 있으면 True 반환
    for keyword in self_keywords:
        if keyword in question:
            return True

    # 해당 키워드가 없으면 본인 질문이 아님
    return False

import re


def extract_employee_name(question: str) -> str | None:
    """
    질문에서 직원 이름으로 보이는 한글 이름을 추출한다.

    예:
    - 임예은 부서 알려줘 -> 임예은
    - 한지아 직급 알려줘 -> 한지아
    """

    if not question:
        return None

    # 질문에서 제거할 일반 단어들
    remove_words = [
        "알려줘",
        "말해줘",
        "뭐야",
        "무엇",
        "어디",
        "부서",
        "팀",
        "직급",
        "직책",
        "이름",
        "사원번호",
        "사번",
        "연봉",
        "급여",
        "성과",
        "평가",
        "주소",
        "내",
        "나의",
        "본인",
        "나는",
        "난",
    ]

    cleaned = question.replace(" ", "")

    for word in remove_words:
        cleaned = cleaned.replace(word, "")

    # 조사 제거
    for particle in ["은", "는", "이", "가", "을", "를", "의", "?"]:
        cleaned = cleaned.replace(particle, "")

    # 한글 2~4글자 이름 추출
    match = re.search(r"[가-힣]{2,4}", cleaned)

    if match:
        return match.group()

    return None