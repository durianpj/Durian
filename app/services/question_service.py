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