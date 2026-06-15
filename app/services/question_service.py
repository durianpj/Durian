import re

from app.services.org_policy_service import (
    DEPARTMENTS,
    TEAMS,
    POSITIONS,
)


# =========================
# 질문 판단 서비스
# =========================


def compact_text(text: str) -> str:
    """
    질문 비교용으로 공백을 제거한다.
    예: "인사부 알려줘" -> "인사부알려줘"
    """

    if not text:
        return ""

    return re.sub(r"\s+", "", text)


def is_self_question(question: str) -> bool:
    """
    질문이 본인 정보 조회인지 판단한다.

    예:
    - 내 부서 알려줘
    - 나의 직급 알려줘
    - 본인 연봉 알려줘
    - 나는 어느 부서야?
    """

    if not question:
        return False

    self_keywords = [
        "내",
        "나의",
        "본인",
        "나는",
        "난",
        "제",
        "저의",
        "내꺼",
        "우리",
    ]
    #any 하나라도 true면 true, 모두 false면 false / question에 self_keywords 중 하나라도 있으면 true, 없으면 false
    return any(keyword in question for keyword in self_keywords)


def extract_employee_name(question: str) -> str | None:
    """
    질문에서 직원 이름으로 보이는 한글 이름을 추출한다.

    예:
    - 오민호 사원번호 알려줘 -> 오민호
    - 오민호 이메일 알려줘 -> 오민호

    주의:
    - 이메일, 부서, 직책, 팀장 같은 필드명/조건명을 이름으로 오해하면 안 된다.
    """

    if not question:
        return None

    cleaned = compact_text(question)

    remove_words = [
        # 요청 표현
        "알려줘",
        "말해줘",
        "보여줘",
        "조회해줘",
        "조회",
        "검색",
        "찾아줘",
        "찾아",
        "반환",
        "뭐야",
        "무엇",
        "어디",
        "어떻게",
        "누구야",
        "누구",
        "그리고",

        # 연결 표현
        "랑",
        "이랑",
        "와",
        "과",
        "하고",
        "및",

        # 목록/집합 표현
        "직원들",
        "직원",
        "팀원들",
        "팀원",
        "구성원",
        "사람들",
        "사람",
        "목록",
        "리스트",
        "종류",
        "전체",
        "모두",
        "모든",
        "전부",

        # 필드명
        "사원번호",
        "사번",
        "이름",
        "회사명",
        "사업장위치",
        "부서",
        "팀",
        "직급",
        "직책",
        "이메일",
        "메일",
        "전화번호",
        "연락처",
        "주소",
        "입사일",
        "근속기간",
        "생년월일",
        "주민등록번호",
        "주민번호",
        "학력",
        "출신대학",
        "학점",
        "채용경로",
        "계약형태",
        "이전직장명",
        "이전최종직급",
        "이전담당업무",
        "연봉",
        "급여",
        "급여은행",
        "은행",
        "계좌번호",
        "계좌",
        "잔업시간",
        "미사용휴가일수",
        "성과점수",
        "성과",
        "평가",
        "인사평가",
        "고과",
        "인사고과",
        "자격증",
        "토익",
        "TOEIC",
        "포상이력",
        "징계이력",
        "징계사유",

        # 본인 표현
        "내",
        "나의",
        "본인",
        "나는",
        "난",
        "제",
        "저의",
        "우리",
    ]

    # 실제 조직명/직책명은 이름이 아니다.
    remove_words.extend(DEPARTMENTS)
    remove_words.extend(TEAMS)
    remove_words.extend(POSITIONS)

    # 긴 단어부터 제거해야 한다.
    # 예: "팀장"보다 "팀"을 먼저 지우면 "장"만 남을 수 있다.
    for word in sorted(remove_words, key=len, reverse=True):
        cleaned = cleaned.replace(word, "")

    # 조사는 전체 replace 하지 말고 끝부분만 제거한다.
    cleaned = re.sub(r"(은|는|이|가|을|를|의|님|씨|\?)+$", "", cleaned)

    match = re.search(r"[가-힣]{2,4}", cleaned)

    if not match:
        return None

    name = match.group()

    invalid_names = set(DEPARTMENTS + TEAMS + POSITIONS)
    invalid_names.update(
        [
            "부서",
            "직원",
            "팀원",
            "팀장",
            "직책",
            "직급",
            "이메일",
            "주소",
            "연봉",
            "평가",
        ]
    )

    if name in invalid_names:
        return None

    return name


def extract_department_or_team(question: str) -> str | None:
    """
    질문에서 부서명 또는 팀명을 추출한다.

    현재 데이터 기준:
    - department = 부서
    - team = 팀
    - position = 직책
    """

    if not question:
        return None

    compact_question = compact_text(question)

    for name in DEPARTMENTS + TEAMS:
        if name in compact_question:
            return name

    return None


def extract_position(question: str) -> str | None:
    """
    질문에서 직책 조건을 추출한다.

    예:
    - 팀장 누구야 -> 팀장
    - 인사부 팀장 알려줘 -> 팀장
    """

    if not question:
        return None

    compact_question = compact_text(question)

    for position in POSITIONS:
        if position in compact_question:
            return position

    return None


def extract_employee_id(question: str) -> str | None:
    """
    질문에서 사번 형태를 추출한다.

    예:
    - EMP0001 부서 알려줘 -> EMP0001
    - emp0609 연봉 알려줘 -> EMP0609
    """

    if not question:
        return None

    match = re.search(r"EMP\d+", question.upper())

    if match:
        return match.group()

    return None


def classify_org_question(question: str) -> str | None:
    """
    예전 키워드 기반 조직 질문 분류 함수.

    현재 main.py의 task 기반 흐름에서는 거의 사용하지 않는다.
    혹시 다른 파일에서 import하고 있을 수 있으므로 호환용으로 남겨둔다.
    """

    if not question:
        return None

    compact_question = compact_text(question)

    if any(word in compact_question for word in ["부서종류", "부서목록", "부서리스트", "어떤부서", "부서뭐"]):
        return "department_list"

    if any(position in compact_question for position in POSITIONS):
        return "manager_search"

    if any(name in compact_question for name in DEPARTMENTS + TEAMS):
        return "department_employee_search"

    return None