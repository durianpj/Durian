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

    for keyword in self_keywords:
        if keyword in question:
            return True

    return False


def extract_employee_name(question: str) -> str | None:
    """
    질문에서 직원 이름으로 보이는 한글 이름을 추출한다.

    예:
    - 임예은 부서 알려줘 -> 임예은
    - 한지아 직급 알려줘 -> 한지아
    """

    if not question:
        return None

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

    for particle in ["은", "는", "이", "가", "을", "를", "의", "?"]:
        cleaned = cleaned.replace(particle, "")

    match = re.search(r"[가-힣]{2,4}", cleaned)

    if match:
        return match.group()

    return None


def classify_org_question(question: str) -> str | None:
    """
    부서/팀/직책/조직 관련 질문 유형을 분류한다.

    반환값:
    - department_list: 부서 목록 질문
    - department_employee_search: 특정 부서/팀 직원 조회
    - manager_search: 팀장/상사 질문
    """

    department_list_keywords = [
        "부서 종류",
        "부서 목록",
        "부서 리스트",
        "부서 조회",
        "부서 알려줘",
        "부서 뭐 있어",
        "어떤 부서",
    ]

    manager_keywords = [
        "팀장",
        "상사",
        "관리자",
        "책임자",
    ]

    employee_list_keywords = [
        "직원",
        "팀원",
        "구성원",
        "사람",
        "찾아줘",
        "조회",
        "반환",
        "알려줘",
    ]

    if any(keyword in question for keyword in department_list_keywords):
        return "department_list"

    if any(keyword in question for keyword in manager_keywords):
        return "manager_search"

    if any(keyword in question for keyword in employee_list_keywords):
        if any(word in question for word in ["부", "팀"]):
            return "department_employee_search"

    return None


def extract_department_or_team(question: str) -> str | None:
    """
    질문에서 부서명 또는 팀명을 추출한다.

    현재 데이터 기준:
    - department = 부서
    - position = 팀
    """

    compact_question = question.replace(" ", "")

    names = [
        "인사부",
        "마케팅부",
        "개발부",
        "영업부",
        "재무부",
        "기획부",
        "인사팀",
        "마케팅팀",
        "개발팀",
        "영업팀",
        "재무팀",
        "기획팀",
    ]

    for name in names:
        if name in compact_question:
            return name

    return None