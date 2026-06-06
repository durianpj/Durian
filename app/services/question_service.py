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

    normalized = question.strip().replace(" ", "")

    # "사내", "내일"처럼 본인 조회가 아닌 단어 안의 "내"는 제외한다.
    if normalized.startswith("내일"):
        return False

    # 본인 조회는 보통 질문 앞에서 "내/나의/본인/나는/난"으로 시작한다.
    self_prefixes = ["나의", "나는", "본인", "내", "난"]

    for prefix in self_prefixes:
        if normalized.startswith(prefix):
            return True

    # 해당 키워드가 없으면 본인 질문이 아님
    return False


def is_department_list_question(question: str) -> bool:
    """
    부서의 종류나 목록을 묻는 질문인지 판단한다.

    예:
    - 부서 종류 알려줘
    - 부서 목록 알려줘
    - 어떤 부서가 있어?
    """

    if not question:
        return False

    normalized = question.replace(" ", "")

    if "부서" not in normalized:
        return False

    list_keywords = ["종류", "목록", "리스트", "어떤", "전체", "전부"]

    return any(keyword in normalized for keyword in list_keywords)


def is_department_members_question(question: str) -> bool:
    """
    특정 부서의 직원/팀원 목록을 묻는 질문인지 판단한다.

    예:
    - 인사부 팀원 다 알려줘
    - 마케팅부 직원 전체 알려줘
    """

    if not question:
        return False

    normalized = question.replace(" ", "")
    departments = ["마케팅부", "기획부", "인사부", "개발부", "영업부", "재무부"]
    member_keywords = ["직원", "팀원", "팀장", "구성원", "사람"]
    list_keywords = ["다", "전체", "전부", "목록", "리스트", "알려줘", "보여줘"]

    has_department = any(department in normalized for department in departments)
    has_member_keyword = any(keyword in normalized for keyword in member_keywords)
    has_list_keyword = any(keyword in normalized for keyword in list_keywords)

    return has_department and has_member_keyword and has_list_keyword


def is_supervisor_question(question: str) -> bool:
    """
    상사나 윗 직급자를 묻는 질문인지 판단한다.

    예:
    - 내 상사들 알려줘
    - 나의 윗사람 알려줘
    - 내 관리자 누구야?
    """

    if not question:
        return False

    normalized = question.replace(" ", "")
    supervisor_keywords = ["상사", "윗사람", "관리자", "윗직급"]

    return any(keyword in normalized for keyword in supervisor_keywords)


def is_phone_number_question(question: str) -> bool:
    """
    전화번호/연락처를 묻는 질문인지 판단한다.
    사원번호와 계좌번호는 별도 항목이므로 제외한다.
    """

    if not question:
        return False

    normalized = question.replace(" ", "")

    if "사원번호" in normalized or "계좌번호" in normalized:
        return False

    phone_keywords = ["전화번호", "연락처", "휴대폰", "번호"]

    return any(keyword in normalized for keyword in phone_keywords)


def is_basic_info_question(question: str) -> bool:
    """
    기본정보 요약을 묻는 질문인지 판단한다.
    """

    if not question:
        return False

    normalized = question.replace(" ", "")

    return "기본정보" in normalized


def extract_employee_name(question: str) -> str | None:
    """
    질문에서 직원 이름으로 보이는 한글 이름을 추출한다.

    예:
    - 임예은 부서 알려줘 -> 임예은
    - 한지아 직급 알려줘 -> 한지아
    """

    if not question:
        return None

    normalized = question.replace(" ", "")

    if is_self_question(normalized):
        return None

    target_fields = (
        r"(?:부서|팀|직급|직책|이름|사원번호|사번|전화번호|연락처|휴대폰|번호|"
        r"입사일|연봉|급여|성과|평가|주소)"
    )
    name_match = re.search(rf"^([가-힣]{{2,4}}?)(?:의|는|이|가|을|를){target_fields}", normalized)

    if not name_match:
        name_match = re.search(rf"^([가-힣]{{2,4}}?){target_fields}", normalized)

    if name_match:
        return name_match.group(1)

    # 질문에서 제거할 일반 단어들
    remove_words = [
        "알려줘",
        "말해줘",
        "찾아줘",
        "직원",
        "일정",
        "뭐야",
        "무엇",
        "어디",
        "부서",
        "팀",
        "직급",
        "직책",
        "사원",
        "대리",
        "과장",
        "차장",
        "부장",
        "이름",
        "사원번호",
        "사번",
        "전화번호",
        "연락처",
        "휴대폰",
        "번호",
        "입사일",
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

    cleaned = normalized

    for department in ["마케팅부", "기획부", "인사부", "개발부", "영업부", "재무부"]:
        cleaned = cleaned.replace(department, "")

    for word in remove_words:
        cleaned = cleaned.replace(word, "")

    # 이름 뒤에 붙은 조사는 끝에 있을 때만 제거한다.
    for particle in ["의", "은", "는", "이", "가", "을", "를", "?"]:
        if len(cleaned) > 2 and cleaned.endswith(particle):
            cleaned = cleaned[: -len(particle)]

    # 한글 2~4글자 이름 추출
    match = re.search(r"[가-힣]{2,4}", cleaned)

    if match:
        return match.group()

    return None
