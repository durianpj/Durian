import re
from common.text_utils import compact_text
from common.hr_fields import FIELD_RULES
from common.hr_master_data import (
    DEPARTMENTS,
    TEAMS,
    JOB_GRADES,
    POSITIONS,
)


# =========================
# 질문 판단 서비스
# =========================


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


def is_all_employee_query(question: str) -> bool:
    """
    질문이 특정 직원 1명이 아니라 전체 직원 집합을 대상으로 하는지 판단한다.
    """

    if not question:
        return False

    compact_question = compact_text(question)
    all_employee_keywords = [
        "모든직원",
        "전체직원",
        "전직원",
        "직원전체",
        "모든사원",
        "전체사원",
        "전사원",
        "사원전체",
        "직원들",
        "사원들",
    ]

    return any(keyword in compact_question for keyword in all_employee_keywords)


def is_org_work_owner_query(question: str) -> bool:
    """
    "인사 업무 담당자는 누구야"처럼 조직/업무 담당자를 묻는 질문은
    앞부분을 사람 이름으로 추정하면 안 된다.
    """

    if not question:
        return False

    compact_question = compact_text(question)
    org_keywords = set(DEPARTMENTS + TEAMS)

    for department in DEPARTMENTS:
        org_keywords.add(department.removesuffix("부"))

    org_keywords.update(["영업", "개발", "인사", "기획", "마케팅", "채용"])

    return (
        any(compact_question.startswith(keyword) for keyword in org_keywords if keyword)
        and "업무" in compact_question
        and any(keyword in compact_question for keyword in ["담당", "담당자", "누구"])
    )


def extract_name_like_prefix(question: str) -> str | None:
    """
    "홍길동 연봉", "없는사람 연봉"처럼 필드명 앞에 붙은 2~4글자 대상 후보를 찾는다.
    직원 목록 표현과 특정 직원 이름을 구분하기 위한 보조 함수다.
    """

    if not question:
        return None

    if is_all_employee_query(question):
        return None

    if is_org_work_owner_query(question):
        return None

    compact_question = compact_text(question)

    field_keywords = [
        "사원번호",
        "사번",
        "이름",
        "부서",
        "팀",
        "직급",
        "직책",
        "이메일",
        "메일",
        "전화번호",
        "연락처",
        "주소",
        "담당업무",
        "이전담당업무",
        "연봉",
        "급여",
        "평가",
        "고과",
        "인사고과",
    ]

    invalid_prefixes = {
        "내",
        "내이름",
        "내성명",
        "나의",
        "나의이름",
        "나의성명",
        "본인",
        "본인이름",
        "본인성명",
        "우리",
        "제이름",
        "제성명",
        "저의이름",
        "저의성명",
        "전체",
        "모든",
        "전부",
        "성과",
        "평가",
        "고과",
        "좋은",
        "높은",
        "낮은",
    }

    for keyword in field_keywords:
        index = compact_question.find(keyword)

        if index <= 0:
            continue

        prefix = compact_question[:index]
        prefix = re.sub(r"(은|는|이|가|을|를|의|님|씨)+$", "", prefix)

        if not re.fullmatch(r"[가-힣]{2,4}", prefix):
            continue

        if prefix in invalid_prefixes:
            continue

        if prefix in DEPARTMENTS or prefix in TEAMS or prefix in JOB_GRADES or prefix in POSITIONS:
            continue

        return prefix

    return None


def is_employee_collection_query(question: str) -> bool:
    """
    질문이 특정 직원 1명이 아니라 직원 목록/집합을 대상으로 하는지 판단한다.
    """

    if not question:
        return False

    compact_question = compact_text(question)

    if is_all_employee_query(question):
        return True

    if extract_name_like_prefix(question):
        return False

    collection_keywords = [
        "직원보여줘",
        "직원조회",
        "직원검색",
        "직원찾아",
        "직원알려줘",
        "직원목록",
        "직원리스트",
        "사원보여줘",
        "사원조회",
        "사원검색",
        "사원찾아",
        "사원알려줘",
        "사원목록",
        "사원리스트",
        "정보있는직원",
        "정보있는사원",
        "정보가있는직원",
        "정보가있는사원",
        "정보있",
        "입사한사람",
        "퇴사한사람",
        "퇴직한사람",
        "입사한직원",
        "퇴사한직원",
        "퇴직한직원",
        "입사자",
        "퇴사자",
        "퇴직자",
        "사람들",
        "사람",
    ]
    date_condition_keywords = ["입사", "퇴사", "퇴직"]
    collection_nouns = ["사람", "직원", "사원", "대상자"]

    if (
        any(keyword in compact_question for keyword in date_condition_keywords)
        and any(noun in compact_question for noun in collection_nouns)
    ):
        return True

    return any(keyword in compact_question for keyword in collection_keywords)

def get_field_label_words() -> list[str]:
    """
    FIELD_RULES에 등록된 사용자 표시용 필드명을 이름 제거 목록으로 만든다.

    예:
    FIELD_RULES["retirement_date"]["label"] = "퇴직일자"
    FIELD_RULES["retirement_date"]["embedding_label"] = "퇴직일자"

    그러면 질문에서 "퇴직일자"를 직원 이름으로 오해하지 않게 제거할 수 있다.
    """

    words = []

    for rule in FIELD_RULES.values():
        label = rule.get("label")
        embedding_label = rule.get("embedding_label")

        if label:
            words.append(label)

        if embedding_label:
            words.append(embedding_label)

    return words


def extract_employee_name(question: str) -> str | None:
    """
    질문에서 직원 이름으로 보이는 한글 이름을 추출한다.

    예:
    - 오민호 사원번호 알려줘 -> 오민호
    - 오민호 이메일 알려줘 -> 오민호

    주의:
    - 직원 목록/집합 질문에서는 이름을 추출하지 않는다.
      예: 2024년 퇴사한 사람 알려줘 -> None
    - 부서명, 팀명, 직책명, 필드명을 이름으로 오해하지 않는다.
    """

    if not question:
        return None

    if is_org_work_owner_query(question):
        return None

    name_like_prefix = extract_name_like_prefix(question)

    if name_like_prefix:
        return name_like_prefix

    # "퇴사한 사람", "입사한 직원" 같은 질문은 특정 1명 조회가 아니다.
    # 이런 경우 이름을 억지로 뽑으면 "년퇴사한" 같은 오탐이 생긴다.
    if is_employee_collection_query(question):
        return None

    cleaned = compact_text(question)

    # 연도 표현 제거
    # 예: "2024년퇴사한사람" -> "퇴사한사람"
    cleaned = re.sub(r"(19|20)\d{2}년?(도)?(에)?", "", cleaned)

    # 조건 표현이 남아 있으면 이름이 아니다.
    # 예: "퇴사한", "입사자", "계약직" 같은 말은 직원명이 아님
    not_name_fragments = [
        "입사",
        "퇴사",
        "퇴직",
        "입사한",
        "퇴사한",
        "퇴직한",
        "입사자",
        "퇴사자",
        "퇴직자",
        "계약직",
        "정규직",
    ]

    if any(word in cleaned for word in not_name_fragments):
        return None

    # 요청/연결/목록 표현
    # 필드명은 아래에서 FIELD_RULES 기준으로 자동 추가한다.
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
        "언제",
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
        "사원들",
        "사원",
        "팀원들",
        "팀원",
        "구성원",
        "사람들",
        "사람",
        "상사들",
        "상사",
        "상급자",
        "윗사람",
        "직속상사",
        "직속상관",
        "보고라인",
        "보고체계",
        "결재라인",
        "목록",
        "리스트",
        "종류",
        "전체",
        "모두",
        "모든",
        "전부",
        "정보있는",
        "정보가있는",
        "정보있",
        "정보",

        # 자주 쓰는 필드 별칭
        # FIELD_RULES에 없는 말만 최소한으로 둔다.
        "번호",
        "사번",
        "메일",
        "연락처",
        "주민번호",
        "급여",
        "은행",
        "계좌",
        "성과",
        "평가",
        "고과",
        "인사고과",
        "토익",
        "TOEIC",

        # 본인 표현
        "내",
        "나의",
        "본인",
        "나는",
        "난",
        "제",
        "저의",
        "우리",

        # 지시 대명사 — 이름이 아님(언어 구조상 다른 대상을 가리키는 단어)
        "그거",
        "그것",
        "이거",
        "이것",
        "그게",
        "이게",
    ]

    # FIELD_RULES에 등록된 필드 라벨을 자동으로 제거 목록에 추가한다.
    # 예: 사원번호, 이름, 부서, 입사일, 퇴직일자, 주소 등
    remove_words.extend(get_field_label_words())

    # 실제 조직명/직책명은 이름이 아니다.
    remove_words.extend(DEPARTMENTS)
    remove_words.extend(TEAMS)
    remove_words.extend(JOB_GRADES)
    remove_words.extend(POSITIONS)

    # 중복 제거
    remove_words = list(set(remove_words))

    # 긴 단어부터 제거해야 한다.
    # 예: "팀장"보다 "팀"을 먼저 지우면 "장"만 남을 수 있다.
    for word in sorted(remove_words, key=len, reverse=True):
        cleaned = cleaned.replace(word, "")

    # 끝에 붙은 조사/호칭 제거
    # 예: "오민호는" -> "오민호"
    stripped = re.sub(r"(은|는|이|가|을|를|의|님|씨|\?)+$", "", cleaned)

    if len(stripped) >= 2:
        cleaned = stripped

    # 남은 값에서 한글 2~4글자를 이름 후보로 본다.
    match = re.search(r"[가-힣]{2,4}", cleaned)

    if not match:
        return None

    name = match.group()

    # 그래도 혹시 남은 조직명/직책명/필드명은 이름으로 인정하지 않는다.
    invalid_names = set(DEPARTMENTS + TEAMS + JOB_GRADES + POSITIONS)
    invalid_names.update(remove_words)

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
