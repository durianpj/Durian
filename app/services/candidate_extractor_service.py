import re
from pathlib import Path

# 기준 데이터 import
# DEPARTMENTS: 부서 목록
# JOB_GRADES: 직급 목록
# POSITIONS: 직책 목록
# TEAMS: 팀 목록
from common.hr_master_data import DEPARTMENTS, JOB_GRADES, POSITIONS, TEAMS
from common.hr_fields import FIELD_RULES

# 문자열 비교를 쉽게 하기 위해 공백 등을 정리하는 함수
# 예: "마케팅 부" -> "마케팅부"
from common.text_utils import compact_text
from app.services.question_service import is_self_question


# config/user_dictionary.txt 파일 경로
# 현재 파일 위치 기준으로 프로젝트 루트의 config 폴더를 찾아감
USER_DICTIONARY_PATH = Path(__file__).resolve().parents[2] / "config" / "user_dictionary.txt"

# 후보를 너무 많이 뽑지 않도록 타입별 최대 개수 제한
MAX_CANDIDATES_PER_TYPE = 20


# user_dictionary.txt를 매번 다시 읽지 않기 위한 캐시
# mtime: 파일 수정 시간
# terms: 읽어온 단어 목록
_user_dictionary_cache: dict = {
    "mtime": None,
    "terms": [],
}


def _unique_keep_order(values: list[str]) -> list[str]:
    """
    중복을 제거하되, 원래 순서는 유지한다.
    예: ["마케팅부", "인사부", "마케팅부"] -> ["마케팅부", "인사부"]
    """

    seen = set()
    result = []

    for value in values:
        # 빈 값이거나 이미 나온 값이면 건너뜀
        if not value or value in seen:
            continue

        seen.add(value)
        result.append(value)

    return result


def _read_user_dictionary_terms() -> list[str]:
    """
    config/user_dictionary.txt에서 후보 단어 목록을 읽어온다.

    이 파일은 OpenSearch Nori 사용자 사전으로도 사용될 수 있지만,
    여기서는 LLM 후보 추출용으로 표면 단어만 사용한다.

    예:
    user_dictionary.txt 안에
    마케팅부
    DX전략실

    이렇게 있으면 ["마케팅부", "DX전략실"]로 읽는다.
    """

    # 사전 파일이 없으면 빈 리스트 반환
    if not USER_DICTIONARY_PATH.exists():
        return []

    # 파일의 마지막 수정 시간 확인
    mtime = USER_DICTIONARY_PATH.stat().st_mtime

    # 파일 수정 시간이 이전과 같으면 캐시된 terms 재사용
    if _user_dictionary_cache["mtime"] == mtime:
        return _user_dictionary_cache["terms"]

    terms = []

    # user_dictionary.txt 읽기
    with USER_DICTIONARY_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            # 앞뒤 공백 제거
            term = line.strip()

            # 빈 줄이거나 주석 줄이면 제외
            if not term or term.startswith("#"):
                continue

            # 후보 단어로 추가
            terms.append(term)

    # 중복 제거 후, 긴 단어가 먼저 매칭되도록 길이 내림차순 정렬
    # 예: "마케팅부"가 "부"보다 먼저 잡히게 하기 위함
    terms = sorted(_unique_keep_order(terms), key=len, reverse=True)

    # 캐시에 저장
    _user_dictionary_cache["mtime"] = mtime
    _user_dictionary_cache["terms"] = terms

    return terms


def _get_field_terms() -> list[str]:
    """
    FIELD_RULES에 등록된 HR 필드 라벨도 LLM 후보 힌트로 사용한다.
    필드명은 질문 분석에 직접적인 힌트이므로 user_dictionary.txt 등록 여부에 의존하지 않는다.
    """

    terms = []

    for field_key, rule in FIELD_RULES.items():
        terms.append(field_key)

        label = rule.get("label")
        if label:
            terms.append(label)

        embedding_label = rule.get("embedding_label")
        if embedding_label:
            terms.append(embedding_label)

    aliases = [
        "계좌",
        "계좌번호",
        "토익",
        "토익점수",
        "TOEIC",
        "TOEIC점수",
        "평가",
        "인사평가",
        "인사고과",
        "고과",
        "징계",
        "징계이력",
        "급여",
        "연봉",
        "이전직장",
        "전직장",
        "이전회사",
        "전회사",
        "이전최종직급",
        "이전담당업무",
        "담당업무",
    ]

    terms.extend(aliases)

    return sorted(_unique_keep_order(terms), key=len, reverse=True)


ORG_ALIAS_CANDIDATES = [
    {
        "keywords": ["영업", "영업관련", "영업쪽", "영업담당"],
        "type": "department",
        "value": "영업부",
    },
    {
        "keywords": ["개발", "개발관련", "개발쪽", "개발담당"],
        "type": "department",
        "value": "개발부",
    },
    {
        "keywords": ["인사", "인사관련", "인사쪽", "인사담당"],
        "type": "department",
        "value": "인사부",
    },
    {
        "keywords": ["기획", "기획관련", "기획쪽", "기획담당"],
        "type": "department",
        "value": "기획부",
    },
    {
        "keywords": ["마케팅", "마케팅관련", "마케팅쪽", "마케팅담당"],
        "type": "department",
        "value": "마케팅부",
    },
    {
        "keywords": ["채용", "채용관련", "채용쪽", "채용담당"],
        "type": "team",
        "value": "채용팀",
    },
]


def _find_org_alias_candidates(question: str) -> dict[str, list[str]]:
    """
    사용자의 질문 텍스트에서 부서(Department) 및 팀(Team) 별칭 후보를 찾아 분류하는 함수
    """
    # 1. 텍스트 매칭 확률을 높이기 위해 질문 내 공백/특수문자 등을 제거(압축)
    compact_question = compact_text(question)
    
    # 결과를 담을 딕셔너리 초기화
    matched = {
        "departments": [],
        "teams": [],
    }

    # '인사팀' 등의 조직이 아닌, 일반 명사(인사고과 등)로 쓰여 제외해야 할 키워드 목록
    non_org_phrases = [
        "인사고과",
        "인사평가",
    ]

    # 미리 정의된 조직 별칭 규칙 목록(ORG_ALIAS_CANDIDATES)을 하나씩 순회
    for rule in ORG_ALIAS_CANDIDATES:
        # 공백 제거 없이 원본 질문에서 키워드를 찾는다.
        # compact_text를 쓰면 "방씨인 사람" → "방씨인사람"이 되어 "인사"가 오탐된다.
        matched_keyword = next(
            (
                keyword
                for keyword in sorted(rule["keywords"], key=len, reverse=True)
                if keyword in question
            ),
            None,
        )

        # 매칭된 키워드가 없다면 다음 규칙으로 패스
        if not matched_keyword:
            continue

        # [예외 처리] 매칭된 단어가 "인사"인데, "인사고과"나 "인사평가" 같은 맥락이라면
        # 실제 인사 부서를 찾는 게 아니므로 결과에서 제외
        if matched_keyword == "인사" and any(
            phrase in compact_question
            for phrase in non_org_phrases
        ):
            continue

        value = rule["value"]

        # 규칙 타입이 부서(department)이고, 유효한 부서 목록(DEPARTMENTS)에 존재하면 추가
        if rule["type"] == "department" and value in DEPARTMENTS:
            matched["departments"].append(value)

        # 규칙 타입이 팀(team)이고, 유효한 팀 목록(TEAMS)에 존재하면 추가
        if rule["type"] == "team" and value in TEAMS:
            matched["teams"].append(value)

    # 순서 보존하면서 중복 제거 후 최종 딕셔너리 반환
    return {
        "departments": _unique_keep_order(matched["departments"]),
        "teams": _unique_keep_order(matched["teams"]),
    }


def _find_terms_in_question(question: str, terms: list[str]) -> list[str]:
    """
    질문 안에 terms 목록의 단어가 포함되어 있는지 찾는다.

    예:
    question = "마케팅부 직원 알려줘"
    terms = ["마케팅부", "인사부"]

    결과:
    ["마케팅부"]
    """

    # 질문을 비교하기 쉬운 형태로 정리
    # 예: "마케팅 부 직원" -> "마케팅부직원"
    compact_question = compact_text(question)

    matched = []

    for term in terms:
        # 후보 단어도 같은 방식으로 정리
        # 예: "마케팅 부" -> "마케팅부"
        compact_term = compact_text(term)

        # 정리된 질문 안에 후보 단어가 포함되어 있으면 매칭
        if compact_term and compact_term in compact_question:
            matched.append(term)

        # 너무 많이 뽑지 않도록 제한
        if len(matched) >= MAX_CANDIDATES_PER_TYPE:
            break

    return matched


def _value_appears_outside_container_terms(
    question: str,
    value: str,
    container_terms: list[str],
) -> bool:
    compact_question = compact_text(question)
    compact_value = compact_text(value)

    if not compact_question or not compact_value:
        return False

    container_spans = []

    for term in container_terms:
        compact_term = compact_text(term)

        if (
            not compact_term
            or compact_term == compact_value
            or compact_value not in compact_term
        ):
            continue

        start = compact_question.find(compact_term)

        while start != -1:
            container_spans.append((start, start + len(compact_term)))
            start = compact_question.find(compact_term, start + 1)

    start = compact_question.find(compact_value)

    while start != -1:
        end = start + len(compact_value)

        if not any(span_start <= start and end <= span_end for span_start, span_end in container_spans):
            return True

        start = compact_question.find(compact_value, start + 1)

    return False


def _is_employee_noun_usage(question: str, value: str | None) -> bool:
    if value != "사원":
        return False

    compact_question = compact_text(question)

    if not compact_question.endswith("사원"):
        return False

    prefix = compact_question[:-len("사원")]

    if not prefix:
        return False

    if prefix.endswith(("부", "팀")):
        return True

    return any(org and org in prefix for org in DEPARTMENTS + TEAMS)


def _find_unknown_org_candidates(question: str, known_org_terms: set[str]) -> list[str]:
    """
    기준 데이터에는 없지만 조직명처럼 보이는 후보를 찾는다.

    예:
    질문: "AI전략실 직원 알려줘"
    known_org_terms에 "AI전략실"이 없다면
    unknown_orgs 후보로 잡을 수 있다.

    찾는 패턴:
    - ~본부
    - ~부서
    - ~부
    - ~팀
    - ~실
    """

    # 질문을 공백 제거 등 비교하기 쉬운 형태로 정리
    if is_self_question(question):
        return []

    compact_question = compact_text(question)

    candidates = []

    # 조직명처럼 보이는 단어 패턴 찾기
    # 예: 마케팅부, DX전략실, 데이터팀
    for match in re.finditer(r"[가-힣A-Za-z0-9]+(?:본부|부서|부|팀|실)", compact_question):
        value = match.group()
        start, end = match.span()

        if value.endswith("부") and re.search(r"\d+(?:살|세)?부$", value):
            after = compact_question[end:end + 2]
            if after.startswith("터"):
                continue

        # 이미 알고 있는 부서/팀/직급/직책이면 unknown 후보에서 제외
        if value in known_org_terms:
            continue

        # 너무 짧은 값은 제외
        if len(value) <= 1:
            continue

        candidates.append(value)

    # 중복 제거 후 최대 개수만 반환
    return _unique_keep_order(candidates)[:MAX_CANDIDATES_PER_TYPE]


def extract_question_candidates(question: str) -> dict:
    """
    LLM이 intent 분석을 하기 전에 질문에서 후보 정보를 먼저 추출한다.

    이 함수의 목적:
    - LLM에게 힌트를 주기 위함
    - 직원명/부서/팀/직급/직책 등을 미리 감지
    - 최종 검증은 나중에 normalize나 permission 처리 단계에서 다시 함

    즉, 여기서 뽑힌 후보가 무조건 정답이라는 뜻은 아니다.
    """

    # 질문이 비어 있으면 모든 후보를 빈 리스트로 반환
    if not question:
        return {
            "employee_names": [],
            "departments": [],
            "teams": [],
            "job_grades": [],
            "positions": [],
            "unknown_orgs": [],
            "terms": [],
        }

    try:
        # 직원명 목록은 hybrid_search_service에서 가져온다.
        # 예: ["김민수", "이서연", "박지훈"]
        from app.services.hybrid_search_service import get_employee_names

        # 질문 안에 직원명이 들어 있는지 찾는다.
        employee_names = _find_terms_in_question(question, get_employee_names())

    except Exception as error:
        # 직원명 후보 추출 실패해도 전체 질문 분석이 죽으면 안 되므로 경고만 출력
        print("[WARN] candidate employee name extraction failed:", error)
        employee_names = []

    # 질문 안에서 부서 후보 찾기
    org_alias_candidates = _find_org_alias_candidates(question)
    departments = _unique_keep_order(
        _find_terms_in_question(question, DEPARTMENTS)
        + org_alias_candidates["departments"]
    )

    # 질문 안에서 팀 후보 찾기
    teams = _unique_keep_order(
        _find_terms_in_question(question, TEAMS)
        + org_alias_candidates["teams"]
    )

    # 질문 안에서 직급 후보 찾기
    field_terms = _get_field_terms()
    job_grades = [
        job_grade
        for job_grade in _find_terms_in_question(question, JOB_GRADES)
        if _value_appears_outside_container_terms(
            question=question,
            value=job_grade,
            container_terms=field_terms,
        )
        and not _is_employee_noun_usage(question, job_grade)
    ]

    # 질문 안에서 직책 후보 찾기
    positions = _find_terms_in_question(question, POSITIONS)

    # 이미 알고 있는 조직/직급/직책 목록
    # unknown_orgs를 찾을 때 제외 기준으로 사용
    known_org_terms = set(DEPARTMENTS + TEAMS + JOB_GRADES + POSITIONS)

    # 기준 데이터에는 없지만 조직명처럼 보이는 후보 찾기
    # 예: "AI전략실", "신사업팀"
    unknown_orgs = _find_unknown_org_candidates(question, known_org_terms)

    # 이미 부서/팀/직급/직책으로 잡힌 후보들
    matched_known_org_terms = set(departments + teams + job_grades + positions)

    # user_dictionary.txt에서 읽은 단어 중 질문에 포함된 것 찾기
    # 단, 이미 다른 후보로 잡힌 값은 terms에서 제외
    dictionary_terms = []

    candidate_terms = _unique_keep_order(
        field_terms + _read_user_dictionary_terms()
    )

    for term in _find_terms_in_question(question, candidate_terms):
        compact_term = compact_text(term)

        if term in matched_known_org_terms or term in employee_names:
            continue

        if any(compact_term and compact_term in compact_text(org) for org in unknown_orgs):
            continue

        dictionary_terms.append(term)

    # LLM에게 넘길 후보 정보 반환
    return {
        "employee_names": employee_names,
        "departments": departments,
        "teams": teams,
        "job_grades": job_grades,
        "positions": positions,
        "unknown_orgs": unknown_orgs,
        "terms": dictionary_terms,
    }


def format_candidates_for_prompt(candidates: dict | None) -> str:
    """
    extract_question_candidates() 결과를 LLM 프롬프트에 넣기 좋은 문자열로 변환한다.

    예:
    candidates = {
        "departments": ["마케팅부"],
        "employee_names": [],
    }

    결과:
    - employee_name candidates: NULL
    - department candidates: 마케팅부
    """

    # 후보가 없으면 프롬프트에 넣을 기본 문구 반환
    if not candidates:
        return "No pre-detected candidates."

    # candidates dict의 key와 LLM에게 보여줄 label 매핑
    labels = [
        ("employee_names", "employee_name candidates"),
        ("departments", "department candidates"),
        ("teams", "team candidates"),
        ("job_grades", "job_grade candidates"),
        ("positions", "position candidates"),
        ("unknown_orgs", "unknown organization candidates"),
        ("terms", "business/domain term candidates"),
    ]

    lines = []

    for key, label in labels:
        # 해당 후보 목록 가져오기
        values = candidates.get(key) or []

        # 값이 있으면 콤마로 연결, 없으면 NULL 표시
        value_text = ", ".join(values) if values else "NULL"

        # 프롬프트에 들어갈 한 줄 생성
        lines.append(f"- {label}: {value_text}")

    # 여러 줄 문자열로 반환
    return "\n".join(lines)
