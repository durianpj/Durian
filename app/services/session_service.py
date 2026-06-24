import json
import re

from common.hr_master_data import DEPARTMENTS, JOB_GRADES, POSITIONS, TEAMS
from app.services.hybrid_search_service import employee_name_exists
from app.services.llm_service import call_llm_completion
from app.services.question_service import (
    compact_text,
    extract_employee_id,
    extract_employee_name,
    is_self_question,
)


conversation_memory = {}
current_session_employee_id = None

def reset_memory_if_employee_changed(requester_employee_id: str) -> None:
    """
    요청자 employee_id가 이전 요청과 달라지면 세션 메모리를 전체 초기화한다.

    예:
    - 이전 요청자: EMP0070
    - 현재 요청자: EMP0001
    -> conversation_memory 전체 삭제

    이유:
    Swagger/FastAPI 테스트 중 employee_id를 바꿔가며 호출하면
    이전 요청자의 대화 기억이 다음 요청에 섞일 수 있기 때문이다.
    """

    global current_session_employee_id

    if not requester_employee_id:
        return

    requester_employee_id = requester_employee_id.strip().upper()

    if current_session_employee_id is None:
        current_session_employee_id = requester_employee_id
        return

    if current_session_employee_id != requester_employee_id:
        conversation_memory.clear()
        current_session_employee_id = requester_employee_id
        print("[DEBUG] conversation memory reset by employee_id change:", requester_employee_id)

NOT_TARGET_NAMES = {
    "계약직",
    "정규직",
    "평가",
    "고과",
    "우수",
    "보통",
    "미흡",
    "양호",
    "탁월",
    "부진",
    "연봉",
    "부서",
    "직급",
    "직책",
    "상사",
    "상사들",
    "상급자",
    "윗사람",
    "직속상사",
    "직속상관",
    "보고라인",
    "보고체계",
    "결재라인",
    "이메일",
    "전화번호",
    "주소",
    "사번",
    "사원번호",
}

FOLLOWUP_FIELD_KEYWORDS = [
    "사원번호",
    "사번",
    "이름",
    "성별",
    "나이",
    "생년월일",
    "주민등록번호",
    "주민번호",
    "병역",
    "입사일",
    "근속기간",
    "학력",
    "출신대학",
    "학점",
    "채용경로",
    "계약형태",
    "이전직장",
    "회사명",
    "사업장",
    "부서",
    "팀",
    "직급",
    "직책",
    "퇴직구분",
    "퇴직일자",
    "이메일",
    "메일",
    "전화번호",
    "연락처",
    "주소",
    "연봉",
    "월급",
    "급여",
    "잔업시간",
    "휴가",
    "급여은행",
    "계좌번호",
    "보험",
    "성과점수",
    "평가",
    "고과",
    "자격증",
    "토익",
    "포상",
    "징계",
]


def is_field_only_followup_question(question: str) -> bool:
    """
    대상 없이 HR 필드 하나만 이어서 묻는 짧은 후속 질문인지 확인한다.

    예:
    - 입사일은?
    - 연봉은?
    - 직책은?
    """

    if not question or is_self_question(question) or has_explicit_target(question):
        return False

    compact_question = compact_text(question)

    if not compact_question:
        return False

    # 직원 목록·조건 검색 표현이 있으면 특정 직원 후속 질문으로 보지 않는다.
    collection_keywords = [
        "직원",
        "사원",
        "사람",
        "목록",
        "리스트",
        "종류",
        "몇명",
        "인원",
        "찾아줘",
        "검색",
    ]
    if any(keyword in compact_question for keyword in collection_keywords):
        return False

    request_words = [
        "알려줘",
        "보여줘",
        "조회해줘",
        "뭐야",
        "무엇이야",
        "어때",
    ]
    normalized_question = compact_question

    for word in request_words:
        normalized_question = normalized_question.replace(word, "")

    normalized_question = normalized_question.rstrip("?!.")
    normalized_question = re.sub(
        r"(은|는|이|가|을|를|도)$",
        "",
        normalized_question,
    )

    return normalized_question in FOLLOWUP_FIELD_KEYWORDS


def has_explicit_target(question: str) -> bool:
    """
    질문 안에 조회 대상이 명시되어 있는지 확인한다.
    """

    if not question:
        return False

    
    compact_question = compact_text(question)

    if extract_employee_id(question):
        return True

    employee_name = extract_employee_name(question)

    if employee_name and (employee_name not in NOT_TARGET_NAMES):
        return True

    #compact_question = 공백 제거된 질문 / DEPARTMENTS, TEAMS, JOB_GRADES, POSITIONS에 있는 값이 compact_question에 하나라도 있으면 True, 없으면 False
    for value in DEPARTMENTS + TEAMS + JOB_GRADES + POSITIONS:
        if value in compact_question:
            return True

    return False


def is_followup_question(question: str) -> bool:
    """
    이전 대상을 이어서 묻는 후속 질문인지 확인한다.

    예:
    - 그 사람 이메일도 알려줘
    - 이메일도 알려줘
    - 그 팀 계약직도 알려줘
    """

    if not question:
        return False

    compact_question = compact_text(question)

    followup_keywords = [
        "그사람",
        "그분",
        "그직원",
        "해당직원",
        "방금사람",
        "방금직원",
        "아까사람",
        "아까직원",
        "그팀",
        "해당팀",
        "그부서",
        "해당부서",
        "거기",
    ]

    if any(keyword in compact_question for keyword in followup_keywords):
        return True
    
    # 명시된 대상이 없고, 질문에 "도"가 포함되어 있으면 후속 질문으로 판단한다.
    return ("도" in compact_question) and (not has_explicit_target(question))


def has_memory_applicable_keyword(question: str) -> bool:
    """
    이전 대상을 붙여 해석할 수 있는 HR 질문인지 확인한다.
    """

    compact_question = compact_text(question)

    keywords = [
        "사원번호",
        "사번",
        "이름",
        "성별",
        "나이",
        "생년월일",
        "주민등록번호",
        "주민번호",
        "병역",
        "입사일",
        "근속기간",
        "학력",
        "출신대학",
        "학점",
        "채용경로",
        "계약형태",
        "계약직",
        "정규직",
        "이전직장",
        "회사명",
        "사업장",
        "부서",
        "팀",
        "직급",
        "직책",
        "퇴직",
        "이메일",
        "메일",
        "전화번호",
        "연락처",
        "주소",
        "연봉",
        "잔업",
        "휴가",
        "급여은행",
        "계좌번호",
        "보험",
        "성과점수",
        "평가",
        "고과",
        "자격증",
        "월급",
        "연봉",
        "급여",
        "봉급",
        "보수",
        "TOEIC",
        "토익",
        "포상",
        "징계",
    ]

    return any(keyword in compact_question for keyword in keywords)


def is_condition_or_collection_question(question: str) -> bool:
    """
    직원 1명의 후속 정보가 아니라 조건/목록 검색으로 봐야 하는 질문인지 판단한다.
    이런 질문에는 이전 employee_id/name을 붙이면 검색 범위가 잘못 좁아진다.
    """

    if not question:
        return False

    compact_question = compact_text(question)

    keywords = [
        "계약직",
        "정규직",
        "입사한",
        "입사자",
        "퇴사한",
        "퇴직한",
        "퇴사자",
        "퇴직자",
        "성과",
        "성과점수",
        "평가",
        "고과",
        "직원",
        "직원들",
        "사원",
        "사원들",
        "사람",
        "사람들",
        "몇명",
        "인원",
        "인원수",
        "명수",
        "목록",
        "리스트",
        "종류",
        "찾아줘",
        "검색",
        "조회",
    ]

    return any(keyword in compact_question for keyword in keywords)


def remove_followup_reference(question: str) -> str:
    """
    후속 질문의 대명사 표현을 제거한다.
    """

    cleaned = question

    reference_words = [
        "그 사람",
        "그사람",
        "그분",
        "그 직원",
        "그직원",
        "해당 직원",
        "해당직원",
        "방금 사람",
        "방금사람",
        "방금 직원",
        "방금직원",
        "아까 사람",
        "아까사람",
        "아까 직원",
        "아까직원",
        "그 팀",
        "그팀",
        "해당 팀",
        "해당팀",
        "그 부서",
        "그부서",
        "해당 부서",
        "해당부서",
        "거기",
    ]

    for word in reference_words:
        cleaned = cleaned.replace(word, " ")

    return " ".join(cleaned.split())


def should_apply_memory_with_llm(question: str, requester_employee_id: str) -> str:
    """
    현재 질문이 이전 대화를 이어받아야 하는지 LLM으로 먼저 판단한다.

    반환값:
    - "employee": 이전 직원 대상을 이어받음
    - "org": 이전 조직 대상을 이어받음
    - "none": 메모리 미적용
    """

    memory = conversation_memory.get(requester_employee_id, {})
    if not memory:
        return "none"

    prompt = f"""
    다음 HR 질문이 이전 대화 기억을 이어받아야 하는지 판단해라.
    반드시 employee, org, none 중 하나만 출력해라.
    추가 설명은 쓰지 마라.

    현재 질문: {question}
    이전 기억 요약: {json.dumps({
        "has_employee": bool(memory.get("last_employee_name") or memory.get("last_employee_id")),
        "has_org": bool(memory.get("last_team") or memory.get("last_department") or memory.get("last_position")),
    }, ensure_ascii=False)}

    판단 기준:
    - employee: 이전 직원 대상이 있고, 현재 질문이 그 직원을 이어서 묻는 질문이다.
    - org: 이전 부서/팀/직책 대상이 있고, 현재 질문이 그 조직을 이어서 묻는 질문이다.
    - none: 새 질문이거나 대상이 불분명하다.

    중요 규칙:
    - 질문이 그 자체로 독립적인 조건/목록 검색이면 반드시 none이다.
    - "평가 정보가 있는 직원 찾아줘", "계약직 직원 찾아줘", "입사일 있는 직원 찾아줘"는 이전 조직을 이어받지 않는 새 질문이다.
    - "그중 평가 정보가 있는 직원 찾아줘", "해당 부서에서 평가 있는 직원 찾아줘", "거기 계약직도 알려줘"처럼 이전 대상을 가리키는 표현이 있을 때만 org다.
    - 단순히 HR 필드명이나 "직원"이라는 단어가 있다는 이유만으로 org 또는 employee를 선택하지 마라.

    예시:
    - 현재 질문: 평가 정보가 있는 직원 찾아줘 -> none
    - 현재 질문: 그중 평가 정보가 있는 직원 찾아줘 -> org
    - 현재 질문: 계약직 직원 찾아줘 -> none
    - 현재 질문: 거기 계약직도 알려줘 -> org
    - 현재 질문: 이메일 알려줘 -> employee
    """.strip()

    try:
        result = call_llm_completion(
            prompt=prompt,
            temperature=0.0,
            max_tokens=4,
            timeout=10,
        ).strip().lower()
    except Exception:
        return "none"

    if result.startswith("employee"):
        return "employee"
    if result.startswith("org"):
        return "org"

    return "none"


def resolve_question_with_memory(question: str, requester_employee_id: str) -> str:
    """
    이전 조회 대상을 사용해 후속 질문을 보정한다.

    보안상 검색 결과나 민감정보는 사용하지 않고, 대상 식별자만 사용한다.
    """

    # 질문이 비어 있으면 보정할 수 없으므로 그대로 반환한다.
    if not question:
        return question

    # "내 연봉", "내 부서"처럼 본인 조회는 이전 대화 대상과 섞이면 안 되므로 그대로 둔다.
    if is_self_question(question):
        return question

    # 질문 안에 사람/조직 대상이 이미 명시되어 있으면 세션 메모리를 붙이지 않는다.
    # 예: "엄성민 급여는?"에 이전 대상 EMP0070을 덧붙이면 오답이 된다.
    # 질문에 이미 대상이 있으면 메모리를 붙이지 않는다.
    if has_explicit_target(question):
        return question

    memory = conversation_memory.get(requester_employee_id, {})
    employee_target = (
        memory.get("last_employee_id")
        or memory.get("last_employee_name")
    )

    # 필드명만 있는 짧은 후속 질문은 LLM 판단보다 이전 직원 대상을 우선한다.
    # 명시적인 본인 표현은 위에서 이미 제외했으므로 requester로 바뀌지 않는다.
    if employee_target and is_field_only_followup_question(question):
        return f"{employee_target} {remove_followup_reference(question)}"

    memory_scope = should_apply_memory_with_llm(question, requester_employee_id)

    if memory_scope == "none" and not is_followup_question(question):
        return question



    # conversation_memory에서 requester_employee_id로 기억된 대상을 가져온다.
    memory = conversation_memory.get(requester_employee_id, {})
    # cleaned_question = question에서 후속 질문 표현 제거 ex) "그 사람 이메일도 알려줘" -> "이메일도 알려줘"
    cleaned_question = remove_followup_reference(question)

    # 이전에 특정 직원을 조회했다면 그 직원을 이번 질문의 대상으로 붙인다.
    # 예: 이전 "김민수 부서 알려줘" 이후 "이메일도 알려줘" -> "김민수 이메일도 알려줘"
    # 사번이 있으면 이름보다 사번을 우선한다. 동명이인 문제를 줄이기 위한 기준이다.
    employee_target = (
        memory.get("last_employee_id")
        or memory.get("last_employee_name")
    )

    # 이전에 특정 조직을 조회했다면 그 조직을 이번 질문의 대상으로 붙인다.
    # 예: 이전 "채용팀 직원 알려줘" 이후 "계약직도 알려줘" -> "채용팀 계약직도 알려줘"
    org_target = (
        memory.get("last_team")
        or memory.get("last_department")
        or memory.get("last_position")
    )

    # 조건/목록 검색 질문에는 직원 1명 기억을 붙이지 않는다.
    # 조직 기억이 있을 때만 조직 범위의 후속 조건 검색으로 해석한다.
    if is_condition_or_collection_question(question):
        if org_target and (
            memory_scope == "org"
            or is_followup_question(question)
        ):
            return f"{org_target} {cleaned_question}"

        return question

    if memory_scope == "employee" and employee_target:
        return f"{employee_target} {cleaned_question}"

    if memory_scope == "org" and org_target:
        return f"{org_target} {cleaned_question}"

    if employee_target:
        return f"{employee_target} {cleaned_question}"

    if org_target:
        return f"{org_target} {cleaned_question}"

    # 사용할 memory가 없으면 질문을 그대로 반환한다.
    return question



def update_memory_from_tasks(
    requester_employee_id: str,
    tasks: list[dict],
) -> None:
    """
    분석된 task에서 다음 질문 해석에 필요한 대상 정보만 저장한다.

    중요:
    - 특정 직원 조회는 직원만 기억한다.
    - 직원 조회에서 나온 department/team/position은 세션 조직 기억으로 저장하지 않는다.
    - 조직 목록/조직 직원 조회일 때만 department/team/position을 기억한다.
    """

    if not requester_employee_id or not isinstance(tasks, list):
        return

    memory = conversation_memory.setdefault(requester_employee_id, {})

    for task in tasks:
        if not isinstance(task, dict):
            continue

        intent = task.get("intent")

        if task.get("is_self"):
            memory["last_employee_id"] = requester_employee_id
            memory.pop("last_employee_name", None)
            memory.pop("last_department", None)
            memory.pop("last_team", None)
            memory.pop("last_position", None)
            continue

        if task.get("employee_id"):
            memory["last_employee_id"] = str(task["employee_id"]).strip().upper()
            memory.pop("last_employee_name", None)
            memory.pop("last_department", None)
            memory.pop("last_team", None)
            memory.pop("last_position", None)
            continue

        if (
            task.get("employee_name")
            and task.get("employee_name") not in NOT_TARGET_NAMES
            and employee_name_exists(task["employee_name"])
        ):
            memory["last_employee_name"] = task["employee_name"]
            memory.pop("last_employee_id", None)
            memory.pop("last_department", None)
            memory.pop("last_team", None)
            memory.pop("last_position", None)
            continue

        if intent not in {"employee_list", "condition_search", "category_list"}:
            continue

        if task.get("department"):
            memory["last_department"] = task["department"]
            memory.pop("last_employee_name", None)
            memory.pop("last_employee_id", None)

        if task.get("team"):
            memory["last_team"] = task["team"]
            memory.pop("last_employee_name", None)
            memory.pop("last_employee_id", None)

        if task.get("position"):
            memory["last_position"] = task["position"]
            memory.pop("last_employee_name", None)
            memory.pop("last_employee_id", None)
