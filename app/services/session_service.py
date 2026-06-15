from app.services.org_policy_service import DEPARTMENTS, POSITIONS, TEAMS
from app.services.question_service import (
    compact_text,
    extract_employee_id,
    extract_employee_name,
    is_self_question,
)


conversation_memory = {}


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
    "이메일",
    "전화번호",
    "주소",
    "사번",
    "사원번호",
}


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

    #compact_question = 공백 제거된 질문 / DEPARTMENTS, TEAMS, POSITIONS에 있는 값이 compact_question에 하나라도 있으면 True, 없으면 False
    for value in DEPARTMENTS + TEAMS + POSITIONS:
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
        "TOEIC",
        "토익",
        "포상",
        "징계",
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

    # 질문 안에 직원/사번/부서/팀/직책 같은 조회 대상이 이미 있으면 memory를 붙이지 않는다.
    # 예: "김민수 이메일 알려줘", "채용팀 계약직 알려줘"
    if has_explicit_target(question):
        return question

    # 후속 질문도 아니고 HR 조회 항목도 없으면 이전 대상을 붙일 근거가 없다.
    # 예: "고마워", "다시 설명해줘" 같은 일반 문장
    if not is_followup_question(question) and not has_memory_applicable_keyword(question):
        return question

    # conversation_memory에서 requester_employee_id로 기억된 대상을 가져온다.
    memory = conversation_memory.get(requester_employee_id, {})
    # cleaned_question = question에서 후속 질문 표현 제거 ex) "그 사람 이메일도 알려줘" -> "이메일도 알려줘"
    cleaned_question = remove_followup_reference(question)

    # 이전에 특정 직원을 조회했다면 그 직원을 이번 질문의 대상으로 붙인다.
    # 예: 이전 "김민수 부서 알려줘" 이후 "이메일도 알려줘" -> "김민수 이메일도 알려줘"
    employee_target = (
        memory.get("last_employee_name")
        or memory.get("last_employee_id")
    )

    if employee_target:
        return f"{employee_target} {cleaned_question}"

    # 이전에 특정 조직을 조회했다면 그 조직을 이번 질문의 대상으로 붙인다.
    # 예: 이전 "채용팀 직원 알려줘" 이후 "계약직도 알려줘" -> "채용팀 계약직도 알려줘"
    org_target = (
        memory.get("last_team")
        or memory.get("last_department")
        or memory.get("last_position")
    )

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
    """

    if not requester_employee_id or not isinstance(tasks, list):
        return

    memory = conversation_memory.setdefault(requester_employee_id, {})

    for task in tasks:
        if not isinstance(task, dict):
            continue

        if task.get("is_self"):
            memory["last_employee_id"] = requester_employee_id
            memory.pop("last_employee_name", None)
            memory.pop("last_department", None)
            memory.pop("last_team", None)
            memory.pop("last_position", None)

        if task.get("employee_id"):
            memory["last_employee_id"] = str(task["employee_id"]).strip().upper()
            memory.pop("last_employee_name", None)
            memory.pop("last_department", None)
            memory.pop("last_team", None)
            memory.pop("last_position", None)

        if task.get("employee_name"):
            memory["last_employee_name"] = task["employee_name"]
            memory.pop("last_employee_id", None)
            memory.pop("last_department", None)
            memory.pop("last_team", None)
            memory.pop("last_position", None)

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
