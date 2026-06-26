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
    "담당업무",
    "이전담당업무",
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
    "담당업무",
    "이전담당업무",
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
    이름 후보는 실제 직원 명부에 존재하는 경우에만 명시 대상으로 인정한다.
    """

    if not question:
        return False


    compact_question = compact_text(question)

    if extract_employee_id(question):
        return True

    employee_name = extract_employee_name(question)

    # extract_employee_name은 "밖에없어" 같은 한글 토막도 잘못 잡을 수 있으므로,
    # 실제 직원 명부에 있는 이름인 경우에만 명시 대상으로 본다.
    if (
        employee_name
        and employee_name not in NOT_TARGET_NAMES
        and employee_name_exists(employee_name)
    ):
        return True

    #compact_question = 공백 제거된 질문 / DEPARTMENTS, TEAMS, JOB_GRADES, POSITIONS에 있는 값이 compact_question에 하나라도 있으면 True, 없으면 False
    for value in DEPARTMENTS + TEAMS + JOB_GRADES + POSITIONS:
        if value in compact_question:
            return True

    return False


def is_followup_question(question: str) -> bool:
    """
    이전 대화의 후속 질문인지 LLM이 판단하지 못한 극단적 경우만 잡는 보조 함수다.
    실제 후속질문 판단은 should_apply_memory_with_llm(LLM 기반)에서 한다.
    """
    # 의도적으로 항상 False를 반환한다.
    # 후속질문 판단은 should_apply_memory_with_llm에 일원화한다.
    return False


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
        "담당업무",
        "이전담당업무",
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
        # 사람/조직 지시 대명사만 제거. 이들은 메모리의 target이 대체한다.
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
        # 사물 지시 대명사("그거","이거" 등)는 일부러 보존한다.
        # 분석기가 "EMP0003 변경이력 그거밖에없어?" 형태로 의미를 이어가도록 한다.
    ]

    for word in reference_words:
        cleaned = cleaned.replace(word, " ")

    return " ".join(cleaned.split())


def compose_followup_question_with_llm(
    question: str,
    requester_employee_id: str,
) -> str | None:
    """
    이전 대화 기억을 이용해 현재 질문이 후속 질문이면 자연스럽게 보강된 질문을 LLM이 생성한다.

    반환값:
    - 보강된 질문 문자열: 메모리를 적용해 완성된 질문
    - None: 메모리를 적용할 필요가 없거나 LLM이 거부했을 때 (원본 질문 사용)
    """

    memory = conversation_memory.get(requester_employee_id, {})
    print(f"[DEBUG] memory state for {requester_employee_id}: {memory}")
    if not memory:
        print("[DEBUG] memory empty -> compose=NONE")
        return None

    last_question = memory.get("last_question") or "(없음)"
    employee_target = (
        memory.get("last_employee_id")
        or memory.get("last_employee_name")
        or ""
    )
    org_target = (
        memory.get("last_team")
        or memory.get("last_department")
        or memory.get("last_position")
        or memory.get("last_job_grade")
        or ""
    )
    topic_labels = ", ".join(memory.get("last_topic_labels") or []) or "(없음)"

    target_lines = []
    if employee_target:
        target_lines.append(f"이전 직원 대상: {employee_target}")
    if org_target:
        target_lines.append(f"이전 조직 대상: {org_target}")
    if not target_lines:
        target_lines.append("이전 대상: (없음)")
    target_text = "\n    ".join(target_lines)

    prompt = f"""
    HR 챗봇 후속질문 처리기다.
    현재 질문이 이전 대화의 후속 질문이면, 이전 대상·토픽을 보강해서 분석기가 이해할 수 있는 완전한 한국어 질문 한 줄을 출력해라.
    후속 질문이 아니면 정확히 NONE 한 단어만 출력해라.
    설명, 인용, 마크다운, 따옴표는 금지한다.

    이전 질문: {last_question}
    {target_text}
    이전 토픽 라벨: {topic_labels}
    현재 질문: {question}

    아주 중요한 원칙:
    - 확실하지 않으면 무조건 NONE을 출력해라. NONE이 안전한 기본값이다.
    - 이전 대화와 완전히 다른 새 주제(새 이름, 새 직무, 완전히 다른 조건)가 나오면 NONE.
    - 현재 질문이 완성된 독립 문장이고 이전 문맥 없이도 의미가 통하면 NONE.
    - 절대로 원본에 없는 사번/이름/부서/팀/직무를 추가하지 마라. 보강은 이전 대화에서 명시된 대상을 그대로 끼워넣는 것만 허용된다.

    후속으로 봐도 되는 경우(아래 패턴에 해당할 때만):
    - 지시 대명사·되묻기: "그거", "이거", "그게 다야?", "더 없어?", "정말?"
    - 한정 조사로 이전 답변을 좁히는 표현: "X도", "X만", "X밖에", "X뿐"
    - 명백히 같은 토픽을 짧게 이어가는 단편: "이메일은?", "그러면 직급은?"
    - 이전 직급/부서/팀만 다른 직급/부서/팀으로 교체해 같은 조건을 다시 물어보는 경우: "과장은?", "차장중엔?", "인사부는?"
    - 이전 조건의 숫자값만 바꿔 다시 묻는 경우: "5천 이상은?", "3천 미만은?"
    - 이전 결과에 새 조건을 추가로 좁히는 경우: "그 중에 여자만", "거기서 계약직만", "그 사람들 중 인사부만"
    - 이전 직원/조직 대상의 다른 필드를 묻는 단편: "거기 이메일은?", "그분 부서는?"

    보강 규칙(후속일 때만 적용):
    - 이전 직원/조직 대상을 사용해 누구에 대한 질문인지 명시한다.
    - 이전 조건(직급·부서·연봉·계약형태 등)을 그대로 유지하되, 현재 질문에서 바뀐 부분만 교체/추가한다.
    - 현재 질문의 어휘는 그대로 보존한다.
    - 답변이나 새로운 정보를 만들지 마라. 질문만 다시 쓴다.

    출력 예시(후속 → 보강):
    - 이전 질문: 내 변경이력 알려줘 / 현재 질문: 그거밖에없어? → EMP0003의 변경이력이 그거밖에 없어?
    - 이전 질문: 김민수 부서 알려줘 / 현재 질문: 이메일은? → 김민수 이메일 알려줘
    - 이전 질문: 김민수 부서 알려줘 / 현재 질문: 직급도 알려줘 → 김민수 직급도 알려줘
    - 이전 질문: 채용팀 직원 알려줘 / 현재 질문: 계약직도 알려줘 → 채용팀 계약직 직원 알려줘
    - 이전 질문: 마케팅부 직원 알려줘 / 현재 질문: 거기 팀장 누구야? → 마케팅부 팀장 누구야?
    - 이전 질문: 대리 중에 연봉 4천 넘는 분 있어? / 이전 조직 대상: 대리 / 현재 질문: 과장중엔? → 과장 중에 연봉 4천 넘는 분 있어?
    - 이전 질문: 대리 중에 연봉 4천 넘는 분 있어? / 이전 조직 대상: 대리 / 현재 질문: 차장은? → 차장 중에 연봉 4천 넘는 분 있어?
    - 이전 질문: 대리 중에 연봉 4천 넘는 분 있어? / 이전 조직 대상: 대리 / 현재 질문: 5천 이상은? → 대리 중에 연봉 5천 이상인 분 있어?
    - 이전 질문: 대리 중에 연봉 4천 넘는 분 있어? / 이전 조직 대상: 대리 / 현재 질문: 여자만 알려줘 → 대리 중에 연봉 4천 넘는 여자 알려줘
    - 이전 질문: 대리 중에 연봉 4천 넘는 분 있어? / 이전 조직 대상: 대리 / 현재 질문: 계약직은? → 계약직 대리 중에 연봉 4천 넘는 분 있어?
    - 이전 질문: 대리 중에 연봉 4천 넘는 분 있어? / 이전 조직 대상: 대리 / 현재 질문: 인사부에선? → 인사부 대리 중에 연봉 4천 넘는 분 있어?
    - 이전 질문: 인사부 직원 중에 계약직 있어? / 현재 질문: 영업부는? → 영업부 직원 중에 계약직 있어?
    - 이전 질문: 가천대학교 나온사람 중에 대학원졸 있어? / 현재 질문: 연세대는? → 연세대 나온사람 중에 대학원졸 있어?

    출력 예시(독립 새 질문 → NONE):
    - 이전 질문: 잔업시간 제일 많은사람 알려줘 / 현재 질문: 프로젝트관리 하는사람 중에 방씨인 사람있지않아? → NONE
    - 이전 질문: 김민수 부서 알려줘 / 현재 질문: 2024년 입사자 알려줘 → NONE
    - 이전 질문: 내 변경이력 알려줘 / 현재 질문: 인턴전환한 사람알려줘 → NONE
    - 이전 질문: 내 변경이력 알려줘 / 현재 질문: 마케팅부 직원 알려줘 → NONE
    - 이전 질문: 엄성민 연봉 알려줘 / 현재 질문: 박서준 부서는? → NONE
    - 이전 질문: 채용팀 직원 알려줘 / 현재 질문: 평가 정보 있는 직원 찾아줘 → NONE

    위 원칙과 예시를 바탕으로 출력해라. 확실하지 않으면 NONE.
    """.strip()

    try:
        result = call_llm_completion(
            prompt=prompt,
            temperature=0.0,
            max_tokens=80,
            timeout=10,
        ).strip()
    except Exception as e:
        print("[DEBUG] followup compose LLM error:", e)
        return None

    print(f"[DEBUG] followup compose raw='{result}' last_q='{last_question}' current='{question}'")

    # 빈 응답·NONE·따옴표 등 정제
    cleaned_result = result.strip().strip('"').strip("'")
    if not cleaned_result or cleaned_result.upper() == "NONE":
        return None

    # 한 줄만 채택해서 부가 텍스트가 묻어와도 안전하게 처리
    first_line = cleaned_result.splitlines()[0].strip()
    if not first_line or first_line.upper() == "NONE":
        return None

    # 자체검증: LLM이 hallucination으로 원본에 없던 새 대상을 끼워넣었으면 reject.
    # 허용되는 추가는 메모리의 이전 대상(employee_target, org_target)뿐이다.
    if not _is_safe_composition(
        composed=first_line,
        original_question=question,
        allowed_targets=[employee_target, org_target],
    ):
        print(f"[DEBUG] followup compose rejected (hallucinated entity) -> use original")
        return None

    return first_line


def _is_safe_composition(
    composed: str,
    original_question: str,
    allowed_targets: list[str],
) -> bool:
    """
    LLM이 보강한 문장이 안전한지 검증한다.

    원칙:
    - 보강된 문장의 '대상'(사번/이름/부서/팀)은 원본 질문에 이미 있었거나 메모리의 이전 대상이어야 한다.
    - 메모리에 없는 새 사번/이름/부서/팀이 추가됐으면 hallucination으로 보고 reject.
    """

    allowed_set = {str(t).strip() for t in allowed_targets if t}

    # 보강된 문장 안의 사번 검사
    composed_emp_ids = set(re.findall(r"EMP\d{4,}", composed.upper()))
    original_emp_ids = set(re.findall(r"EMP\d{4,}", original_question.upper()))

    for emp_id in composed_emp_ids:
        if emp_id in original_emp_ids:
            continue
        if emp_id in allowed_set:
            continue
        return False

    # 부서/팀 명시 검사
    for value in DEPARTMENTS + TEAMS:
        if value in composed and value not in original_question and value not in allowed_set:
            return False

    # 직원 이름은 메모리의 last_employee_name과 비교한다(원본에 없고 메모리에도 없는 이름은 위험).
    # 단, 일반 단어와 충돌할 수 있으므로 한글 2~4글자 + 직원 명부 존재 검증을 같이 한다.
    composed_korean_chunks = set(re.findall(r"[가-힣]{2,4}", composed))
    original_korean_chunks = set(re.findall(r"[가-힣]{2,4}", original_question))
    new_chunks = composed_korean_chunks - original_korean_chunks

    for chunk in new_chunks:
        # 메모리 허용 대상에 포함되면 OK
        if chunk in allowed_set:
            continue
        # 진짜 직원 명부에 있는 이름이 새로 끼어든 경우만 위험으로 본다.
        # (조사·일반 단어가 우연히 매칭되는 false positive 방지)
        if employee_name_exists(chunk):
            return False

    return True


def resolve_question_with_memory(question: str, requester_employee_id: str) -> str:
    """
    이전 조회 대상을 사용해 후속 질문을 보정한다.
    후속 판단과 자연어 보강은 LLM이 한 번에 처리한다.

    보안상 검색 결과나 민감정보는 사용하지 않고, 대상 식별자만 사용한다.
    """

    if not question:
        return question

    # 본인 명시 질문은 그대로 둔다. LLM 호출도 생략한다.
    if is_self_question(question):
        return question

    # 이미 명시 대상이 있으면 메모리를 섞지 않는다.
    if has_explicit_target(question):
        return question

    memory = conversation_memory.get(requester_employee_id, {})
    if not memory:
        return question

    # 후속 판단 + 자연어 보강을 LLM에게 일임한다.
    composed = compose_followup_question_with_llm(question, requester_employee_id)
    if composed:
        return composed

    return question



def update_memory_from_tasks(
    requester_employee_id: str,
    tasks: list[dict],
    last_question: str | None = None,
) -> None:
    """
    분석된 task에서 다음 질문 해석에 필요한 대상 정보만 저장한다.

    중요:
    - 특정 직원 조회는 직원만 기억한다.
    - 직원 조회에서 나온 department/team/position은 세션 조직 기억으로 저장하지 않는다.
    - 조직 목록/조직 직원 조회일 때만 department/team/position을 기억한다.
    - last_question / last_topic_labels는 다음 후속질문 보강에 사용된다.
    """

    if not requester_employee_id or not isinstance(tasks, list):
        return

    memory = conversation_memory.setdefault(requester_employee_id, {})

    # 다음 후속질문 판단에 쓸 수 있도록 직전 질문 원문을 저장한다.
    if last_question:
        memory["last_question"] = last_question

    # 직전 토픽(target_fields의 한글 라벨)을 저장한다.
    # 후속질문에서 토픽이 사라지더라도 라벨을 다시 붙여 분석기가 흐름을 이어가게 한다.
    from common.hr_fields import FIELD_RULES

    topic_labels: list[str] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        for field in task.get("target_fields", []) or []:
            rule = FIELD_RULES.get(field)
            if not rule:
                continue
            label = rule.get("label")
            if label and label not in topic_labels:
                topic_labels.append(label)

    if topic_labels:
        memory["last_topic_labels"] = topic_labels

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

        if task.get("job_grade"):
            memory["last_job_grade"] = task["job_grade"]
            memory.pop("last_employee_name", None)
            memory.pop("last_employee_id", None)
