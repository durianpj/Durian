import time
import json

from fastapi import HTTPException

# 요청자의 권한 레벨을 조회하는 함수
from app.services.hybrid_search_service import get_user_permission_level, is_retired_employee

# 질문을 LLM으로 분석해서 task 목록으로 나누는 함수들
from app.services.question_analyzer_service import (
    analyze_question_to_tasks,
    normalize_tasks,
)

# 이전 대화 기억을 이용해서 후속 질문을 보강하는 함수들
from app.services.session_service import (
    extract_employee_id,
    has_memory_applicable_keyword,
    reset_memory_if_employee_changed,
    resolve_question_with_memory,
    update_memory_from_tasks,
)
from app.services.candidate_extractor_service import extract_question_candidates
from app.services.question_service import is_employee_collection_query

# task 하나를 실제로 처리하는 서비스
# 권한 판단, 검색, 답변 생성 등이 여기서 수행된다.
from app.services.task_processor_service import is_supervisor_query, process_task
from app.services.llm_service import get_active_llm_label

def sanitize_sources(sources: list[dict]) -> list[dict]:
    """
    API 응답에서 소스(출처)의 식별자(Index, ID)만 남기고 정리하는 함수
    """

    sanitized = []

    for source in sources:
        sanitized.append(
            {
                "index": source.get("index"),
                "doc_id": source.get("_id") or source.get("doc_id"),
            }
        )

    return sanitized


def summarize_permission(task_results: list[dict], permission_level: int) -> dict:
    """
    여러 태스크 결과들의 권한 상태를 종합하여 최종 권한 정보를 반환하는 함수
    """
    task_permissions = [
        result.get("permission", {})
        for result in task_results
        if result.get("permission")
    ]

    return {
        # 하나라도 허용된 권한이 있다면 True 반환
        "allowed": any(permission.get("allowed") for permission in task_permissions),
        "permission_level": permission_level,
        # 필요한 권한 레벨 중 가장 높은 값을 채택 (기본값 1)
        "required_level": max(
            [
                permission.get("required_level", 1)
                for permission in task_permissions
            ]
            or [1]
        ),
    }


def build_public_error(success: bool, permission: dict) -> dict | None:
    """
    실패 상황(성공 flag가 False)일 때, 권한 부족 또는 결과 없음에 대한 공개용 에러 메시지를 생성하는 함수
    """
    if success:
        return None

    # 권한이 없는 경우 접근 거부 에러 반환
    if not permission.get("allowed"):
        return {
            "code": "ACCESS_DENIED",
            "message": (
                f"요청하신 정보는 Level {permission.get('required_level', 1)} "
                "이상만 조회할 수 있습니다."
            ),
        }

    # 권한은 있으나 데이터가 없는 경우 결과 없음 에러 반환
    return {
        "code": "NO_RESULT",
        "message": "조건에 맞는 조회 결과가 없습니다.",
    }


def build_debug_response(full_response: dict, source_limit: int = 5) -> dict:
    """
    전체 응답에서 답변 텍스트를 제외하고 출처(Source) 개수를 제한하여 디버깅용 응답을 만드는 함수
    """
    debug_response = full_response.copy()
    sources = full_response.get("sources", [])

    debug_response.pop("answer", None)
    debug_response["sources"] = sources[:source_limit]
    debug_response["source_count"] = len(sources)

    return debug_response


# 인사 정보 외의 질문을 했을 때 반환할 기본 안내 메시지
NON_HR_QUESTION_MESSAGE = (
    "저는 인사 정보 조회를 돕는 챗봇입니다. "
    "인사 정보와 관련된 내용을 물어봐 주세요."
)


def is_basic_profile_question(question: str) -> bool:
    """
    사용자의 질문이 '기본 인사 정보'나 '프로필'을 조회하려는 의도인지 키워드로 판단하는 함수
    """
    compact_question = question.replace(" ", "")
    return any(
        keyword in compact_question
        for keyword in [
            "기본정보",
            "인사정보",
            "기본인사정보",
            "인사프로필",
        ]
    )


def is_non_hr_task_result(tasks: list[dict]) -> bool:
    """
    분석된 태스크가 인사 정보 조회와 전혀 관련이 없는 '알 수 없는(unknown)' 요청인지 확인하는 함수
    """
    if len(tasks) != 1:
        return False

    task = tasks[0]

    return (
        task.get("intent") == "unknown"
        and task.get("target_fields") == ["unknown"]
        and not task.get("employee_name")
        and not task.get("employee_id")
        and not task.get("department")
        and not task.get("team")
        and not task.get("job_grade")
        and not task.get("position")
        and not task.get("unknown_orgs")
        and not task.get("filters")
    )


def split_single_lookup_tasks_by_employee_candidates(
    tasks: list[dict],
    employee_names: list[str],
) -> list[dict]:
    """
    동명이인이나 여러 명의 이름 후보가 존재할 때, 하나의 단일 조회 태스크를 각 이름별 독립된 태스크로 쪼개주는 함수
    """
    if len(employee_names) <= 1 or len(tasks) != 1:
        return tasks

    task = tasks[0]

    if not isinstance(task, dict):
        return tasks

    if task.get("intent") != "single_lookup":
        return tasks

    if task.get("employee_id") or task.get("is_self"):
        return tasks

    split_tasks = []
    for employee_name in employee_names:
        cloned_task = task.copy()
        cloned_task["employee_name"] = employee_name
        cloned_task["is_self"] = False
        split_tasks.append(cloned_task)

    return split_tasks


def suppress_hidden_retired_results_for_multi_name_query(
    task_results: list[dict],
    employee_names: list[str],
) -> list[dict]:
    """
    여러 이름을 동시에 조회했을 때, 유효한 결과가 하나라도 존재한다면 권한이 없거나 숨겨진 퇴사자 결과는 필터링하여 감추는 함수
    """
    if len(employee_names) <= 1:
        return task_results

    has_visible_result = any(
        result.get("permission", {}).get("allowed") and result.get("sources")
        for result in task_results
    )

    if not has_visible_result:
        return task_results

    return [
        result
        for result in task_results
        if result.get("permission", {}).get("allowed")
    ]


def handle_rag_chat(question: str, employee_id: str) -> dict:
    """
    /rag-chat 요청의 전체 처리 흐름을 담당하는 함수.

    전체 흐름:
    1. 요청자 사번 정리
    2. 요청자가 바뀐 경우 이전 대화 기억 초기화
    3. 요청자의 권한 레벨 조회
    4. 이전 대화 기억을 이용해 후속 질문 보강
    5. LLM으로 질문을 task 목록으로 분해
    6. task를 코드에서 안전하게 정규화
    7. task별로 권한 판단, 검색, 답변 생성
    8. 대화 기억 업데이트
    9. 최종 응답 형태로 조립
    """

    # 전체 요청 처리 시간을 측정하기 위한 시작 시간
    # request_start_time = time.perf_counter()

    # 사번 앞뒤 공백 제거 후 대문자로 통일
    # 예: emp0070 -> EMP0070
    employee_id = employee_id.strip().upper()
    # 입력된 질문의 앞뒤 공백 제거
    question = question.strip()

    # 디버깅 및 추적을 위해 정제된 요청자 사번과 질문을 로그에 기록
    print(f"[QUESTION] employee_id={employee_id} question={question}")

    # 대화 세션 유지를 위해, 요청자가 변경되었다면 이전 사용자의 대화 기억(메모리)을 초기화
    reset_memory_if_employee_changed(employee_id)

    # =========================
    # 1. 요청자 권한 조회
    # =========================

    # 권한 조회에 걸리는 시간을 측정하기 위한 시작 시간
    # permission_start_time = time.perf_counter()

    # 요청자 사번을 기준으로 permission_level 조회
    # 예: 일반 직원 1, 중간 권한 2, 관리자 3
    permission_level = get_user_permission_level(employee_id)

    # 권한 조회 소요 시간 출력
    # print(
    #     "[TIME] get_user_permission_level:",
    #     f"{time.perf_counter() - permission_start_time:.3f}s",
    # )

    # 요청자 사번이 존재하지 않으면 404 에러 반환
    if permission_level is None:
        raise HTTPException(
            status_code=404,
            detail="요청자 사번을 찾을 수 없습니다.",
        )

    # 권한 레벨이 1, 2, 3 중 하나가 아니면 잘못된 데이터로 판단
    if permission_level not in [1, 2, 3]:
        raise HTTPException(
            status_code=400,
            detail="계산된 permission_level이 유효하지 않습니다.",
        )

    # 요청자가 이미 퇴사한 사원인지 검증
    if is_retired_employee(employee_id):
        # 퇴사자 계정인 경우 처리를 중단하고 거부 응답 반환
        return {
            "success": False,
            "answer": "퇴사 처리된 계정입니다.",
            "permission": {
                "allowed": False,
                "permission_level": permission_level,
                "required_level": 1,
            },
            "sources": [],
            "error": {
                "code": "RETIRED_EMPLOYEE",
                "message": "퇴사 처리된 계정입니다.",
            },
        }

    # =========================
    # 2. 이전 대화 기억으로 질문 보강
    # =========================

    # 후속 질문인 경우 이전 대화에서 기억한 부서/팀/직원 정보를 질문에 보강한다.
    #
    # 예:
    # 이전 질문: "마케팅부 직원 알려줘"
    # 현재 질문: "그중 팀장은?"
    # 보강 후: "마케팅부 직원 중 팀장은?"
    resolved_question = resolve_question_with_memory(
        question=question,
        requester_employee_id=employee_id,
    )

    # 실제 질문이 보강되었는지 디버그용으로 출력
    if resolved_question != question:
        print("[QUESTION] resolved_question:", resolved_question)

    # =========================
    # 3. 질문을 task 목록으로 분석(LLM한테 의도 분석만 맡기고 권한 판단은 절대 맡기지 않는다)
    # =========================

    # LLM 질문 분석 시간 측정 시작
    # analyze_start_time = time.perf_counter()

    # 자연어 질문을 LLM에게 보내서 task 목록으로 분해한다.
    #
    # 예:
    # "김민수의 부서랑 연봉 알려줘"
    # -> [
    #      {"intent": "single_lookup", "target_fields": ["department"], ...},
    #      {"intent": "single_lookup", "target_fields": ["salary"], ...}
    #    ]
    #
    # 주의:
    # 여기서 LLM은 의도 분석만 한다.
    # 권한 판단은 절대 LLM에게 맡기지 않는다.
    # 사용자 질문에서 LLM에게 넘기기 전에 먼저 후보 단어를 추출한다.
    # 예: "마케팅부 직원 알려줘" → ["마케팅부"]
    candidates = extract_question_candidates(resolved_question)

    # 후보 추출 결과를 디버그용으로 출력한다.
    # ensure_ascii=False를 쓰면 한글이 깨지지 않고 그대로 출력된다.
    print(
        "[DEBUG] candidate extraction for LLM hints:",
        json.dumps(candidates, ensure_ascii=False),
    )

    # 질문과 함께 미리 뽑은 후보 단어를 LLM 분석 함수에 전달한다.
    # LLM은 이 candidates를 참고해서 department/team/employee_name 등을 더 정확히 판단한다.
    analysis = analyze_question_to_tasks(resolved_question, candidates=candidates)

    # LLM 분석 소요 시간 출력
    # print(
    #     "[TIME] analyze_question_to_tasks:",
    #     f"{time.perf_counter() - analyze_start_time:.3f}s",
    # )

    # =========================
    # 4. task 정규화
    # =========================

    # 정규화 시간 측정 시작
    # normalize_start_time = time.perf_counter()

    # LLM이 만든 결과를 코드에서 다시 안전하게 정리한다.
    #
    # 이유:
    # LLM이 잘못된 intent나 field를 줄 수 있기 때문에
    # 허용된 intent / 허용된 field만 남기도록 보정한다.
    tasks = normalize_tasks(analysis, resolved_question, candidates=candidates)
    # 보정 및 정규화가 완료된 최종 task 목록을 디버그 로그로 출력
    print(
        "[DEBUG] normalized task after LLM correction:",
        json.dumps(tasks, ensure_ascii=False),
    )

    # 추출된 후보군 중에서 직원 이름 목록을 안전하게 가져옴 (없으면 빈 리스트)
    employee_name_candidates = candidates.get("employee_names") or []

    # 질문 텍스트 내에 명시적인 사번 패턴(예: 사번 정규식 매칭)이 있는지 추출
    explicit_employee_id = extract_employee_id(resolved_question)
    if explicit_employee_id:
        # 질문 안에 사번이 들어 있으면 task에도 다시 심어서 검색 기준을 고정한다.
        for task in tasks:
            # task가 딕셔너리 형태이고 기존에 설정된 사번이 없다면 명시적 사번 주입
            if isinstance(task, dict) and not task.get("employee_id"):
                task["employee_id"] = explicit_employee_id

    # 여러 명의 직원 이름이 포함되어 있는 경우, 단일 조회(single_lookup) task를 직원별로 쪼갬
    tasks = split_single_lookup_tasks_by_employee_candidates(
        tasks=tasks,
        employee_names=employee_name_candidates,
    )

    # 정규화 소요 시간과 task 개수 출력
    # print(
    #     "[TIME] normalize_tasks:",
    #     f"{time.perf_counter() - normalize_start_time:.3f}s",
    #     "tasks:",
    #     len(tasks),
    # )

    # =========================
    # 5. task별 처리
    # =========================

    # 전체 task 처리 시간 측정 시작
    # tasks_start_time = time.perf_counter()

    # task 하나마다 독립적으로 처리한다.
    #
    # process_task 내부에서 하는 일:
    # 1. target_fields 기준으로 권한 판단
    # 2. 허용된 필드만 검색 대상으로 선택
    # 3. intent에 따라 직접 조회 또는 hybrid 검색 실행
    # 4. task별 답변 생성
    
    # 후보 단어(부서명, 직원명 등)가 하나라도 존재하는지 확인
    has_candidates = any(candidates.get(key) for key in candidates)
    
    # 인사(HR) 정보조회와 무관한 성격의 질문이거나, 컨텍스트가 전혀 없는 질문인지 필터링
    if (
        is_non_hr_task_result(tasks)
        or (
            not has_candidates
            and not has_memory_applicable_keyword(resolved_question)
            and not is_employee_collection_query(resolved_question)
            and not is_supervisor_query(resolved_question)
            and not is_basic_profile_question(resolved_question)
        )
    ):
        # HR 관련 질문이 아니라고 판단되면 기본 안내 메시지와 함께 즉시 응답 반환
        return {
            "success": True,
            "answer": NON_HR_QUESTION_MESSAGE,
            "permission": {
                "allowed": True,
                "permission_level": permission_level,
                "required_level": 1,
            },
            "sources": [],
        }

    # 분해된 각 task 리스트를 순회하며 개별 검색 및 답변 생성 알고리즘 실행 (리스트 컴프리헨션)
    task_results = [
        process_task(
            task=task,
            original_question=resolved_question,
            requester_employee_id=employee_id,
            permission_level=permission_level,
        )
        for task in tasks
    ]
    
    # 다중 이름 조회 시, 퇴사자 정보 노출을 제한해야 하는 비즈니스 규칙에 따라 결과 마스킹/숨김 처리
    task_results = suppress_hidden_retired_results_for_multi_name_query(
        task_results=task_results,
        employee_names=employee_name_candidates,
    )

    # =========================
    # 6. 대화 기억 업데이트
    # =========================

    # 이번 질문에서 추출된 부서/팀/직원 정보를 세션 메모리에 저장한다.
    #
    # 예:
    # 사용자가 "마케팅부 직원 알려줘"라고 물으면
    # 이후 "그중 팀장은?" 같은 후속 질문에서 마케팅부를 기억할 수 있다.
    update_memory_from_tasks(
        requester_employee_id=employee_id,
        tasks=tasks,
        last_question=question,
    )

    # 전체 task 처리 소요 시간 출력
    # print(
    #     "[TIME] all process_task:",
    #     f"{time.perf_counter() - tasks_start_time:.3f}s",
    # )

    # =========================
    # 7. 답변 모으기
    # =========================

    # task별 결과에서 answer가 있는 것만 모은다.
    answers = [
        result["answer"]
        for result in task_results
        if result.get("answer")
    ]

    # =========================
    # 8. 출처 모으기
    # =========================

    # task별 sources를 하나의 리스트로 합친다.
    sources = []
    for result in task_results:
        sources.extend(result.get("sources", []))

    # 데이터 정합성 검증을 위해 추출된 출처 데이터 중 상위 5개 미리보기 로그 출력
    print(
        "[DEBUG] source preview for verification:",
        json.dumps(sources[:5], ensure_ascii=False),
    )

    # 전체 /rag-chat 처리 시간 출력
    # print(
    #     "[TIME] rag_chat total:",
    #     f"{time.perf_counter() - request_start_time:.3f}s",
    # )

    # =========================
    # 9. 최종 응답 조립
    # =========================

    full_response = {
        # 하나라도 허용된 task가 있으면 success를 true로 반환한다.
        "success": any(
            result.get("permission", {}).get("allowed")
            for result in task_results
        ),

        # task별 답변을 빈 줄로 구분해서 하나의 답변으로 합친다.
        "answer": "\n\n".join(answers),

        # 요청자의 권한 정보와 task별 권한 판단 결과를 함께 반환한다.
        "permission": {
            "employee_id": employee_id,
            "permission_level": permission_level,
            "tasks": [
                result.get("permission", {})
                for result in task_results
            ],
        },

        # LLM 분석 후 정규화된 task 목록
        "tasks": tasks,

        # 검색 결과 출처 목록
        "sources": sources,

        # 이전 대화 기억으로 보강된 최종 질문
        "resolved_question": resolved_question,

        # 사용한 LLM 모델명
        "model_type": get_active_llm_label(),

        # 검색 결과는 있었지만 퇴사자 정책으로 차단된 경우를 최종 응답까지 전달한다.
        "blocked_reason": next(
            (
                result.get("blocked_reason")
                for result in task_results
                if result.get("blocked_reason")
            ),
            None,
        ),
    }

    # 보안이나 가독성을 위해 본문을 제외한 메타데이터만 디버그 로그로 기록
    print(
        "[DEBUG] response metadata without answer:",
        json.dumps(build_debug_response(full_response), ensure_ascii=False),
    )

    # 외부(클라이언트)용으로 반환할 최종 권한 요약 데이터 생성
    public_permission = summarize_permission(
        task_results=task_results,
        permission_level=permission_level,
    )
    public_success = full_response["success"]
    public_answer = full_response["answer"]

    # 만약 작업 수행에 실패했고 권한도 허용되지 않는 경우, 보안을 위해 에러 문구로 응답을 대체
    blocked_reason = full_response.get("blocked_reason")

    if not public_success and not public_permission.get("allowed"):
        if blocked_reason == "retired_employee":
            public_answer = "퇴사자 정보는 관리자만 조회할 수 있습니다."
        else:
            public_answer = "요청하신 정보는 현재 권한으로 조회할 수 없습니다."

    # 외부 클라이언트에 노출 가능한 정보망으로 구성된 public response 객체 조립
    public_response = {
        "success": public_success,
        "answer": public_answer,
        "permission": public_permission,
        "sources": sanitize_sources(sources), # 민감 정보가 마스킹된 출처 데이터
    }

    # 실패 케이스일 경우 대외용 표준 에러 객체 빌드 및 주입
    public_error = build_public_error(
        success=public_success,
        permission=public_permission,
    )

    if public_error:
        public_response["error"] = public_error

    # 최종적으로 유저에게 나가는 완성형 답변 본문을 로그에 기록
    print("[ANSWER]", public_response["answer"])

    return public_response