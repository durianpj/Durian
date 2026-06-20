import time
import json

from fastapi import HTTPException

# 요청자의 권한 레벨을 조회하는 함수
from app.services.hybrid_search_service import get_user_permission_level

# 질문을 LLM으로 분석해서 task 목록으로 나누는 함수들
from app.services.question_analyzer_service import (
    analyze_question_to_tasks,
    normalize_tasks,
)

# 이전 대화 기억을 이용해서 후속 질문을 보강하는 함수들
from app.services.session_service import (
    reset_memory_if_employee_changed,
    resolve_question_with_memory,
    update_memory_from_tasks,
)

# task 하나를 실제로 처리하는 서비스
# 권한 판단, 검색, 답변 생성 등이 여기서 수행된다.
from app.services.task_processor_service import process_task
from app.services.llm_service import get_active_llm_label


def sanitize_sources(sources: list[dict]) -> list[dict]:
    """
    Keep only source identifiers in the API response.
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
    task_permissions = [
        result.get("permission", {})
        for result in task_results
        if result.get("permission")
    ]

    return {
        "allowed": any(permission.get("allowed") for permission in task_permissions),
        "permission_level": permission_level,
        "required_level": max(
            [
                permission.get("required_level", 1)
                for permission in task_permissions
            ]
            or [1]
        ),
    }


def build_public_error(success: bool, permission: dict) -> dict | None:
    if success:
        return None

    if not permission.get("allowed"):
        return {
            "code": "ACCESS_DENIED",
            "message": (
                f"요청하신 정보는 Level {permission.get('required_level', 1)} "
                "이상만 조회할 수 있습니다."
            ),
        }

    return {
        "code": "NO_RESULT",
        "message": "조건에 맞는 조회 결과가 없습니다.",
    }


def build_debug_response(full_response: dict, source_limit: int = 5) -> dict:
    debug_response = full_response.copy()
    sources = full_response.get("sources", [])

    debug_response["sources"] = sources[:source_limit]
    debug_response["source_count"] = len(sources)

    return debug_response


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
    question = question.strip()

    print(f"[QUESTION] employee_id={employee_id} question={question}")

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
    analysis = analyze_question_to_tasks(resolved_question)

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
    tasks = normalize_tasks(analysis, resolved_question)
    print(
        "[DEBUG] normalized tasks:",
        json.dumps(tasks, ensure_ascii=False),
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
    task_results = [
        process_task(
            task=task,
            original_question=resolved_question,
            requester_employee_id=employee_id,
            permission_level=permission_level,
        )
        for task in tasks
    ]

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
    }

    print(
        "[RESPONSE_DEBUG]",
        json.dumps(build_debug_response(full_response), ensure_ascii=False),
    )

    public_permission = summarize_permission(
        task_results=task_results,
        permission_level=permission_level,
    )
    public_success = full_response["success"]
    public_answer = full_response["answer"]

    if not public_success and not public_permission.get("allowed"):
        public_answer = "요청하신 정보는 현재 권한으로 조회할 수 없습니다."

    public_response = {
        "success": public_success,
        "answer": public_answer,
        "permission": public_permission,
        "sources": sanitize_sources(sources),
    }

    public_error = build_public_error(
        success=public_success,
        permission=public_permission,
    )

    if public_error:
        public_response["error"] = public_error

    print("[ANSWER]", public_response["answer"])

    return public_response
