import re
import time
import json

from common.filter_utils import (
    add_filter_if_missing,
    get_filter_fields,
    unique_keep_order,
    value_appears_in_question,
)
from common.text_utils import compact_text
from app.services.llm_service import generate_answer
from app.services.question_service import (
    extract_employee_id,
    extract_employee_name,
    extract_name_like_prefix,
    is_employee_collection_query,
    is_self_question,
)
from app.services.hybrid_search_service import (
    build_context,
    count_employees_by_conditions,
    employee_name_exists,
    filter_hits_with_answer_values,
    find_employee_name_in_question,
    get_allowed_field_value,
    get_category_values,
    make_sources,
    search_employees_by_conditions,
    search_employees_by_filter_conditions,
    search_hits_by_employee_ids,
    search_hybrid,
)
from common.hr_fields import (
    ACCESSIBLE_INDICES,
    FIELD_RULES,
    get_max_required_level,
    select_indices_by_fields,
    split_fields_by_permission,
)
from common.hr_master_data import (
    DEPARTMENTS,
    TEAMS,
    JOB_GRADES,
    POSITIONS,
)


COMMON_ERROR_MESSAGE = "요청하신 정보를 확인할 수 없습니다."
PERMISSION_DENIED_MESSAGE = "해당 정보에 접근할 권한이 없습니다."
NO_SEARCH_RESULT_MESSAGE = "조건에 맞는 조회 결과가 없습니다."
MAX_OUTPUT_EMPLOYEES = 50
UNSUPPORTED_SUPERVISOR_MESSAGE = (
    "현재 데이터에는 상사 관계 정보가 없어 상사 조회를 지원할 수 없습니다."
)


def is_supervisor_query(question: str) -> bool:
    """
    현재 인사 데이터에는 manager_id/reports_to 같은 상사 관계 필드가 없다.
    상사 질문은 직급이나 직책으로 추론하지 않고 미지원으로 처리한다.
    """

    compact_question = compact_text(question)

    supervisor_keywords = [
        "상사",
        "상급자",
        "윗사람",
        "보고라인",
        "보고체계",
        "결재라인",
        "직속상관",
        "직속상사",
    ]

    return any(keyword in compact_question for keyword in supervisor_keywords)


def build_denied_message(denied_fields: list[str]) -> str:
    if not denied_fields:
        return ""

    labels = [
        FIELD_RULES.get(field, {}).get("label", field)
        for field in denied_fields
    ]

    return "접근 권한이 없어 제공할 수 없는 정보: " + ", ".join(labels)


def build_category_answer(field_key: str, permission_level: int) -> str:
    """
    부서/팀/직급/직책 목록 질문에 대한 답변을 만든다.
    """

    values = get_category_values(field_key, permission_level)
    label = FIELD_RULES.get(field_key, {}).get("label", field_key)

    if not values:
        return f"조회 가능한 {label} 목록이 없습니다."

    return (
        f"{label} 목록은 다음과 같습니다.\n"
        + "\n".join([f"- {value}" for value in values])
    )

def build_employee_count_answer(task: dict, count: int) -> str:
    """
    직원 수 조회 결과를 사용자 답변 문장으로 만든다.
    """

    conditions = []

    if task.get("department"):
        conditions.append(task["department"])

    if task.get("team"):
        conditions.append(task["team"])

    if task.get("job_grade"):
        conditions.append(task["job_grade"])

    if task.get("position"):
        conditions.append(task["position"])

    if conditions:
        condition_text = " ".join(conditions)
        return f"{condition_text} 직원 수는 {count}명입니다."

    return f"직원 수는 {count}명입니다."

def normalize_task_org_values(task: dict) -> dict:
    """
    task 안의 department/team/position 값이 실제 조직 목록에 있는지 검증한다.
    잘못된 값은 검색 조건에서 제거한다.
    """

    normalized_task = task.copy()

    raw_department = normalized_task.get("department")
    raw_team = normalized_task.get("team")
    raw_job_grade = normalized_task.get("job_grade")
    raw_position = normalized_task.get("position")

    normalized_task["department"] = (
        # raw_department이 조직 정책의 부서 목록에 있으면 그대로 쓰고, 없으면 None으로 바꾼다.
        raw_department
        if raw_department in DEPARTMENTS
        else None
    )

    normalized_task["team"] = (
        raw_team
        if raw_team in TEAMS
        else None
    )

    normalized_task["job_grade"] = (
        raw_job_grade
        if raw_job_grade in JOB_GRADES
        else None
    )

    normalized_task["position"] = (
        raw_position
        if raw_position in POSITIONS
        else None
    )

    return normalized_task


def sanitize_filters(filters: list[dict]) -> list[dict]:
    """
    LLM이 만든 filters를 실행 가능한 안전한 조건만 남기도록 정리한다.

    중요:
    - FIELD_RULES에 없는 field는 버린다.
    - department/team/job_grade/position은 실제 정책 목록에 있는 값만 허용한다.
    - 알 수 없는 op는 eq로 낮춘다.
    """

    if not isinstance(filters, list):
        return []

    allowed_ops = {"eq", "contains", "gt", "gte", "lt", "lte", "between", "exists"}
    safe_filters = []

    for item in filters:
        if not isinstance(item, dict):
            continue

        field = item.get("field")
        op = item.get("op", "eq")
        value = item.get("value")

        if field not in FIELD_RULES:
            continue

        if value is None or value == "":
            continue

        if isinstance(value, str) and value.strip().upper() == "NULL":
            continue

        if op not in allowed_ops:
            op = "eq"

        if field == "employee_name" and value in DEPARTMENTS + TEAMS + JOB_GRADES + POSITIONS:
            continue

        if field == "department" and value not in DEPARTMENTS:
            continue

        if field == "team" and value not in TEAMS:
            continue

        if field == "job_grade" and value not in JOB_GRADES:
            continue

        if field == "position" and value not in POSITIONS:
            continue

        safe_filters.append(
            {
                "field": field,
                "op": op,
                "value": value,
            }
        )

    return safe_filters


def remove_hallucinated_org_values(task: dict, original_question: str) -> dict:
    """
    특정 직원 조회에서 LLM이 만들어낸 department/team/position 값을 제거한다.

    예:
    질문: 오민호 부서 알려줘
    잘못된 task: department=영업부
    -> 영업부는 질문 원문에 없으므로 제거
    """

    normalized_task = task.copy()

    has_employee_target = bool(
        normalized_task.get("employee_name")
        or normalized_task.get("employee_id")
    )

    if not has_employee_target:
        return normalized_task

    for key in ["department", "team", "job_grade", "position"]:
        value = normalized_task.get(key)

        if value and not value_appears_in_question(original_question, value):
            normalized_task[key] = None

    return normalized_task


def remove_hallucinated_org_filters(
    filters: list[dict],
    original_question: str,
    task: dict,
) -> list[dict]:
    """
    특정 직원 조회에서 질문 원문에 없던 조직 filter를 제거한다.
    """

    has_employee_target = bool(
        task.get("employee_name")
        or task.get("employee_id")
    )

    if not has_employee_target:
        return filters

    safe_filters = []

    for item in filters:
        field = item.get("field")
        value = item.get("value")

        if field == "job_grade":
            continue

        if field in {"department", "team", "job_grade", "position"}:
            if not value_appears_in_question(original_question, value):
                continue

        safe_filters.append(item)

    return safe_filters

def remove_ambiguous_previous_task_filter(
    filters: list[dict],
    original_question: str,
    task: dict,
) -> list[dict]:
    """
    '개발 관련 업무담당자' 같은 질문에서
    LLM이 previous_task 조건을 잘못 만든 경우 제거한다.

    previous_task는 '이전담당업무'다.
    사용자가 명확히 이전 업무/경력/이전담당업무를 물어본 게 아니면
    부서/팀 담당자 조회로 처리한다.
    """

    if not filters:
        return filters

    compact_question = compact_text(original_question)

    explicit_previous_task_keywords = [
        "이전담당업무",
        "이전업무",
        "이전직무",
        "이전경력",
        "경력",
        "전직장",
        "전에하던업무",
    ]

    # 사용자가 진짜 previous_task를 물어본 경우는 제거하면 안 된다.
    if any(keyword in compact_question for keyword in explicit_previous_task_keywords):
        return filters

    # 부서/팀이 잡힌 상태에서 previous_task가 끼어든 경우만 제거한다.
    if not (task.get("department") or task.get("team")):
        return filters

    return [
        item
        for item in filters
        if item.get("field") != "previous_task"
    ]


# 놓친 조건이 filters에 빠져 있으면 추가하는 함수
# filters의 예시 : filters = [{"field": "department", "op": "eq", "value": "마케팅부"},
#                            {"field": "position", "op": "eq", "value": "대리"}]
def build_task_question(original_question: str, task: dict) -> str:
    """
    hybrid 검색에 사용할 질문을 만든다.

    원문 질문만으로 검색 키워드가 부족할 수 있으므로, task에서 추출한 이름/조직/필드명을
    함께 붙여 BM25와 벡터 검색이 같은 의도를 더 잘 보도록 한다.
    """

    parts = [original_question]
    # 원본질문에 LLM이 추출한 이름, 사번, 부서, 팀, 포지션이 포함되어있지 않으면 parts에 추가
    identity_keys = ["employee_id"]

    if task.get("intent") == "single_lookup":
        identity_keys.insert(0, "employee_name")

    for key in identity_keys + ["department", "team", "position"]:
        value = task.get(key)

        if value and str(value) not in " ".join(parts):
            parts.append(str(value))

    # 원본질문에 LLM이 추출한 filters의 각 value값들이 포함되어있지 않으면 parts에 추가
    filters = task.get("filters", [])

    if isinstance(filters, list):
        for item in filters:
            if not isinstance(item, dict):
                continue

            value = item.get("value")

            if value and str(value) not in " ".join(parts):
                parts.append(str(value))


    # 원본질문에 LLM이 추출한 field의 라벨값들이 포함되어있지 않으면 parts에 추가
    for field in task.get("target_fields", []):
        label = FIELD_RULES.get(field, {}).get("label")

        if label and label not in " ".join(parts):
            parts.append(label)

    return " ".join(parts)


def limit_hits_by_employee_count(
    hits: list[dict],
    limit: int = MAX_OUTPUT_EMPLOYEES,
) -> list[dict]:
    """
    검색/검증은 전체로 하되, 최종 출력은 employee_id 기준 최대 limit명으로 제한한다.
    같은 직원의 여러 문서는 함께 유지한다.
    """

    if not hits or limit <= 0:
        return []

    selected_employee_ids = []
    limited_hits = []
    fallback_count = 0

    for hit in hits:
        employee_id = hit.get("_source", {}).get("employee_id")

        if not employee_id:
            if fallback_count < limit:
                limited_hits.append(hit)
                fallback_count += 1
            continue

        if employee_id not in selected_employee_ids:
            if len(selected_employee_ids) >= limit:
                continue

            selected_employee_ids.append(employee_id)

        if employee_id in selected_employee_ids:
            limited_hits.append(hit)

    return limited_hits


def format_allowed_hits_answer(
    hits: list[dict],
    allowed_fields: list[str],
    limit: int = MAX_OUTPUT_EMPLOYEES,
) -> str:
    """
    검색 결과(hits)를 사용자에게 보여줄 답변 문자열로 변환한다.

    - make_sources()를 사용해서 허용된 필드만 추출한다.
    - employee_id 기준으로 같은 직원의 정보를 하나로 묶는다.
    - allowed_fields에 포함된 필드만 답변에 표시한다.
    """

    # 검색 결과에서 사용자에게 보여줘도 되는 필드만 추출한다.
    source_items = make_sources(
        limit_hits_by_employee_count(hits, limit=limit),
        allowed_fields=allowed_fields,
    )

    # 보여줄 데이터가 없으면 조회 결과 없음 메시지를 반환한다.
    if not source_items:
        return NO_SEARCH_RESULT_MESSAGE

    # employee_id가 있는 데이터는 직원별로 묶기 위해 dict 사용
    grouped_items = {}

    # employee_id가 없는 데이터는 따로 보관
    fallback_items = []

    # 검색 결과를 하나씩 확인한다.
    for item in source_items:
        # 직원 식별 기준이 되는 employee_id를 꺼낸다.
        employee_id = item.get("employee_id")
        group_key = employee_id

        # employee_id가 없으면 직원별 그룹핑을 할 수 없으므로 fallback_items에 넣는다.(예외처리)
        if not group_key:
            fallback_items.append(item)
            continue

        # 해당 직원이 아직 grouped_items에 없으면 빈 dict를 만든다.
        if group_key not in grouped_items:
            grouped_items[group_key] = {}

        # 해당 직원의 누적 정보 dict를 가져온다.
        grouped_item = grouped_items[group_key]

        # item 안의 필드들을 하나씩 확인한다.
        for key, value in item.items():
            # 값이 있고, 아직 같은 key가 저장되지 않았으면 저장한다.
            # 같은 직원 정보가 여러 문서에 나뉘어 있을 때 중복 저장을 막기 위함이다.
            if value and key not in grouped_item:
                grouped_item[key] = value

    # 최종 답변 줄들을 담을 리스트
    lines = []

    # 직원별로 묶인 데이터 + employee_id가 없던 데이터를 함께 순회한다.
    for item in list(grouped_items.values()) + fallback_items:
        # 한 사람 또는 한 결과에 대해 출력할 값들을 담는다.
        values = []

        # allowed_fields에 employee가 있으면 이름/사번을 표시한다.
        if "employee" in allowed_fields:
            employee_name = item.get("employee_name")
            employee_id = item.get("employee_id")

            # 이름과 사번 중 존재하는 값만 " / "로 연결한다.
            employee_label = " / ".join(
                value
                for value in [employee_name, employee_id]
                if value
            )

            # 이름 또는 사번이 있으면 출력 값에 추가한다.
            if employee_label:
                values.append(employee_label)

        # 허용된 필드들을 하나씩 출력한다.
        for field in allowed_fields:
            # employee는 위에서 이미 처리했으므로 건너뛴다.
            if field == "employee":
                continue

            # 해당 필드 값이 있으면 답변에 추가한다.
            if item.get(field):
                # FIELD_RULES에 정의된 한글 라벨을 가져온다.
                # 없으면 field 이름을 그대로 사용한다.
                label = FIELD_RULES.get(field, {}).get("label", field)

                # 예: "부서: 마케팅부", "직급: 대리"
                values.append(f"{label}: {item[field]}")

        # 출력할 값이 있으면 "- 값 / 값 / 값" 형태로 한 줄 추가한다.
        if values:
            lines.append("- " + " / ".join(values))

    # 만들어진 줄이 있으면 줄바꿈으로 합쳐 반환한다.
    # 없으면 조회 결과 없음 메시지를 반환한다.
    return "\n".join(lines) if lines else NO_SEARCH_RESULT_MESSAGE

def to_number(value):
    """
    숫자 비교 조건에 사용할 수 있도록 문자열에서 숫자만 꺼낸다.
    """

    if value is None:
        return None

    text = str(value).replace(",", "")
    match = re.search(r"-?\d+(\.\d+)?", text)

    if not match:
        return None

    number_text = match.group()

    if "." in number_text:
        return float(number_text)

    return int(number_text)


def filter_match(actual_value, op: str, expected_value) -> bool:
    """
    검색 결과의 실제 값이 filter 조건 하나를 만족하는지 비교한다.
    """

    if actual_value is None or actual_value == "":
        return False

    actual_text = str(actual_value).strip()
    expected_text = str(expected_value).strip()

    evaluation_aliases = {
        "우수": "A",
        "탁월": "A",
        "최상": "A",
        "좋음": "A",
        "양호": "B",
        "보통": "C",
        "미흡": "D",
        "부진": "D",
    }

    expected_text = evaluation_aliases.get(expected_text, expected_text)

    if op == "exists":
        missing_values = {"미입력", "NULL", "None", "none", "null", "-"}
        return actual_text not in missing_values

    if op == "eq":
        return actual_text == expected_text

    if op == "contains":
        return expected_text in actual_text

    if op in {"gt", "gte", "lt", "lte"}:
        actual_number = to_number(actual_value)
        expected_number = to_number(expected_value)

        if actual_number is None or expected_number is None:
            return False

        if op == "gt":
            return actual_number > expected_number

        if op == "gte":
            return actual_number >= expected_number

        if op == "lt":
            return actual_number < expected_number

        if op == "lte":
            return actual_number <= expected_number

    if op == "between":
        if not isinstance(expected_value, list) or len(expected_value) != 2:
            return False

        actual_number = to_number(actual_value)
        start_number = to_number(expected_value[0])
        end_number = to_number(expected_value[1])

        if actual_number is None or start_number is None or end_number is None:
            return False

        return start_number <= actual_number <= end_number
    

    return False


def filter_hits_by_filters(hits: list[dict], filters: list[dict]) -> list[dict]:
    """
    hybrid 검색 결과를 employee_id 기준으로 묶은 뒤 filters 조건을 후처리한다.

    같은 직원의 정보가 여러 인덱스/문서에 나뉘어 있을 수 있으므로, 문서 하나가 아니라
    직원 단위로 조건을 만족하는지 판단한다.
    """

    if not filters:
        return hits

    grouped_hits = {}

    for hit in hits:
        source = hit.get("_source", {})
        employee_id = source.get("employee_id")

        if not employee_id:
            continue

        grouped_hits.setdefault(employee_id, []).append(hit)

    matched_employee_ids = set()

    for employee_id, employee_hits in grouped_hits.items():
        matched_all_filters = True

        for filter_item in filters:
            field = filter_item.get("field")
            op = filter_item.get("op", "eq")
            expected_value = filter_item.get("value")

            if field not in FIELD_RULES:
                continue

            matched_this_filter = False

            for hit in employee_hits:
                source = hit.get("_source", {})
                embedding_text = source.get("embedding_text", "")

                actual_value = get_allowed_field_value(
                    source=source,
                    embedding_text=embedding_text,
                    field_key=field,
                )

                if filter_match(
                    actual_value=actual_value,
                    op=op,
                    expected_value=expected_value,
                ):
                    matched_this_filter = True
                    break

            if not matched_this_filter:
                matched_all_filters = False
                break

        if matched_all_filters:
            matched_employee_ids.add(employee_id)

    return [
        hit
        for hit in hits
        if hit.get("_source", {}).get("employee_id") in matched_employee_ids
    ]


EVALUATION_SORT_SCORE = {
    "A": 4,
    "B": 3,
    "C": 2,
    "D": 1,
}


def get_sort_value_from_hits(employee_hits: list[dict], field_key: str):
    for hit in employee_hits:
        source = hit.get("_source", {})
        embedding_text = source.get("embedding_text", "")
        value = get_allowed_field_value(
            source=source,
            embedding_text=embedding_text,
            field_key=field_key,
        )

        if value is None or value == "" or value == "미입력":
            continue

        if field_key.startswith("evaluation"):
            return EVALUATION_SORT_SCORE.get(str(value).strip())

        number_value = to_number(value)

        if number_value is not None:
            return number_value

        return str(value)

    return None


def sort_hits_by_task_sort(hits: list[dict], sort: dict | None) -> list[dict]:
    """
    직원 단위로 정렬한 뒤 limit에 해당하는 직원들의 hit만 남긴다.
    """

    if not sort:
        return hits

    field_key = sort.get("field")

    if field_key not in FIELD_RULES:
        return hits

    grouped_hits = {}

    for hit in hits:
        employee_id = hit.get("_source", {}).get("employee_id")

        if not employee_id:
            continue

        grouped_hits.setdefault(employee_id, []).append(hit)

    sortable_items = []

    for employee_id, employee_hits in grouped_hits.items():
        sort_value = get_sort_value_from_hits(employee_hits, field_key)

        if sort_value is None:
            continue

        sortable_items.append(
            {
                "employee_id": employee_id,
                "sort_value": sort_value,
                "hits": employee_hits,
            }
        )

    if not sortable_items:
        return hits

    reverse = sort.get("order", "desc") != "asc"
    sortable_items = sorted(
        sortable_items,
        key=lambda item: item["sort_value"],
        reverse=reverse,
    )

    limit = int(sort.get("limit") or len(sortable_items))
    selected_employee_ids = {
        item["employee_id"]
        for item in sortable_items[:limit]
    }

    return [
        hit
        for hit in hits
        if hit.get("_source", {}).get("employee_id") in selected_employee_ids
    ]


def has_non_org_filters(filters: list[dict]) -> bool:
    """
    department/team/job_grade/position 외의 추가 조건이 있는지 확인한다.
    추가 조건이 있으면 단순 직접 검색이 아니라 hybrid 검색 후 조건 검증 흐름을 탄다.
    """

    org_fields = {"department", "team", "job_grade", "position"}

    for item in filters:
        if item.get("field") not in org_fields:
            return True

    return False


def process_task(
    task: dict,
    original_question: str,
    requester_employee_id: str,
    permission_level: int,
) -> dict:
    
    #task 처리 시간 측정 시작
    # task_start_time = time.perf_counter()
    #task에서 의도 출력
    intent = task.get("intent", "unknown")

    if is_supervisor_query(original_question):
        return {
            "answer": UNSUPPORTED_SUPERVISOR_MESSAGE,
            "sources": [],
            "permission": {
                "allowed": False,
                "ignored": True,
                "reason": "상사 관계 필드가 없어 처리할 수 없습니다.",
                "permission_level": permission_level,
                "required_level": 1,
                "allowed_fields": [],
                "denied_fields": [],
                "is_self": bool(task.get("is_self", False)),
            },
        }

    requested_employee_collection = is_employee_collection_query(original_question)

    if requested_employee_collection:
        task["employee_name"] = None
        task["employee_id"] = None

        if intent == "single_lookup":
            intent = "condition_search"

    # print("[TIME] process_task start:", intent)


    # LLM이 추출한 조직 값은 반드시 코드의 조직 정책 목록으로 다시 검증한다.
    task = normalize_task_org_values(task)

    task = remove_hallucinated_org_values(task, original_question)

    # LLM이 만든 filters를 바로 쓰지 않고 허용 필드/연산자/조직 값 기준으로 정리한다.
    filters = sanitize_filters(task.get("filters", []))

    filters = remove_hallucinated_org_filters(
        filters=filters,
        original_question=original_question,
        task=task,
    )

    filters = remove_ambiguous_previous_task_filter(
        filters=filters,
        original_question=original_question,
        task=task,
    )
    # LLM이 department 관련 조건을 놓쳤으면 filters에 추가한다.
    if task.get("department") and value_appears_in_question(original_question, task.get("department")):
        filters = add_filter_if_missing(
            filters=filters,
            field="department",
            op="eq",
            value=task.get("department"),
        )

    if task.get("team") and value_appears_in_question(original_question, task.get("team")):
        filters = add_filter_if_missing(
            filters=filters,
            field="team",
            op="eq",
            value=task.get("team"),
        )


    # 특정 직원 이름이 있으면 직급은 호칭일 수 있으므로, 이름이 없을 때만 job_grade 필터를 추가한다.
    # ex) "김민수 대리의 부서 알려줘" -> "김민수"를 대상으로 검색해야 하므로
    if (
        task.get("job_grade")
        and not task.get("employee_name")
        and value_appears_in_question(original_question, task.get("job_grade"))
    ):
        filters = add_filter_if_missing(
            filters=filters,
            field="job_grade",
            op="eq",
            value=task.get("job_grade"),
        )

    # 특정 직원 이름이 있으면 직책은 호칭일 수 있으므로, 이름이 없을 때만 position 필터를 추가한다.
    # ex) "김민수 팀장의 부서 알려줘" -> "김민수"를 대상으로 검색해야 하므로
    if (
        task.get("position")
        and not task.get("employee_name")
        and value_appears_in_question(original_question, task.get("position"))
    ):
        filters = add_filter_if_missing(
            filters=filters,
            field="position",
            op="eq",
            value=task.get("position"),
        )

    task["filters"] = filters
    sort = task.get("sort")
    sort_fields = [
        sort.get("field")
    ] if isinstance(sort, dict) and sort.get("field") in FIELD_RULES else []


    target_fields = [
        field
        for field in task.get("target_fields", [])
        if field in FIELD_RULES
    ]
    target_fields = unique_keep_order(target_fields + sort_fields)

    # filters에서 권한 판단에 필요한 field 목록을 꺼낸다.
    filter_fields = get_filter_fields(filters)

    if intent == "unknown" and target_fields:
        intent = "single_lookup"

    if intent == "unknown" or not target_fields:
        # 이 task는 조건에 맞지 않아 처리하지 않고 건너뛴다.
        # print(
        #     "[TIME] process_task에서 무시됨:",
        #     f"{time.perf_counter() - task_start_time:.3f}s",
        # )
        return {
            "answer": COMMON_ERROR_MESSAGE,
            "sources": [],
            "permission": {
                "allowed": False,
                "ignored": True,
                "reason": "분석 가능한 유효 필드가 없습니다.",
                "permission_level": permission_level,
                "required_level": 1,
                "allowed_fields": [],
                "denied_fields": [],
                "is_self": bool(task.get("is_self", False)),
            },
        }

    if (
        intent == "single_lookup"
        and not requested_employee_collection
        and not bool(task.get("is_self", False))
    ):
        if not task.get("employee_name") and not task.get("employee_id") and not task.get("job_grade"):
            guessed_name = extract_employee_name(original_question)

            if guessed_name:
                task["employee_name"] = guessed_name

    if (
        intent == "condition_search"
        and not requested_employee_collection
        and not filters
        and not task.get("employee_name")
        and not task.get("employee_id")
        and not task.get("job_grade")
    ):
        guessed_name = extract_employee_name(original_question)

        if guessed_name:
            task["employee_name"] = guessed_name
            intent = "single_lookup"

    has_explicit_target = any(
        task.get(key)
        for key in ["employee_name", "employee_id", "department", "team", "position"]
    )

    if (
        intent == "single_lookup"
        and not filters
        and not has_explicit_target
        and is_self_question(original_question)
    ):
        task["is_self"] = True

    is_self = bool(task.get("is_self", False))

    actual_employee_name = find_employee_name_in_question(original_question)

    if actual_employee_name:
        task["employee_name"] = actual_employee_name
    elif task.get("employee_name") and task.get("job_grade") and not employee_name_exists(task["employee_name"]):
        task["employee_name"] = None

    # 단일 직원 조회에서 LLM이 이름을 놓친 경우에만 정규식 기반 이름 추출로 보완한다.
    if intent == "single_lookup" and not requested_employee_collection and not is_self:
        if not task.get("employee_name") and not task.get("employee_id") and not task.get("job_grade"):
            guessed_name = extract_employee_name(original_question)

            if guessed_name:
                task["employee_name"] = guessed_name

    code_extracted_name = extract_name_like_prefix(original_question)

    should_validate_employee_name = (
        intent == "single_lookup"
        and code_extracted_name
        and not actual_employee_name
        and not requested_employee_collection
        and not is_self
        and not task.get("employee_id")
        and (
            not task.get("employee_name")
            or task.get("employee_name") == code_extracted_name
        )
    )

    if should_validate_employee_name:
        task["employee_name"] = code_extracted_name

        if not employee_name_exists(code_extracted_name):
            return {
                "answer": NO_SEARCH_RESULT_MESSAGE,
                "sources": [],
                "permission": {
                    "allowed": True,
                    "permission_level": permission_level,
                    "required_level": 1,
                    "allowed_fields": [],
                    "denied_fields": [],
                    "is_self": False,
                },
            }

    # 답변 필드뿐 아니라 조건 필드까지 포함해서 필요한 권한 레벨을 계산한다.
    search_related_fields = unique_keep_order(target_fields + filter_fields + sort_fields)
    # target_fields는 사용자에게 보여줄 필드, filter_fields는 검색 조건으로 필요한 필드다. 둘 다 권한 판단에 필요하다.

    # required_level은 이 task를 처리하기 위해 필요한 권한 레벨이다. 검색 관련 필드 중에서 가장 높은 레벨이 기준이 된다. 검색 관련 필드가 없으면 1레벨로 간주한다(기본 정보는 모두에게 허용된다고 가정).
    required_level = (
        get_max_required_level(search_related_fields)
        if search_related_fields
        else 1
    )

    # 
    answer_fields, denied_answer_fields = split_fields_by_permission(
        target_fields=target_fields,
        permission_level=permission_level,
        is_self=is_self,
    )

    allowed_filter_fields, denied_filter_fields = split_fields_by_permission(
        target_fields=filter_fields,
        permission_level=permission_level,
        is_self=is_self,
    )

    denied_fields = unique_keep_order(denied_answer_fields + denied_filter_fields)
    denied_message = build_denied_message(denied_fields)

    # 조건 필드 권한이 없으면 검색 자체를 막는다. 조건 검색은 조건값 존재 여부도 정보가 될 수 있다.
    if denied_filter_fields:
        # print(
        #     "[TIME] process_task denied_filter:",
        #     f"{time.perf_counter() - task_start_time:.3f}s",
        # )
        return {
            "answer": PERMISSION_DENIED_MESSAGE,
            "sources": [],
            "permission": {
                "allowed": False,
                "permission_level": permission_level,
                "required_level": required_level,
                "allowed_fields": answer_fields,
                "denied_fields": denied_fields,
                "is_self": is_self,
            },
        }

    identity_only_fields = {"employee", "employee_name", "employee_id"}

    if denied_answer_fields and set(answer_fields).issubset(identity_only_fields):
        return {
            "answer": PERMISSION_DENIED_MESSAGE,
            "sources": [],
            "permission": {
                "allowed": False,
                "permission_level": permission_level,
                "required_level": required_level,
                "allowed_fields": answer_fields,
                "denied_fields": denied_fields,
                "is_self": is_self,
            },
        }

    if not answer_fields:
        # print(
        #     "[TIME] process_task denied_answer:",
        #     f"{time.perf_counter() - task_start_time:.3f}s",
        # )
        return {
            "answer": PERMISSION_DENIED_MESSAGE,
            "sources": [],
            "permission": {
                "allowed": False,
                "permission_level": permission_level,
                "required_level": required_level,
                "allowed_fields": [],
                "denied_fields": denied_fields,
                "is_self": is_self,
            },
        }

    # allowed_fields는 사용자에게 보여줄 필드, context_fields는 검색 검증/LLM context에 필요한 필드다.
    # answer_fields = 최종 답변에 보여줄 필드
    # context_fields = LLM 답변 생성에 참고시킬 필드
    # employee = 누구 정보인지 표시하기 위해 추가
    # search_permission_level = 검색할 때 적용할 권한 레벨

    allowed_fields = list(answer_fields)
    context_fields = unique_keep_order(answer_fields + allowed_filter_fields + sort_fields)

    if intent in {"single_lookup", "employee_list", "employee_count", "condition_search"}:
        if "employee" not in allowed_fields:
            allowed_fields.insert(0, "employee")

        if "employee" not in context_fields:
            context_fields.insert(0, "employee")

    search_permission_level = permission_level

    # 본인 조회는 requester_employee_id로 검색 대상을 고정한 뒤 필요한 인덱스까지 조회한다.
    if is_self:
        search_permission_level = max(permission_level, required_level)

    accessible_indices = ACCESSIBLE_INDICES.get(search_permission_level, [])

    # 답변 필드와 허용된 조건 필드가 들어 있는 인덱스만 검색 대상으로 선택한다.
    search_fields = unique_keep_order(answer_fields + allowed_filter_fields + sort_fields)

    # indices는 실제 검색에 사용할 인덱스 목록이다. 
    indices = select_indices_by_fields(
        accessible_indices=accessible_indices,
        allowed_fields=search_fields,
    )

    search_employee_id = None

    if is_self:
        search_employee_id = requester_employee_id
    elif task.get("employee_id"):
        search_employee_id = str(task["employee_id"]).strip().upper()
    # llm이 employee_id 를 추출을 못할 시 직접 짜놓은 로직으로 추출
    else:
        search_employee_id = extract_employee_id(original_question)

    # hybrid 검색에 사용할 질문 생성
    task_question = build_task_question(original_question, task)

        # 직원 수 질문은 검색 결과 size에 의존하지 않고 OpenSearch count로 계산한다.
    if intent == "employee_count":
        # count_start_time = time.perf_counter()

        count = count_employees_by_conditions(
            permission_level=search_permission_level,
            department=task.get("department"),
            team=task.get("team"),
            job_grade=task.get("job_grade"),
            position=task.get("position"),
        )

        answer = build_employee_count_answer(
            task=task,
            count=count,
        )

        if denied_message:
            answer = f"{answer}\n\n{denied_message}"

        # print(
        #     "[TIME] employee_count:",
        #     f"{time.perf_counter() - count_start_time:.3f}s",
        # )
        # print(
        #     "[TIME] process_task total:",
        #     f"{time.perf_counter() - task_start_time:.3f}s",
        # )

        return {
            "answer": answer,
            "sources": [],
            "permission": {
                "allowed": True,
                "permission_level": permission_level,
                "required_level": required_level,
                "allowed_fields": answer_fields,
                "denied_fields": denied_fields,
                "is_self": is_self,
            },
        }

    # 카테고리 목록 질문은 hybrid 검색보다 전체 기본 인덱스 집계가 더 정확하다.
    if intent == "category_list":
        # category_start_time = time.perf_counter()
        answers = [
            build_category_answer(field, search_permission_level)
            for field in answer_fields
            if field in {"department", "team", "job_grade", "position"}
        ]

        answer = (
            "\n\n".join(answers)
            if answers
            else "카테고리 필드를 찾을 수 없습니다."
        )

        if denied_message:
            answer = f"{answer}\n\n{denied_message}"

        # print(
        #     "[TIME] category_list:",
        #     f"{time.perf_counter() - category_start_time:.3f}s",
        # )
        # print(
        #     "[TIME] process_task total:",
        #     f"{time.perf_counter() - task_start_time:.3f}s",
        # )

        return {
            "answer": answer,
            "sources": [],
            "permission": {
                "allowed": True,
                "permission_level": permission_level,
                "required_level": required_level,
                "allowed_fields": answer_fields,
                "denied_fields": denied_fields,
                "is_self": is_self,
            },
        }

    search_hits = []

    # 조직/직급/직책만 묻는 단순 직원 목록은 직접 조건 검색.
    if (
        intent in {"employee_list", "condition_search"}
        and (task.get("department") or task.get("team") or task.get("job_grade") or task.get("position"))
        and set(answer_fields) == {"employee"}
        and not has_non_org_filters(filters)
        and not sort
    ):
        # direct_search_start_time = time.perf_counter()

        # 직접 검색 로직
        search_hits = search_employees_by_conditions(
            permission_level=search_permission_level,
            department=task.get("department"),
            team=task.get("team"),
            job_grade=task.get("job_grade"),
            position=task.get("position"),
            size=10000,
        )

        # 실제 사용자에게 보여줄 답변 문자열로 변환한다. 
        answer = format_allowed_hits_answer(
            hits=search_hits,
            allowed_fields=allowed_fields,
        )
        # print(
        #     "[TIME] direct_employee_search:",
        #     f"{time.perf_counter() - direct_search_start_time:.3f}s",
        # )

    # 필터/정렬이 있는 조건 검색은 유사도 상위 50개가 아니라 전체 후보를 조회한다.
    elif intent in {"condition_search", "employee_list"} and (filters or sort):
        filter_indices = select_indices_by_fields(
            accessible_indices=accessible_indices,
            allowed_fields=unique_keep_order(filter_fields + sort_fields),
        )

        search_hits = search_employees_by_filter_conditions(
            indices=filter_indices or indices,
            filters=filters,
            size=10000,
        )

        search_hits = filter_hits_by_filters(
            hits=search_hits,
            filters=filters,
        )

        matched_employee_ids = unique_keep_order(
            [
                hit.get("_source", {}).get("employee_id")
                for hit in search_hits
                if hit.get("_source", {}).get("employee_id")
            ]
        )

        if matched_employee_ids:
            search_hits = search_hits_by_employee_ids(
                indices=indices,
                employee_ids=matched_employee_ids,
                size=10000,
            )

        search_hits = filter_hits_with_answer_values(
            hits=search_hits,
            answer_fields=answer_fields,
        )
        search_hits = sort_hits_by_task_sort(search_hits, sort)

        if not search_hits:
            answer = NO_SEARCH_RESULT_MESSAGE
        else:
            answer = format_allowed_hits_answer(
                hits=search_hits,
                allowed_fields=allowed_fields,
            )

    # 그 외 단일 조회/복합 조건 검색은 hybrid 검색 후 코드에서 조건을 재검증한다.
    else:
        # hybrid_start_time = time.perf_counter()

        search_employee_name = None

        if intent == "single_lookup":
            search_employee_name = task.get("employee_name")
        
        # BM25 검색과 벡터 검색을 각각 수행한 뒤, RRF 방식으로 결과를 병합한다.
        search_hits = search_hybrid(
            question=task_question,
            permission_level=search_permission_level,
            employee_id=search_employee_id,
            employee_name=search_employee_name,
            extract_name=False,
            size=50,
            indices=indices,
            filters=filters,
        )
        # print(
        #     "[TIME] search_hybrid call:",
        #     f"{time.perf_counter() - hybrid_start_time:.3f}s",
        # )

        # filter_start_time = time.perf_counter()

        # hybrid 검색 결과를 employee_id 기준으로 묶은 뒤 filters 조건을 후처리한다.
        # 같은 직원의 정보가 여러 인덱스/문서에 나뉘어 있을 수 있으므로, 문서 하나가 아니라 직원 단위로 조건을 만족하는지 판단한다.
        search_hits = filter_hits_by_filters(
            hits=search_hits,
            filters=filters,
        )

        # 사용자가 실제로 요청한 필드 값이 있는 검색 결과만 남긴다.
        search_hits = filter_hits_with_answer_values(
            hits=search_hits,
            answer_fields=answer_fields,
        )
        search_hits = sort_hits_by_task_sort(search_hits, sort)
        # print(
        #     "[TIME] filter_hits:",
        #     f"{time.perf_counter() - filter_start_time:.3f}s",
        #     "hits:",
        #     len(search_hits),
        # )

        if not search_hits:
            answer = NO_SEARCH_RESULT_MESSAGE
        elif requested_employee_collection:
            answer = format_allowed_hits_answer(
                hits=search_hits,
                allowed_fields=allowed_fields,
            )
        elif intent in {"condition_search", "employee_list"} and filters:
            answer = format_allowed_hits_answer(
                hits=search_hits,
                allowed_fields=allowed_fields,
            )
        elif intent == "condition_search" and (task.get("department") or task.get("team")):
            answer = format_allowed_hits_answer(
                hits=search_hits,
                allowed_fields=allowed_fields,
            )
        else:
            # context_start_time = time.perf_counter()
            context = build_context(
                search_hits,
                task_question,
                allowed_fields=context_fields,
            )
            # print(
            #     "[TIME] build_context:",
            #     f"{time.perf_counter() - context_start_time:.3f}s",
            # )

            answer = generate_answer(
                question=task_question,
                context=context,
            )

    if denied_message:
        answer = f"{answer}\n\n{denied_message}"

    # sources_start_time = time.perf_counter()
    sources = make_sources(
        limit_hits_by_employee_count(search_hits),
        allowed_fields=allowed_fields,
    )
    # print(
    #     "[TIME] make_sources:",
    #     f"{time.perf_counter() - sources_start_time:.3f}s",
    # )
    # print(
    #     "[TIME] process_task total:",
    #     f"{time.perf_counter() - task_start_time:.3f}s",
    # )

    return {
        "answer": answer,
        "sources": sources,
        "permission": {
            "allowed": True,
            "permission_level": permission_level,
            "required_level": required_level,
            "allowed_fields": answer_fields,
            "denied_fields": denied_fields,
            "is_self": is_self,
        },
    }
