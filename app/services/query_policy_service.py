FIELD_RULES = {
    # =========================
    # 직원 표시용
    # =========================
    "employee": {
        "label": "직원",
        "doc_type": "basic",
        "index_level": 1,
        "required_level": 1,
    },

    # =========================
    # 기본인사정보 - hr_basic_1
    # 샘플 기준으로 embedding_text에 들어가 있는 기본 공개/조직 정보
    # =========================
    "employee_id": {
        "label": "사원번호",
        "doc_type": "basic",
        "index_level": 1,
        "required_level": 1,
        "source_field": "employee_id",
    },
    "employee_name": {
        "label": "이름",
        "doc_type": "basic",
        "index_level": 1,
        "required_level": 1,
        "source_field": "employee_name",
    },
    "company": {
        "label": "회사명",
        "doc_type": "basic",
        "index_level": 1,
        "required_level": 1,
        "embedding_label": "회사명",
    },
    "work_location": {
        "label": "사업장위치",
        "doc_type": "basic",
        "index_level": 1,
        "required_level": 1,
        "embedding_label": "사업장위치",
    },
    "department": {
        "label": "부서",
        "doc_type": "basic",
        "index_level": 1,
        "required_level": 1,
        "source_field": "department",
    },
    "team": {
        "label": "팀",
        "doc_type": "basic",
        "index_level": 1,
        "required_level": 1,
        "embedding_label": "팀",
    },
    "department_level": {
        "label": "부서레벨",
        "doc_type": "basic",
        "index_level": 1,
        "required_level": 1,
        "source_field": "department_level",
    },
    "job_grade": {
        "label": "직급",
        "doc_type": "basic",
        "index_level": 1,
        "required_level": 1,
        "source_field": "job_grade",
    },
    "position": {
        "label": "직책",
        "doc_type": "basic",
        "index_level": 1,
        "required_level": 1,
        "embedding_label": "직책",
    },
    "job_grade_level": {
        "label": "직급레벨",
        "doc_type": "basic",
        "index_level": 1,
        "required_level": 1,
        "source_field": "job_grade_level",
    },
    "retirement_type": {
        "label": "퇴직구분",
        "doc_type": "basic",
        "index_level": 1,
        "required_level": 1,
        "embedding_label": "퇴직구분",
    },
    "retirement_date": {
        "label": "퇴직일자",
        "doc_type": "basic",
        "index_level": 1,
        "required_level": 1,
        "embedding_label": "퇴직일자",
    },
    "email": {
        "label": "이메일",
        "doc_type": "basic",
        "index_level": 1,
        "required_level": 1,
        "embedding_label": "이메일",
    },

    # 샘플에서 hr_basic_1 embedding_text에 보였던 필드
    # 단, 민감도는 required_level로 따로 관리한다.
    "gender": {
        "label": "성별",
        "doc_type": "basic",
        "index_level": 1,
        "required_level": 2,
        "embedding_label": "성별",
    },
    "age": {
        "label": "나이",
        "doc_type": "basic",
        "index_level": 1,
        "required_level": 2,
        "embedding_label": "나이",
    },
    "hire_date": {
        "label": "입사일",
        "doc_type": "basic",
        "index_level": 1,
        "required_level": 2,
        "embedding_label": "입사일",
    },
    "tenure": {
        "label": "근속기간",
        "doc_type": "basic",
        "index_level": 1,
        "required_level": 2,
        "embedding_label": "근속기간",
    },
    "hire_path": {
        "label": "채용경로",
        "doc_type": "basic",
        "index_level": 1,
        "required_level": 2,
        "embedding_label": "채용경로",
    },
    "contract_type": {
        "label": "계약형태",
        "doc_type": "basic",
        "index_level": 1,
        "required_level": 2,
        "embedding_label": "계약형태",
    },

    # =========================
    # 기본인사정보 - hr_basic_2
    # 개인 상세 정보
    # =========================
    "birth_date": {
        "label": "생년월일",
        "doc_type": "basic",
        "index_level": 2,
        "required_level": 2,
        "embedding_label": "생년월일",
    },
    "phone": {
        "label": "전화번호",
        "doc_type": "basic",
        "index_level": 2,
        "required_level": 2,
        "embedding_label": "전화번호",
    },
    "military": {
        "label": "병역",
        "doc_type": "basic",
        "index_level": 2,
        "required_level": 2,
        "embedding_label": "병역",
    },
    "education": {
        "label": "학력",
        "doc_type": "basic",
        "index_level": 2,
        "required_level": 2,
        "embedding_label": "학력",
    },
    "university": {
        "label": "출신대학",
        "doc_type": "basic",
        "index_level": 2,
        "required_level": 2,
        "embedding_label": "출신대학",
    },
    "gpa": {
        "label": "학점",
        "doc_type": "basic",
        "index_level": 2,
        "required_level": 2,
        "embedding_label": "학점",
    },
    "previous_company": {
        "label": "이전직장명",
        "doc_type": "basic",
        "index_level": 2,
        "required_level": 2,
        "embedding_label": "이전직장명",
    },
    "previous_job_grade": {
        "label": "이전최종직급",
        "doc_type": "basic",
        "index_level": 2,
        "required_level": 2,
        "embedding_label": "이전최종직급",
    },
    "previous_task": {
        "label": "이전담당업무",
        "doc_type": "basic",
        "index_level": 2,
        "required_level": 2,
        "embedding_label": "이전담당업무",
    },

    # =========================
    # 기본인사정보 - hr_basic_3
    # 매우 민감한 개인정보
    # =========================
    "rrn": {
        "label": "주민등록번호",
        "doc_type": "basic",
        "index_level": 3,
        "required_level": 3,
        "embedding_label": "주민등록번호",
    },
    "address": {
        "label": "주소",
        "doc_type": "basic",
        "index_level": 3,
        "required_level": 3,
        "embedding_label": "주소",
    },

    # =========================
    # 급여정보 - hr_salary_2
    # =========================
    "overtime": {
        "label": "잔업시간",
        "doc_type": "salary",
        "index_level": 2,
        "required_level": 2,
        "embedding_label": "잔업시간",
    },
    "unused_vacation": {
        "label": "미사용휴가일수",
        "doc_type": "salary",
        "index_level": 2,
        "required_level": 2,
        "embedding_label": "미사용휴가일수",
    },

    # =========================
    # 급여정보 - hr_salary_3
    # =========================
    "salary": {
        "label": "연봉",
        "doc_type": "salary",
        "index_level": 3,
        "required_level": 3,
        "embedding_label": "연봉",
    },
    "salary_bank": {
        "label": "급여은행",
        "doc_type": "salary",
        "index_level": 3,
        "required_level": 3,
        "embedding_label": "급여은행",
    },
    "account": {
        "label": "계좌번호",
        "doc_type": "salary",
        "index_level": 3,
        "required_level": 3,
        "embedding_label": "계좌번호",
    },
    "insurance": {
        "label": "4대보험가입여부",
        "doc_type": "salary",
        "index_level": 3,
        "required_level": 3,
        "embedding_label": "4대보험가입여부",
    },

    # =========================
    # 역량성과 - hr_performance_2
    # =========================
    "performance_score": {
        "label": "성과점수",
        "doc_type": "performance",
        "index_level": 2,
        "required_level": 2,
        "embedding_label": "성과점수",
    },
    "evaluation": {
        "label": "인사고과",
        "doc_type": "performance",
        "index_level": 2,
        "required_level": 2,
        "embedding_label": "인사고과_2024",
    },
    "evaluation_2020": {
        "label": "인사고과_2020",
        "doc_type": "performance",
        "index_level": 2,
        "required_level": 2,
        "embedding_label": "인사고과_2020",
    },
    "evaluation_2021": {
        "label": "인사고과_2021",
        "doc_type": "performance",
        "index_level": 2,
        "required_level": 2,
        "embedding_label": "인사고과_2021",
    },
    "evaluation_2022": {
        "label": "인사고과_2022",
        "doc_type": "performance",
        "index_level": 2,
        "required_level": 2,
        "embedding_label": "인사고과_2022",
    },
    "evaluation_2023": {
        "label": "인사고과_2023",
        "doc_type": "performance",
        "index_level": 2,
        "required_level": 2,
        "embedding_label": "인사고과_2023",
    },
    "evaluation_2024": {
        "label": "인사고과_2024",
        "doc_type": "performance",
        "index_level": 2,
        "required_level": 2,
        "embedding_label": "인사고과_2024",
    },
    "certificate": {
        "label": "자격증",
        "doc_type": "performance",
        "index_level": 2,
        "required_level": 2,
        "embedding_label": "자격증",
    },
    "toeic": {
        "label": "TOEIC점수",
        "doc_type": "performance",
        "index_level": 2,
        "required_level": 2,
        "embedding_label": "TOEIC점수",
    },
    "certificate_allowance": {
        "label": "자격증수당여부",
        "doc_type": "performance",
        "index_level": 2,
        "required_level": 2,
        "embedding_label": "자격증수당여부",
    },
    "award_history": {
        "label": "포상이력",
        "doc_type": "performance",
        "index_level": 2,
        "required_level": 2,
        "embedding_label": "포상이력",
    },

    # =========================
    # 역량성과 - hr_performance_3
    # =========================
    "disciplinary_history": {
        "label": "징계이력",
        "doc_type": "performance",
        "index_level": 3,
        "required_level": 3,
        "embedding_label": "징계이력",
    },
    "disciplinary_reason": {
        "label": "징계사유",
        "doc_type": "performance",
        "index_level": 3,
        "required_level": 3,
        "embedding_label": "징계사유",
    },
}


ALLOWED_INTENTS = {
    "single_lookup",
    "employee_list",
    "category_list",
    "condition_search",
    "unknown",
}


ALLOWED_FIELDS = set(FIELD_RULES.keys()) | {"unknown"}


def split_fields_by_permission(
    target_fields: list[str],
    permission_level: int,
    is_self: bool,
):
    """
    요청 필드를 권한 기준으로 허용/차단으로 나눈다.

    중요:
    - LLM이 권한 판단을 하지 않는다.
    - 권한 판단은 반드시 FIELD_RULES와 permission_level로만 한다.
    """

    allowed_fields = []
    denied_fields = []

    for field in target_fields:
        rule = FIELD_RULES.get(field)

        if not rule:
            continue

        required_level = rule["required_level"]

        if is_self or permission_level >= required_level:
            allowed_fields.append(field)
        else:
            denied_fields.append(field)

    return allowed_fields, denied_fields


def select_doc_types_by_fields(fields: list[str]) -> list[str]:
    """
    필드 목록을 보고 필요한 문서 유형을 고른다.

    예:
    - salary -> salary
    - address -> basic
    - performance_score -> performance
    """

    doc_types = set()

    for field in fields:
        rule = FIELD_RULES.get(field)

        if not rule:
            continue

        doc_types.add(rule["doc_type"])

    return list(doc_types)


def select_indices_by_fields(
    accessible_indices: list[str],
    allowed_fields: list[str],
) -> list[str]:
    """
    필드 목록을 보고 실제 검색할 인덱스를 고른다.

    예:
    - department -> hr_basic_1
    - phone -> hr_basic_2
    - address -> hr_basic_3
    - salary -> hr_salary_3
    """

    selected_index_names = set()

    for field in allowed_fields:
        rule = FIELD_RULES.get(field)

        if not rule:
            continue

        doc_type = rule.get("doc_type")
        index_level = rule.get("index_level")

        if not doc_type or not index_level:
            continue

        selected_index_names.add(f"hr_{doc_type}_{index_level}")

    if not selected_index_names:
        return accessible_indices

    return [
        index
        for index in accessible_indices
        if index in selected_index_names
    ]


def get_max_required_level(fields: list[str]) -> int:
    """
    여러 필드 중 가장 높은 required_level을 반환한다.
    """

    if not fields:
        return 1

    max_level = 1

    for field in fields:
        rule = FIELD_RULES.get(field)

        if rule is None:
            max_level = max(max_level, 999)
            continue

        max_level = max(max_level, rule.get("required_level", 999))

    return max_level