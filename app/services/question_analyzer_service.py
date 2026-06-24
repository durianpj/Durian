import json
import re
import requests
from common.filter_utils import add_filter_if_missing, unique_keep_order, value_appears_in_question
from common.text_utils import compact_text, parse_bool, to_none
from app.services.llm_service import call_llm_completion
from app.services.candidate_extractor_service import format_candidates_for_prompt

from app.services.question_service import (
    extract_employee_id,
    extract_employee_name,
    is_employee_collection_query,
    is_self_question,
)
from common.hr_fields import (
    ALLOWED_FIELDS,
    ALLOWED_INTENTS,
    FIELD_RULES,
)

from common.hr_master_data import (
    DEPARTMENTS,
    TEAMS,
    JOB_GRADES,
    POSITIONS,
)


# Gemma/Ollama로 되돌릴 때 참고용으로 남겨둔다.
# OLLAMA_URL = "http://localhost:11434/api/generate"
# MODEL_NAME = "gemma3:4b"


DEFAULT_TASK = {
    "intent": "unknown",
    "target_fields": ["unknown"],
    "employee_name": None,
    "employee_id": None,
    "department": None,
    "team": None,
    "job_grade": None,
    "position": None,
    "filters": [],
    "sort": None,
    "is_self": False,
}


ALLOWED_FILTER_OPS = {
    "eq",        # 정확히 일치
    "contains", # 포함
    "gt",       # 초과
    "gte",      # 이상
    "lt",       # 미만
    "lte",      # 이하
    "between",  # 범위
    "is_date",  # YYYY-MM-DD 날짜 형식인지 확인
    "exists",   # 값이 실제로 입력되어 있는지 확인
}


def build_field_schema_text() -> str:
    """
    FIELD_RULES 기준으로 LLM에게 보여줄 HR 필드 목록을 만든다.

    예)  
    - employee: 직원
    - department: 부서
    - salary: 연봉
    - unknown: 알 수 없는 필드 로 바꿉니다.     

    FIELD_RULES 로 관리하면 된다.
    """

    lines = []

    for field_key, rule in FIELD_RULES.items():
        label = rule.get("label", field_key)
        lines.append(f"- {field_key}: {label}")

    lines.append("- unknown: 알 수 없는 필드")

    return "\n".join(lines)


def clean_json_text(text: str) -> str:
    """
    LLM 응답에서 JSON 부분만 추출한다.
    """

    text = text.strip()
    # LLM 응답이 ```json ... ``` 형태의 마크다운 코드블록으로 감싸져 올 수 있으므로 제거한다.
    text = re.sub(r"^```json", "", text)

    # json 언어 표시 없이 ``` 로만 시작하는 코드블록도 제거한다.
    text = re.sub(r"^```", "", text)

    # 응답 마지막에 붙은 코드블록 종료 표시 ``` 를 제거한다.
    text = re.sub(r"```$", "", text)
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and start < end:
        text = text[start : end + 1]

    return text

def parse_target_fields(value) -> list[str]:
    """
    target_fields=employee_name,department 형태를 리스트로 바꾼다.
    """

    value = to_none(value)

    if not value:
        return ["unknown"]

    fields = [
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    ]

    if not fields:
        return ["unknown"]

    return fields


def parse_filters(value) -> list[dict]:
    """
    filters=field|op|value 형태를 filters 리스트로 바꾼다.

    예:
    filters=hire_date|contains|2024
    ->
    [{"field": "hire_date", "op": "contains", "value": "2024"}]
    """

    value = to_none(value)

    if not value:
        return []

    filters = []

    normalized_value = re.sub(r"\s+and\s+", ";", str(value), flags=re.IGNORECASE)

    for raw_filter in normalized_value.split(";"):
        parts = [
            item.strip()
            for item in raw_filter.split("|")
        ]

        if len(parts) != 3:
            continue

        field, op, filter_value = parts
        filter_value = to_none(filter_value)

        if filter_value is None:
            continue

        filters.append(
            {
                "field": field,
                "op": op,
                "value": filter_value,
            }
        )

    return filters


def parse_key_value_analysis(raw_text: str) -> dict:
    """
    LLM이 반환한 key=value 텍스트를 기존 {"tasks": [task]} 구조로 조립한다.
    """

    parsed = {}

    for line in raw_text.splitlines():
        line = line.strip()

        if not line:
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip()

    task = {
        "intent": to_none(parsed.get("intent")) or "unknown",
        "target_fields": parse_target_fields(parsed.get("target_fields")),
        "employee_name": to_none(parsed.get("employee_name")),
        "employee_id": to_none(parsed.get("employee_id")),
        "department": to_none(parsed.get("department")),
        "team": to_none(parsed.get("team")),
        "job_grade": to_none(parsed.get("job_grade")),
        "position": to_none(parsed.get("position")),
        "filters": parse_filters(parsed.get("filters")),
        "sort": None,
        "is_self": parse_bool(parsed.get("is_self")),
    }

    return {
        "tasks": [task]
    }


def build_fallback_analysis(question: str) -> dict:
    """
    LLM 응답 JSON이 깨졌을 때 사용하는 최소 fallback 분석기.

    정상 경로는 LLM 분석 결과를 사용하고, 이 함수는 JSON 파싱 실패 시에만 사용한다.
    """

    compact_question = compact_text(question)
    target_fields = []

    if (
        is_employee_collection_query(question)
        and ("퇴사" in compact_question or "퇴직" in compact_question)
    ):
        return {
            "tasks": [
                {
                    "intent": "condition_search",
                    "target_fields": ["employee"],
                    "employee_name": None,
                    "employee_id": None,
                    "department": None,
                    "team": None,
                    "position": None,
                    "filters": [
                        {
                            "field": "retirement_date",
                            "op": "exists",
                            "value": True,
                        }
                    ],
                    "is_self": False,
                }
            ]
        }

    keyword_to_field = [
        (["주민등록번호", "주민번호"], "rrn"),
        (["사원번호", "사번"], "employee_id"),
        (["이메일", "메일"], "email"),
        (["전화번호", "연락처", "휴대폰", "핸드폰"], "phone"),
        (["주소"], "address"),
        (["연봉"], "salary"),
        (["계좌번호", "계좌"], "account"),
        (["부서"], "department"),
        (["팀"], "team"),
        (["직급"], "job_grade"),
        (["직책"], "position"),
        (["입사일"], "hire_date"),
        (["계약형태", "계약직", "정규직"], "contract_type"),
        (["성과점수"], "performance_score"),
        (["평가", "고과", "인사고과"], get_evaluation_field_from_question(compact_question)),
        (["토익점수", "토익", "TOEIC점수", "TOEIC"], "toeic"),
        (["징계이력", "징계"], "disciplinary_history"),
    ]

    for keywords, field in keyword_to_field:
        if any(keyword in compact_question for keyword in keywords):
            target_fields.append(field)

    if not target_fields:
        return {"tasks": [DEFAULT_TASK.copy()]}

    employee_id = extract_employee_id(question)
    employee_name = None if employee_id else extract_employee_name(question)
    is_self = is_self_question(question)
    intent = "condition_search" if is_employee_collection_query(question) else "single_lookup"

    return {
        "tasks": [
            {
                "intent": intent,
                "target_fields": target_fields,
                "employee_name": employee_name,
                "employee_id": employee_id,
                "department": None,
                "team": None,
                "position": None,
                "filters": [],
                "is_self": is_self,
            }
        ]
    }


def find_known_value(question: str, values: list[str]) -> str | None:
    """
    질문 안에서 미리 정의된 부서/팀/직책 값을 찾는다.
    이건 조건 하드코딩이 아니라 조직 기준값 보정이다.
    """

    compact_question = compact_text(question)

    for value in values:
        if value in compact_question:
            return value

    return None


ORG_ALIAS_RULES = [
    {
        "keywords": ["영업", "영업관련", "영업쪽", "영업담당"],
        "field": "department",
        "value": "영업부",
    },
    {
        "keywords": ["개발", "개발관련", "개발쪽", "개발담당"],
        "field": "department",
        "value": "개발부",
    },
    {
        "keywords": ["인사", "인사관련", "인사쪽", "인사담당"],
        "field": "department",
        "value": "인사부",
    },
    {
        "keywords": ["기획", "기획관련", "기획쪽", "기획담당"],
        "field": "department",
        "value": "기획부",
    },
    {
        "keywords": ["마케팅", "마케팅관련", "마케팅쪽", "마케팅담당"],
        "field": "department",
        "value": "마케팅부",
    },
    {
        "keywords": ["채용", "채용관련", "채용쪽", "채용담당"],
        "field": "team",
        "value": "채용팀",
    },
]


def find_org_alias(question: str) -> tuple[str, str] | None:
    """
    질문의 조직 별칭을 실제 부서/팀 값으로 보정한다.

    예:
    - "영업 관련 직원" -> ("department", "영업부")
    - "채용 계약직" -> ("team", "채용팀")

    조건 if문을 늘리는 용도가 아니라, 조직 기준값 사전 보정만 담당한다.
    """

    compact_question = compact_text(question)
    non_org_phrases = [
        "인사고과",
        "인사평가",
    ]

    for rule in ORG_ALIAS_RULES:
        matched_keyword = next(
            (
                keyword
                for keyword in sorted(rule["keywords"], key=len, reverse=True)
                if keyword in compact_question
            ),
            None,
        )

        if not matched_keyword:
            continue

        if any(phrase in compact_question for phrase in non_org_phrases):
            if matched_keyword == "인사":
                continue

        return rule["field"], rule["value"]

    return None


def is_org_alias_text(text: str | None) -> bool:
    """
    employee_name 자리에 들어오면 안 되는 조직/별칭 표현인지 확인한다.
    """

    if not text:
        return False

    compact_value = compact_text(str(text))

    if compact_value in DEPARTMENTS or compact_value in TEAMS or compact_value in JOB_GRADES or compact_value in POSITIONS:
        return True

    if compact_value.endswith("관련"):
        return True

    for rule in ORG_ALIAS_RULES:
        if compact_value in rule["keywords"]:
            return True

    return False


def is_self_placeholder_name(text: str | None) -> bool:
    """
    본인 표현이 employee_name으로 잘못 들어온 경우를 구분한다.

    예:
    - "내부서", "내연봉", "본인주소"는 실제 직원명이 아니다.
    - 반면 LLM이 명확한 직원명을 넣은 경우에는 본인 조회로 덮어쓰지 않는다.
    """

    if not text:
        return False

    compact_value = compact_text(str(text))

    invalid_self_names = {
        "true",
        "false",
        "null",
        "none",
        "나",
        "내",
        "본인",
        "내이름",
        "내성명",
        "본인이름",
        "본인성명",
        "내부서",
        "내연봉",
        "내주소",
        "내계약형태",
    }

    if compact_value in invalid_self_names:
        return True

    self_prefixes = ["내", "본인"]
    field_words = ["부서", "연봉", "주소", "계약", "정규직", "이름", "사번"]

    return any(compact_value.startswith(prefix) for prefix in self_prefixes) and any(
        word in compact_value
        for word in field_words
    )


def is_self_placeholder_filter_value(field: str | None, value) -> bool:
    """
    "내 이메일" 같은 질문을 LLM이 email == "내이메일" 조건으로 만든 경우를 제거하기 위한 판별.
    본인 조회에서 이런 값은 검색 조건이 아니라 조회 대상 표현이다.
    """

    if not field or value is None:
        return False

    if not isinstance(value, str):
        return False

    compact_value = compact_text(value)

    if compact_value in {"나", "내", "본인"}:
        return True

    rule = FIELD_RULES.get(field, {})
    field_words = [
        str(word)
        for word in [
            rule.get("label"),
            rule.get("embedding_label"),
            field,
        ]
        if word
    ]

    field_words.extend(
        {
            "메일" if field == "email" else "",
            "급여" if field == "salary" else "",
            "정규직" if field == "contract_type" else "",
            "계약직" if field == "contract_type" else "",
        }
    )
    field_words = [compact_text(word) for word in field_words if word]

    return (
        compact_value.startswith("내")
        or compact_value.startswith("본인")
    ) and any(word and word in compact_value for word in field_words)


def is_limit_placeholder_name(text: str | None) -> bool:
    if not text:
        return False

    compact_value = compact_text(str(text))

    return bool(
        re.fullmatch(r"\d*(명|개|건)만?", compact_value)
        or compact_value in {"명만", "개만", "건만"}
    )


EVALUATION_GRADE_WORDS = {
    "우수": "A",
    "탁월": "A",
    "최상": "A",
    "좋음": "A",
    "양호": "B",
    "보통": "C",
    "미흡": "D",
    "부진": "D",
}


def get_evaluation_field_from_question(question: str) -> str:
    """
    평가 연도가 질문에 있으면 해당 연도 필드를, 없으면 최신 평가 필드를 사용한다.
    """

    for year in ["2020", "2021", "2022", "2023", "2024"]:
        if year in question:
            return f"evaluation_{year}"

    return "evaluation_2024"


def normalize_semantic_filters(filters: list[dict], question: str) -> list[dict]:
    """
    LLM이 자주 헷갈리는 의미 필터를 보정한다.

    예:
    - "평가가 우수한"은 최신 인사고과_2024 eq "A"다.
    - "2023년 평가가 우수한"은 evaluation_2023 eq "A"다.
    - "성과점수 80점 이상"처럼 숫자와 점수가 명시된 경우는 기존 숫자 조건을 유지한다.
    """

    if not filters:
        return filters

    compact_question = compact_text(question)
    evaluation_grade = None

    for word, grade in EVALUATION_GRADE_WORDS.items():
        if word in compact_question:
            evaluation_grade = grade
            break

    if not evaluation_grade:
        return filters

    normalized_filters = []
    has_evaluation_filter = False
    evaluation_field = get_evaluation_field_from_question(compact_question)
    evaluation_fields = {
        "evaluation",
        "evaluation_2020",
        "evaluation_2021",
        "evaluation_2022",
        "evaluation_2023",
        "evaluation_2024",
    }

    for item in filters:
        if not isinstance(item, dict):
            continue

        field = item.get("field")

        if field in evaluation_fields:
            has_evaluation_filter = True
            normalized_filters.append(
                {
                    "field": field if field != "evaluation" else evaluation_field,
                    "op": "eq",
                    "value": evaluation_grade,
                }
            )
            continue

        normalized_filters.append(item)

    if not has_evaluation_filter and ("평가" in compact_question or "고과" in compact_question):
        normalized_filters.append(
            {
                "field": evaluation_field,
                "op": "eq",
                "value": evaluation_grade,
            }
        )

    return normalized_filters


def normalize_employee_date_year_filter(filters: list[dict], question: str) -> list[dict]:
    """
    '2024년에 입사한 사람', '2024년에 퇴사한 사람' 같은 질문을
    날짜 조건 필터로 보정한다.

    예:
    - 2024년에 입사한 사람 -> hire_date contains 2024
    - 2024년에 퇴사한 사람 -> retirement_date contains 2024
    """

    compact_question = compact_text(question)

    year_match = re.search(r"(19|20)\d{2}", compact_question)

    if not year_match:
        return filters

    year = year_match.group()

    for item in filters:
        if not isinstance(item, dict):
            continue

        if (
            item.get("field") in {"hire_date", "retirement_date"}
            and item.get("op") == "eq"
            and str(item.get("value", "")).strip() == year
        ):
            item["op"] = "contains"

    if "입사" in compact_question:
        filters = add_filter_if_missing(
            filters=filters,
            field="hire_date",
            op="contains",
            value=year,
        )

    if "퇴사" in compact_question or "퇴직" in compact_question:
        filters = add_filter_if_missing(
            filters=filters,
            field="retirement_date",
            op="contains",
            value=year,
        )

    return filters
def normalize_retired_employee_filter(filters: list[dict], question: str) -> list[dict]:
    """
    '퇴사한 사람', '퇴직자'처럼 연도가 없는 퇴사자 질문을 보정한다.

    데이터 기준:
    - 퇴직일자: 미입력       -> 재직자
    - 퇴직일자: 2024-10-17  -> 퇴사자

    따라서 퇴사자는 retirement_date 값이 "미입력"이 아닌 사람으로 본다.
    """

    compact_question = compact_text(question)

    retirement_keywords = ["퇴사", "퇴직", "퇴사자", "퇴직자"]
    collection_keywords = ["사람", "직원", "사원", "대상자", "퇴사자", "퇴직자"]

    # 퇴사/퇴직 관련 질문이 아니면 건드리지 않는다.
    if not any(keyword in compact_question for keyword in retirement_keywords):
        return filters

    # 직원 집합을 묻는 질문이 아니면 건드리지 않는다.
    if not any(keyword in compact_question for keyword in collection_keywords):
        return filters

    filters = [
        item
        for item in filters
        if not (
            item.get("field") == "retirement_date"
            and to_none(item.get("value")) is None
        )
    ]

    # 이미 유효한 퇴직일자 조건이 있으면 추가하지 않는다.
    # 예: "2024년 퇴사한 사람"은 contains 2024가 이미 들어간다.
    for item in filters:
        if item.get("field") == "retirement_date":
            return filters

    filters.append(
        {
            "field": "retirement_date",
            "op": "exists",
            "value": True,
        }
    )

    return filters


def normalize_numeric_threshold_filters(filters: list[dict], question: str) -> list[dict]:
    """
    "근속기간 10년 이상", "성과점수 80점 이상" 같은 숫자 조건을 보정한다.
    """

    compact_question = compact_text(question)

    numeric_targets = [
        {
            "field": "tenure",
            "keywords": ["근속기간", "근속"],
            "unit": "년",
        },
        {
            "field": "performance_score",
            "keywords": ["성과점수", "성과"],
            "unit": "점",
        },
        {
            "field": "age",
            "keywords": ["나이"],
            "unit": "세",
        },
        {
            "field": "overtime",
            "keywords": ["잔업시간", "잔업"],
            "unit": "시간",
        },
        {
            "field": "unused_vacation",
            "keywords": ["미사용휴가일수", "미사용휴가"],
            "unit": "일",
        },
    ]

    op_by_word = {
        "이상": "gte",
        "이하": "lte",
        "초과": "gt",
        "미만": "lt",
    }

    for target in numeric_targets:
        if not any(keyword in compact_question for keyword in target["keywords"]):
            continue

        for word, op in op_by_word.items():
            patterns = [
                rf"(?:{'|'.join(target['keywords'])})(\d+)(?:{target['unit']})?{word}",
                rf"(\d+)(?:{target['unit']})?{word}(?:.*)(?:{'|'.join(target['keywords'])})",
            ]

            for pattern in patterns:
                match = re.search(pattern, compact_question)

                if not match:
                    continue

                filters = add_filter_if_missing(
                    filters=filters,
                    field=target["field"],
                    op=op,
                    value=match.group(1),
                )
                break

    return filters


def infer_sort_from_question(question: str) -> dict | None:
    """
    "가장 높은/낮은/오래된" 같은 표현을 정렬 조건으로 바꾼다.
    """

    compact_question = compact_text(question)

    sort_keywords = [
        "가장",
        "제일",
        "최고",
        "최저",
        "높은",
        "좋은",
        "우수",
        "낮은",
        "많은",
        "적은",
        "긴",
        "짧은",
        "오래",
    ]

    if not any(keyword in compact_question for keyword in sort_keywords):
        return None

    sort_field = None

    if "근속" in compact_question:
        sort_field = "tenure"
    elif (
        "성과점수" in compact_question
        or "성과" in compact_question
        or "성적" in compact_question
    ):
        sort_field = "performance_score"
    elif "평가" in compact_question or "고과" in compact_question:
        sort_field = get_evaluation_field_from_question(compact_question)
    elif "연봉" in compact_question or "급여" in compact_question:
        sort_field = "salary"
    elif "나이" in compact_question:
        sort_field = "age"
    elif "잔업" in compact_question:
        sort_field = "overtime"
    elif "미사용휴가" in compact_question:
        sort_field = "unused_vacation"
    elif "직급" in compact_question:
        sort_field = "job_grade"

    if not sort_field:
        return None

    ascending_keywords = ["최저", "낮은", "적은", "짧은"]
    order = "asc" if any(keyword in compact_question for keyword in ascending_keywords) else "desc"

    return {
        "field": sort_field,
        "order": order,
        "limit": 1,
    }




def normalize_target_fields_by_question(
    target_fields: list[str],
    question: str,
) -> list[str]:
    """
    LLM이 비슷한 식별자 필드를 헷갈린 경우 질문 원문 기준으로 보정한다.

    예:
    - "주민번호 알려줘"를 employee_id로 잘못 분석하면 rrn으로 바꾼다.
    - "사번 알려줘"는 employee_id 그대로 둔다.
    """
    #질문 원문에서 공백 제거한 버전을 만든다.
    compact_question = compact_text(question)
    normalized_fields = [
        field
        for field in target_fields
        if field != "unknown"
    ]
    explicit_evaluation_fields = [
        field
        for field in normalized_fields
        if re.fullmatch(r"evaluation_\d{4}", str(field))
    ]
    evaluation_field = (
        explicit_evaluation_fields[0]
        if explicit_evaluation_fields
        else get_evaluation_field_from_question(compact_question)
    )
    detected_fields = []

    keyword_to_field = [
        (["주민등록번호", "주민번호"], "rrn"),
        (["사원번호", "사번"], "employee_id"),
        (["이름", "성명"], "employee"),
        (["이메일", "메일"], "email"),
        (["전화번호", "연락처", "휴대폰", "핸드폰"], "phone"),
        (["주소"], "address"),
        (["연봉", "급여"], "salary"),
        (["계좌번호", "계좌"], "account"),
        (["부서"], "department"),
        (["팀"], "team"),
        (["직급"], "job_grade"),
        (["직책"], "position"),
        (["입사일"], "hire_date"),
        (["계약형태", "계약직", "정규직"], "contract_type"),
        (["성과점수"], "performance_score"),
        (["평가", "고과", "인사고과"], evaluation_field),
        (["토익점수", "토익", "TOEIC점수", "TOEIC"], "toeic"),
        (["징계이력", "징계"], "disciplinary_history"),
    ]

    field_positions = []

    for keywords, field in keyword_to_field:
        positions = [
            compact_question.find(keyword)
            for keyword in keywords
            if keyword in compact_question
        ]

        if positions:
            field_positions.append(
                {
                    "field": field,
                    "position": min(positions),
                }
            )

    detected_fields = unique_keep_order(
        item["field"]
        for item in sorted(field_positions, key=lambda item: item["position"])
    )

    if detected_fields:
        normalized_fields = detected_fields
    else:
        normalized_fields = [
            "employee" if field == "employee_name" else field
            for field in normalized_fields
        ]

    normalized_fields = unique_keep_order(normalized_fields)

    # 사용자가 입력한 주민등록번호(rrn)를 물어본 것을 LLM이 employee_id로 잘못 분석하는 경우 rrnㅇ로 보정한다.
    if "주민등록번호" in compact_question or "주민번호" in compact_question:
        rrn_fields = []

        for field in normalized_fields:
            # LLM이 주민번호 질문을 사번/이름/unknown으로 잘못 잡는 경우 rrn으로 보정한다.
            if field in {"employee_id", "employee_name", "unknown"}:
                field = "rrn"

            if field not in rrn_fields:
                rrn_fields.append(field)

        if "rrn" not in rrn_fields:
            rrn_fields.insert(0, "rrn")

        return rrn_fields

    if (
        "성적" in compact_question
        or "성과" in compact_question
        or "성과점수" in compact_question
    ):
        if "performance_score" not in normalized_fields:
            normalized_fields.append("performance_score")

        return normalized_fields

    # 주민등록번호 관련 질문이 아니면
    # 질문 원문 기준으로 보정한 target_fields를 반환한다.
    return normalized_fields or target_fields

def normalize_employee_identity_fields(task: dict) -> tuple[str | None, str | None]:
    """
    LLM이 직원 이름을 employee_id에 넣은 경우를 보정한다.

    employee_id는 EMP0001 같은 실제 사번만 허용한다.
    그 외 값은 employee_name으로 옮긴다.
    """

    employee_name = task.get("employee_name")
    employee_id = task.get("employee_id")

    if employee_name is not None:
        employee_name_text = str(employee_name).strip()

        if employee_name_text.lower() in {"true", "false", "null", "none"}:
            employee_name = None
        else:
            employee_name = employee_name_text

    if employee_id:
        employee_id_text = str(employee_id).strip()

        if re.fullmatch(r"EMP\d+", employee_id_text.upper()):
            return employee_name, employee_id_text.upper()

        if not employee_name:
            employee_name = employee_id_text

        employee_id = None

    return employee_name, employee_id


def normalize_filters(raw_filters) -> list[dict]:
    """
    LLM이 반환한 filters를 안전한 형태로 정리하는 함수.

    filters는 검색 조건을 의미한다.

    예:
    "채용팀 직원 알려줘"
    -> team이 채용팀인 직원만 찾기 위해 filter가 필요하다.

    최종적으로 이 함수는 아래처럼 안전한 리스트 형태를 반환한다.

    [
        {"field": "team", "op": "eq", "value": "채용팀"},
        {"field": "contract_type", "op": "eq", "value": "계약직"}
    ]

    중요:
    - LLM이 만든 filters를 그대로 믿지 않는다.
    - 허용된 field인지 검사한다.
    - 허용된 op인지 검사한다.
    - value가 비어 있으면 제거한다.
    - 부서/팀/직책 값이 실제 목록에 있는지도 검사한다.
    """

    # 최종적으로 정리된 필터들을 담을 리스트
    normalized_filters = []

    # =========================
    # 1. dict 형태 filters 변환
    # =========================

    # 원래 filters는 list 형태가 표준이다.
    #
    # 표준 형태:
    # [
    #     {"field": "team", "op": "eq", "value": "채용팀"}
    # ]
    #
    # 그런데 LLM이나 예전 코드가 아래처럼 dict 형태로 줄 수도 있다.
    #
    # {
    #     "team": "채용팀",
    #     "contract_type": "계약직"
    # }
    #
    # 이 경우 뒤쪽 코드에서 처리하기 쉽도록 list 형태로 변환한다.
    if isinstance(raw_filters, dict):
        raw_filters = [
            {
                # dict의 key를 field로 사용한다.
                # 예: "team"
                "field": field,

                # dict 형태에는 op 정보가 없으므로 기본값 eq를 넣는다.
                # eq는 "같다"라는 뜻이다.
                # 예: team == "채용팀"
                "op": "eq",

                # dict의 value를 검색 기준 값으로 사용한다.
                # 예: "채용팀"
                "value": value,
            }
            for field, value in raw_filters.items()
        ]

    # =========================
    # 2. filters 타입 검사
    # =========================

    # dict 변환까지 했는데도 list가 아니면 잘못된 값이다.
    # 이런 값은 처리할 수 없으므로 빈 리스트를 반환한다.
    if not isinstance(raw_filters, list):
        return []

    # =========================
    # 3. filter 하나씩 검사
    # =========================

    for item in raw_filters:
        # filter 하나는 반드시 dict여야 한다.
        if not isinstance(item, dict):
            continue

        # 어떤 필드를 조건으로 볼지 꺼낸다.
        # 예: "team", "department", "position"
        field = item.get("field")

        # 어떤 방식으로 비교할지 꺼낸다.
        # 없으면 기본값으로 eq를 사용한다.
        #
        # op 예:
        # eq  -> 같다
        # gt  -> 크다
        # gte -> 크거나 같다
        # lt  -> 작다
        # lte -> 작거나 같다
        op = item.get("op", "eq")

        # 비교할 기준 값을 꺼낸다.
        # 예: "채용팀", "마케팅부", "대리"
        value = item.get("value")

        # =========================
        # 4. field 검증
        # =========================

        # field가 허용된 필드 중 하나인지 검사한다.
        if field not in FIELD_RULES:
            continue

        if field == "employee":
            continue

        # =========================
        # 5. op 검증
        # =========================

        # op가 허용된 연산자가 아니면 기본값 eq로 바꾼다.
        if op not in ALLOWED_FILTER_OPS:
            op = "eq"

        # =========================
        # 6. value 검증
        # =========================

        # value가 없으면 검색 조건으로 사용할 수 없다.
        # 뭐와 같아야 하는지 기준값이 없으면 제거한다.
        if value is None or value == "":
            continue

        if isinstance(value, str) and value.strip().upper() == "NULL":
            continue

        if is_self_placeholder_filter_value(field, value):
            continue

        # =========================
        # 7. 이름/부서/팀/직급/직책 값 검증
        # =========================

        # employee_name 필터에 직급/조직 값이 들어오면 이름 조건으로 쓰지 않는다.
        if field == "employee_name" and value in DEPARTMENTS + TEAMS + JOB_GRADES + POSITIONS:
            continue

        # department 필터라면 value가 실제 부서 목록에 있어야 한다.
        if field == "department" and value not in DEPARTMENTS:
            continue

        # team 필터라면 value가 실제 팀 목록에 있어야 한다.
        if field == "team" and value not in TEAMS:
            continue

        # job_grade 필터라면 value가 실제 직급 목록에 있어야 한다.
        if field == "job_grade" and value not in JOB_GRADES:
            continue

        # position 필터라면 value가 실제 직책 목록에 있어야 한다.
        if field == "position" and value not in POSITIONS:
            continue

        # =========================
        # 8. 안전한 filter만 최종 리스트에 추가
        # =========================

        # 위 검사를 모두 통과한 filter만 normalized_filters에 넣는다.
        normalized_filters.append(
            {
                "field": field,
                "op": op,
                "value": value,
            }
        )

    # 최종적으로 안전하게 정리된 filters 반환
    return normalized_filters


def analyze_question_to_tasks(question: str, candidates: dict | None = None) -> dict:
    field_schema_text = build_field_schema_text()
    candidate_context_text = format_candidates_for_prompt(candidates)


    prompt = f"""
        너는 한국어 HR RAG API의 질문 분석기다.
        JSON을 만들지 말고, 아래 key=value 10줄만 반환해라.
        마크다운, 설명, 코드블록, 중괄호, 대괄호를 절대 쓰지 마라.

        반환 형식:
        intent=
        target_fields=
        employee_name=
        employee_id=
        department=
        team=
        job_grade=
        position=
        filters=
        is_self=

        사용 가능한 intent:
        single_lookup, employee_list, employee_count, category_list, condition_search, unknown

        사용 가능한 fields:
        {field_schema_text}

        사전/캐시 기반으로 질문에서 미리 감지한 후보:
        {candidate_context_text}

        핵심 원칙:
        - target_fields는 사용자가 답변으로 보고 싶은 필드다. 사용자가 답변으로 보고 싶어 하는 정보를 target_fields로 정한다.
        - filters는 검색 대상을 줄이기 위한 조건이다.
        - 필드 이름만 언급되면 target_fields에 넣고 filters에는 넣지 않는다.
        - 필드 값이 조건으로 언급된 경우에만 filters에 넣는다.
        - 사용자 질문 원문과 후보에 없는 값을 새로 만들지 않는다.
        - 예시 안의 값은 참고용이며, 사용자 질문에 없는 값을 복사하지 않는다.
        - 권한 판단은 하지 않는다.
        - 사용 가능한 fields만 사용한다.
        - 확실하지 않으면 target_fields=unknown 으로 둔다.
        

        후보 사용 규칙:
        - 후보는 최종 정답이 아니라 intent/slot 판단을 돕는 힌트다.
        - employee_name 후보가 사용자 질문에 있으면 employee_name으로 우선 고려한다.
        - department/team/job_grade/position 후보가 있으면 해당 slot과 filters 판단에 참고한다.
        - business/domain term 후보는 기본적으로 target_fields 판단에만 참고한다.
        - business/domain term만으로 filters를 만들지 않는다.
        - unknown organization candidates는 등록되지 않은 조직명이다. 이를 부분 단어로 나누어 department/team으로 만들지 않는다.

        
        기본 출력 규칙:
        - 값이 없으면 NULL을 쓴다.
        - filters는 사용자 질문에 검색 대상을 제한하는 조건값이 명확히 있을 때만 만든다.
        - 같은 field라도 문맥에 따라 target_fields가 될 수도 있고 filters가 될 수도 있다.
        - target_fields가 여러 개면 쉼표로 구분한다.
        - filters가 없으면 NULL을 쓴다.
        - filters 형식은 field|op|value 이다.
        - filters가 여러 개면 세미콜론(;)으로 구분한다.
        - filter op는 eq, contains, gt, gte, lt, lte, between, exists 중 하나만 사용한다.
        - is_self는 true 또는 false만 쓴다.
        - filters는 사용자가 질문에서 명확히 말한 조건만 넣는다. 절대 추측해서 넣지 않는다
        - 확실하지 않으면 target_fields=unknown 으로 둔다.
        - 권한 판단은 하지 않는다.
        - filters는 검색 조건이며 답변 필드가 아니다.

        target_fields / filters 구분:
        - 예) "입사일 알려줘" → target_fields=hire_date, filters=NULL
        - 예) "부서 알려줘" → target_fields=department, filters=NULL
        - 예) "인사부 직원 알려줘" → target_fields=employee, department=인사부, filters=department|eq|인사부
        - 예) "2024년에 입사한 사람" → target_fields=employee, filters=hire_date|contains|2024
        - 예) "계약직 직원" → target_fields=employee, filters=contract_type|eq|계약직


        본인 질문 규칙:
        - "나", "내", "본인", "내이름", "me", "my"를 의미하면 is_self=true.
        - 본인 질문이면 다른 사람 이름이 명확히 나온 경우가 아니면 employee_name=NULL.
        - employee_name에 "나", "내", "본인", "내이름"을 넣지 마라.
        - is_self=true이면 filters는 반드시 NULL로 둔다. "내 이메일", "내 연봉", "본인 주소" 같은 표현을 검색 조건 값으로 만들지 마라.

        이름 판단 규칙:
        - employee_name은 실제 사람 이름이 명확히 언급된 경우에만 넣는다.
        - 사람 이름이 확실하지 않으면 employee_name=NULL.
        - employee_name에는 true, false 같은 boolean 값을 절대 넣지 마라.
        - 본인 질문이면 employee_name=NULL, is_self=true로만 표시한다.
        - 업무명, 부서명, 팀명, 직책명, 직급명, 고용형태, 조건 표현은 employee_name이 아니다.
        - "인사업무", "영업업무", "마케팅업무", "담당자", "계약직", "정규직", "입사자", "퇴직자", "팀장", "부서장", "직원", "사람"은 employee_name이 아니다.
        - "업무 담당자 누구야?"는 특정 사람 이름 조회가 아니라 조건에 맞는 직원 조회다.

        intent 판단:
        - 특정 직원 1명의 정보를 묻는 질문이면 single_lookup.
        - 조직/직책/조건에 맞는 직원 목록이면 employee_list 또는 condition_search.
        - 조건 필터가 필요한 직원 조회는 condition_search.
        - "몇 명", "몇명", "인원수", "인원", "명수"가 있으면 employee_count, target_fields=employee.
        - "5명만", "5개만", "상위 5명"처럼 개수를 제한하는 표현은 employee_name이나 filters로 만들지 마라.
        - "성적 좋은 사람", "성과 좋은 사람"은 performance_score가 높은 순서로 정렬하는 뜻이다. 점수 기준이 명시되지 않았으면 performance_score|gte 같은 필터를 만들지 마라.
        - 부서/팀/직급/직책 종류나 목록을 물으면 category_list.

        Field hints:
        - 이름, 성명 -> employee_name
        - 사번, 직원번호 -> employee_id
        - 부서 -> department
        - 팀 -> team
        - 직급 -> job_grade
        - 직책, 포지션 -> position
        - 이메일, 메일 -> email
        - 전화번호, 연락처, 휴대폰 -> phone
        - 주소 -> address
        - 연봉, 급여 -> salary
        - 계좌번호, 계좌 -> account
        - 입사일 -> hire_date
        - 퇴직일, 퇴직일자 -> retirement_date
        - 계약직, 정규직, 계약형태 -> contract_type
        - 성과점수 -> performance_score
        - 평가, 고과, 인사평가 -> evaluation_2024
        - 토익점수, 토익, TOEIC점수, TOEIC -> toeic
        - 징계이력, 징계 -> disciplinary_history
        - 주민등록번호, 주민번호 -> rrn

        target_fields 최종 점검:
        - 최종 반환 전에 사용자 질문 원문을 다시 읽고, 답변으로 보고 싶어 하는 필드가 target_fields에서 빠졌는지 반드시 점검한다.
        - 질문에 "이름", "부서", "직책", "직급", "이메일", "입사일", "팀"처럼 여러 필드명이 나열되면 나열된 모든 필드를 target_fields에 포함한다.
        - 본인 질문이라도 target_fields를 줄이지 마라. 본인 여부는 is_self로만 표시한다.
        - 필드명으로 쓰인 부서/팀/직급/직책은 filters로 만들지 말고 target_fields에 넣는다. 단, "개발부", "인프라팀", "대리", "팀장"처럼 구체적인 값이 조건으로 쓰인 경우에만 filters를 만든다.
        - 질문에 없는 내용은 추가하지 않는다.
        - "2023년 평가 정보가 있는 직원의 2024년 평가"처럼 조건 연도와 답변 연도가 다르면, filters에는 evaluation_2023|exists|true를 넣고 target_fields에는 evaluation_2024를 넣는다.
        - "2023년 평가 정보가 있는 직원 찾아줘"처럼 답변 연도가 따로 없으면 target_fields에는 employee,evaluation_2023을 넣는다.

        본인 복수 필드 조회 예시:
        질문: 내 이름 부서 직책 직급 이메일 입사일 팀 알려줘
        intent=single_lookup
        target_fields=employee,department,position,job_grade,email,hire_date,team
        employee_name=NULL
        employee_id=NULL
        department=NULL
        team=NULL
        job_grade=NULL
        position=NULL
        filters=NULL
        is_self=true

        조건 예시:
        - 2024년에 입사한 사람 -> filters=hire_date|contains|2024
        - 2024년에 퇴직한 사람 -> filters=retirement_date|contains|2024
        - 퇴사한 사람, 퇴직자 -> filters=retirement_date|exists|true
        - 계약직 직원 -> filters=contract_type|eq|계약직
        - 인사부 직원 -> department=인사부, filters=department|eq|인사부
        - 백엔드팀 직원 -> team=백엔드팀, filters=team|eq|백엔드팀
        - 대리 직원 -> job_grade=대리, filters=job_grade|eq|대리
        - 팀장 누구야 -> position=팀장, filters=position|eq|팀장
        - 인사 업무 담당자 누구야 -> department=인사부, filters=department|eq|인사부

        출력 예시:
        intent=condition_search
        target_fields=employee
        employee_name=NULL
        employee_id=NULL
        department=인사부
        team=NULL
        job_grade=NULL
        position=NULL
        filters=department|eq|인사부
        is_self=false

        날짜/연도 조건 규칙:
        - 연도, 월, 날짜가 사용자 질문 원문에 직접 나온 경우에만 날짜 filters를 만든다.
        - 질문에 연도, 월, 날짜가 없으면 날짜 filters를 절대 만들지 않는다.
        - "입사일", "퇴직일", "평가" 같은 날짜/연도 관련 필드명만 나온 경우는 filters가 아니라 target_fields로 판단한다.

        1. "2024년도", "2024년"처럼 특정 연도만 말하면 contains를 사용한다.
        예: "2024년도 입사자" → hire_date|contains|2024
        예: "2024년에 입사한 사람" → hire_date|contains|2024

        2. "2024년 이전", "2024년까지", "2024년 이하"라고 명시된 경우에만 lte를 사용한다.
        예: "2024년 이전 입사자" → hire_date|lte|2024
        예: "2024년까지 입사한 사람" → hire_date|lte|2024

        3. "2024년 이후", "2024년부터", "2024년 이상"이라고 명시된 경우에만 gte를 사용한다.
        예: "2024년 이후 입사자" → hire_date|gte|2024
        예: "2024년부터 입사한 사람" → hire_date|gte|2024

        4. "2024년 3월", "2024-03"처럼 월까지 말하면 eq를 사용한다.
        예: "2024년 3월 입사자" → hire_date|eq|2024-03

        5. "2024년 3월 15일", "2024-03-15"처럼 정확한 날짜를 말하면 eq를 사용한다.
        예: "2024년 3월 15일 입사자" → hire_date|eq|2024-03-15

        주의:
        - 사용자가 "이전", "까지", "이하"라고 말하지 않았으면 lte를 쓰지 않는다.
        - 사용자가 "이후", "부터", "이상"이라고 말하지 않았으면 gte를 쓰지 않는다.
        - 단순히 "2024년도"라고만 말하면 반드시 contains를 사용한다.

        최종 출력 전 검증:
        - 사용자 질문 원문에 없는 숫자, 연도, 날짜, 이름, 부서명, 팀명, 직급명, 직책명을 새로 만들지 않는다.
        - 사용자 질문 원문에 없는 값은 filters에 넣지 않는다.
        - 후보에 있는 값이라도 사용자 질문의 의미와 맞지 않으면 filters에 넣지 않는다.
        - 사용자가 어떤 정보를 "알려줘", "조회해줘", "뭐야"라고 물은 경우 그 정보는 filters가 아니라 target_fields다.
        - 사용자가 어떤 값에 해당하는 직원을 찾는 경우에만 filters를 만든다.
        - is_self=true이고 특정 직원 1명의 정보를 조회하는 질문이면 filters=NULL이다.
        - employee_id가 명확해서 특정 직원 1명이 확정된 질문이면 filters=NULL이다.
        - 예시 안에 나온 값은 참고용일 뿐이며, 사용자 질문에 없는 값을 출력에 복사하지 않는다.

    사용자 질문:
    {question}
    """


    try:
        raw_text = call_llm_completion(
            prompt=prompt,
            temperature=0,
            max_tokens=400,
            timeout=180,
        ).strip()
        print("[DEBUG] question analysis raw:", raw_text)

        # Gemma/Ollama 호출로 되돌릴 때 참고용으로 남겨둔다.
        # response = requests.post(
        #     OLLAMA_URL,
        #     json={
        #         "model": MODEL_NAME,
        #         "prompt": prompt,
        #         "stream": False,
        #         "options": {
        #             "temperature": 0,
        #             "num_predict": 400,
        #         },
        #     },
        #     timeout=180,
        # )
        # response.raise_for_status()
        # raw_text = response.json().get("response", "").strip()


        try:
            analysis = parse_key_value_analysis(raw_text)
            print(
                "[DEBUG] question analysis parsed:",
                json.dumps(analysis, ensure_ascii=False),
            )
            return analysis

        except Exception as e:
            print("[ERROR] 질문 분석 결과 key=value 변환 실패")
            print("[DEBUG] error:", e)
            print("[DEBUG] raw_text:", raw_text)
            return build_fallback_analysis(question)

    except requests.exceptions.Timeout:
        print("[ERROR] 질문 분석 LLM 요청 시간 초과")
        fallback = build_fallback_analysis(question)
        print(
            "[DEBUG] question analysis fallback:",
            json.dumps(fallback, ensure_ascii=False),
        )
        return fallback

    except (requests.exceptions.RequestException, ValueError) as e:
        print("[ERROR] 질문 분석 LLM 요청 실패:", e)
        fallback = build_fallback_analysis(question)
        print(
            "[DEBUG] question analysis fallback:",
            json.dumps(fallback, ensure_ascii=False),
        )
        return fallback


def normalize_tasks(
    analysis: dict,
    question: str = "",
    candidates: dict | None = None,
) -> list[dict]:
    """
    LLM 분석 결과를 코드에서 안전하게 보정하는 함수.

    LLM이 반환한 tasks를 그대로 사용하면 위험할 수 있다.
    예를 들어 LLM이 없는 필드명을 만들거나,
    잘못된 intent를 만들거나,
    부서/팀/직급을 제대로 못 잡을 수 있다.

    그래서 이 함수에서 하는 일은 다음과 같다.

    1. analysis 안에서 tasks 목록을 꺼낸다.
    2. 질문 문장에 실제로 들어있는 부서/팀/직급을 코드로 다시 찾는다.
    3. intent가 허용된 값인지 확인한다.
    4. target_fields가 허용된 필드인지 확인한다.
    5. filters 형식을 안전하게 정리한다.
    6. 부서/팀/직급 조건을 filters에 추가한다.
    7. category_list / employee_list / condition_search 같은 intent를 보정한다.
    8. 최종적으로 안전한 task 목록을 반환한다.

    중요:
    - 여기서는 권한 판단을 하지 않는다.
    - 권한 판단은 task_processor_service.py 또는 common.hr_fields 쪽에서 한다.
    - 이 함수는 질문 분석 결과를 "정리/보정"하는 역할만 한다.
    """

    # LLM 분석 결과에서 tasks 배열을 꺼낸다.
    # analysis 예시:
    # {
    #     "tasks": [
    #         {
    #             "intent": "employee_list",
    #             "target_fields": ["employee"],
    #             "department": "마케팅부"
    #         }
    #     ]
    # }
    tasks = analysis.get("tasks", [])

    # tasks가 리스트가 아니거나 비어 있으면
    # 아래 for문에서 처리할 수 없으므로 빈 리스트로 초기화한다.
    if (not isinstance(tasks, list)) or (not tasks):
        tasks = []

    # =========================
    # 1. 질문 문장에서 조직 정보 직접 탐색
    # =========================

    # LLM이 department/team/position을 못 잡을 수도 있기 때문에
    # 코드에서 질문 문장에 실제 부서명이 있는지 다시 찾는다.
    found_department = find_known_value(question, DEPARTMENTS)

    # 질문 문장에 실제 팀명이 있는지 찾는다.
    found_team = find_known_value(question, TEAMS)

    # 질문 문장에 실제 직급명이 있는지 찾는다.
    found_job_grade = find_known_value(question, JOB_GRADES)

    # 질문 문장에 실제 직책명이 있는지 찾는다.
    found_position = find_known_value(question, POSITIONS)

    # 사용자가 정확한 부서명/팀명이 아니라 별칭처럼 물어볼 수 있다.
    # 이런 표현을 실제 department 또는 team 값으로 바꿔줄 수 있는지 찾는다.
    unknown_org_candidates = (candidates or {}).get("unknown_orgs") or []
    found_org_alias = None if unknown_org_candidates else find_org_alias(question)

    # 별칭이 발견되었고,
    # 아직 정확한 부서/팀을 못 찾은 경우에만 별칭 결과를 사용한다.
    #
    # 즉, 정확한 부서명이 이미 있으면 그 값을 우선한다.
    if found_org_alias and (not found_department) and (not found_team):
        alias_field, alias_value = found_org_alias

        # 별칭이 department로 해석되면 found_department에 넣는다.
        if alias_field == "department":
            found_department = alias_value

        # 별칭이 team으로 해석되면 found_team에 넣는다.
        if alias_field == "team":
            found_team = alias_value

    # =========================
    # 2. 목록 질문인지 판단
    # =========================

    # 질문에서 공백 등을 제거해 비교하기 쉽게 만든다.
    compact_question = compact_text(question)

    # 사용자가 "목록 자체"를 물어볼 때 쓰는 키워드들
    list_keywords = [
        "종류",
        "목록",
        "리스트",
        "전체부서",
        "전체팀",
        "전체직급",
        "전체직책",
        "어떤부서",
        "어떤팀",
        "부서뭐",
        "팀뭐",
        "직급뭐",
        "직책뭐",
    ]

    # 질문 안에 목록형 키워드가 하나라도 있으면
    # category_list 요청으로 볼 가능성이 있다.
    requested_category_list = any(
        keyword in compact_question
        for keyword in list_keywords
    )

    count_keywords = [
        "몇명",
        "인원수",
        "인원",
        "명수",
        "총몇명",
        "몇명이야",
        "몇명인지",
        "몇명돼",
    ]

    requested_employee_count = any(
        keyword in compact_question
        for keyword in count_keywords
    )
    requested_employee_collection = is_employee_collection_query(question)
    question_is_self = is_self_question(question)

    # 최종적으로 정리된 task들을 담을 리스트
    normalized_tasks = []

    # =========================
    # 3. LLM이 만든 task 하나씩 검사
    # =========================

    for task in tasks:
        # task는 dict 형태여야 한다.
        # dict가 아니면 잘못된 값이므로 건너뛴다.
        if not isinstance(task, dict):
            continue

        # -------------------------
        # 3-1. intent 안전성 검사
        # -------------------------

        # LLM이 추출한 intent를 가져온다.
        # 없으면 unknown으로 둔다.
        intent = task.get("intent", "unknown")

        # intent가 허용된 목록에 없으면 unknown으로 보정한다.
        if intent not in ALLOWED_INTENTS:
            intent = "unknown"

        # -------------------------
        # 3-2. target_fields 안전성 검사
        # -------------------------

        # LLM이 추출한 target_fields를 가져온다.
        # 없으면 ["unknown"]으로 둔다.
        target_fields = task.get("target_fields", ["unknown"])

        # target_fields는 반드시 리스트여야 한다.
        # 문자열이나 다른 타입이면 안전하게 unknown으로 바꾼다.
        if not isinstance(target_fields, list):
            target_fields = ["unknown"]

        # 허용된 필드만 남긴다.
        safe_fields = [
            field
            for field in target_fields
            if field in ALLOWED_FIELDS
        ]

        # 허용된 필드가 하나도 없으면 unknown으로 둔다.
        if not safe_fields:
            safe_fields = ["unknown"]

        # 질문 문장을 기준으로 target_fields를 한 번 더 보정한다.
        safe_fields = normalize_target_fields_by_question(
            target_fields=safe_fields,
            question=question,
        )

        basic_profile_question = any(
            keyword in compact_text(question)
            for keyword in [
                "기본정보",
                "인사정보",
                "기본인사정보",
                "인사프로필",
            ]
        )

        if basic_profile_question:
            safe_fields = unique_keep_order(
                safe_fields + ["employee", "department", "team", "job_grade", "email"]
            )
            if not task.get("employee_name") and not task.get("employee_id"):
                compact_question = compact_text(question)
                guessed_name = None

                for keyword in [
                    "기본인사정보",
                    "기본정보",
                    "인사정보",
                    "인사프로필",
                ]:
                    if keyword in compact_question:
                        prefix = compact_question.split(keyword, 1)[0]
                        prefix = re.sub(r"(님|씨|은|는|이|가|을|를|도|만|요)+$", "", prefix)
                        if re.fullmatch(r"[가-힣]{2,4}", prefix):
                            guessed_name = prefix
                        break

                if not guessed_name:
                    guessed_name = extract_employee_name(question)

                if guessed_name:
                    employee_name = guessed_name
            intent = "single_lookup"



        # -------------------------
        # 3-3. 부서/팀/직급/직책 값 보정
        # -------------------------

        # LLM이 추출한 부서/팀/직급/직책 값을 가져온다.
        raw_department = task.get("department")
        raw_team = task.get("team")
        raw_job_grade = task.get("job_grade")
        raw_position = task.get("position")

        # LLM이 준 department가 실제 DEPARTMENTS 목록에 있으면 사용한다.
        # 아니면 질문 문장에서 코드로 찾은 found_department를 사용한다.
        department = (
            raw_department
            if raw_department in DEPARTMENTS and value_appears_in_question(question, raw_department)
            else found_department
        )

        # LLM이 준 team이 실제 TEAMS 목록에 있으면 사용한다.
        # 아니면 질문 문장에서 코드로 찾은 found_team을 사용한다.
        team = (
            raw_team
            if raw_team in TEAMS and value_appears_in_question(question, raw_team)
            else found_team
        )
        # LLM이 준 job_grade가 실제 JOB_GRADES 목록에 있으면 사용한다.
        # 아니면 질문 문장에서 코드로 찾은 found_job_grade를 사용한다.
        job_grade = (
            raw_job_grade
            if raw_job_grade in JOB_GRADES and value_appears_in_question(question, raw_job_grade)
            else found_job_grade
        )

        # LLM이 준 position이 실제 POSITIONS 목록에 있으면 사용한다.
        # 아니면 질문 문장에서 코드로 찾은 found_position을 사용한다.
        position = (
            raw_position
            if raw_position in POSITIONS and value_appears_in_question(question, raw_position)
            else found_position
        )

        if "position" in safe_fields and not position and "직책" not in compact_question:
            safe_fields = [
                field
                for field in safe_fields
                if field != "position"
            ]

            if not safe_fields:
                safe_fields = ["employee"]

        # -------------------------
        # 3-4. filters 정규화
        # -------------------------

        # LLM이 만든 filters를 안전한 형식으로 정리한다.
        #
        # filters 예시:
        # [
        #     {"field": "department", "op": "eq", "value": "마케팅부"}
        # ]
        filters = normalize_filters(task.get("filters", []))

        # 질문 의미를 보고 filters를 한 번 더 보정한다.
        #
        # 예:
        # "성과 좋은 직원" 같은 표현이 있으면
        # performance 관련 조건으로 보정할 수 있다.
        filters = normalize_semantic_filters(filters, question)
        filters = normalize_employee_date_year_filter(filters, question)
        filters = normalize_retired_employee_filter(filters, question)
        filters = normalize_numeric_threshold_filters(filters, question)
        sort = infer_sort_from_question(question)

        # 질문에서 부서가 확인되었으면 filters에 department 조건을 추가한다.
        #
        # 이미 같은 조건이 있으면 중복 추가하지 않는다.
        if department:
            filters = add_filter_if_missing(
                filters=filters,
                field="department",
                op="eq",
                value=department,
            )

        # 질문에서 팀이 확인되었으면 filters에 team 조건을 추가한다.
        if team:
            filters = add_filter_if_missing(
                filters=filters,
                field="team",
                op="eq",
                value=team,
            )

        # 질문에서 직급이 확인되었고,
        # 특정 직원 이름을 묻는 질문이 아니면 job_grade 조건을 추가한다.
        #
        # 예:
        # "대리 직원 알려줘" -> job_grade 필터 필요
        #
        # 반대로
        # "김민수 대리의 부서 알려줘"처럼 employee_name이 있으면
        # job_grade를 조건으로 강하게 걸지 않는다.
        if job_grade and not task.get("employee_name"):
            filters = add_filter_if_missing(
                filters=filters,
                field="job_grade",
                op="eq",
                value=job_grade,
            )

        # 질문에서 직책이 확인되었고,
        # 특정 직원 이름을 묻는 질문이 아니면 position 조건을 추가한다.
        #
        # 예:
        # "팀장 직원 알려줘" -> position 필터 필요
        #
        # 반대로
        # "김민수 팀장의 부서 알려줘"처럼 employee_name이 있으면
        # position을 조건으로 강하게 걸지 않는다.
        if position and not task.get("employee_name"):
            filters = add_filter_if_missing(
                filters=filters,
                field="position",
                op="eq",
                value=position,
            )

        # =========================
        # 4. intent 보정
        # =========================

        # "인사부 알려줘" 같은 질문은 주의해야 한다.
        #
        # LLM은 "인사부"라는 부서명을 보고
        # category_list로 잘못 판단할 수 있다.
        #
        # 하지만 사용자가 "부서 목록"을 물어본 게 아니라
        # 특정 부서의 직원을 알려달라는 의미일 수 있다.
        #
        # 그래서 실제 부서/팀/직급이 있고,
        # "종류/목록/전체부서" 같은 목록형 키워드가 없으면
        # category_list를 employee_list로 보정한다.
        if (
            intent == "category_list"
            and not requested_category_list
            and (department or team or position)
        ):
            intent = "employee_list"
            safe_fields = ["employee"]

        # 직원 이름이나 사번이 명시된 필드 조회를 LLM이 category_list로
        # 잘못 분류한 경우 특정 직원 단일 조회로 되돌린다.
        if (
            intent == "category_list"
            and not requested_category_list
            and safe_fields != ["unknown"]
            and (
                task.get("employee_name")
                or task.get("employee_id")
                or extract_employee_name(question)
                or extract_employee_id(question)
            )
        ):
            intent = "single_lookup"

        # filters가 있는데 intent가 unknown이면
        # 조건 검색으로 볼 수 있으므로 condition_search로 보정한다.
        #
        # 예:
        # "마케팅부 대리 알려줘"
        # -> department, position 필터가 있으므로 조건 검색
        if filters and intent == "unknown":
            intent = "condition_search"

        if filters and intent == "employee_list":
            intent = "condition_search"

        # filters는 있는데 target_fields를 못 잡은 경우
        # 최소한 직원 목록을 찾는 질문으로 보고 employee 필드를 사용한다.
        #
        # 예:
        # "마케팅부 알려줘"
        # 필터: department=마케팅부
        # target_fields: unknown
        # -> employee 목록 조회로 보정
        if filters and safe_fields == ["unknown"]:
            safe_fields = ["employee"]

        if filters and safe_fields == ["employee_name"]:
            safe_fields = ["employee"]
        if requested_employee_count:
            intent = "employee_count"
            safe_fields = ["employee"]

        elif requested_employee_collection:
            intent = "condition_search"

            if safe_fields == ["unknown"]:
                safe_fields = ["employee"]

        if sort and safe_fields != ["unknown"] and sort["field"] not in safe_fields:
            safe_fields.append(sort["field"])

        # -------------------------
        # 5. 직원 식별 정보 정규화
        # -------------------------

        # employee_name, employee_id를 안전하게 정리한다.
        #
        # 예:
        # "emp0070" -> "EMP0070"
        # 이름이 없으면 None
        employee_name, employee_id = normalize_employee_identity_fields(task)

        if is_limit_placeholder_name(employee_name):
            employee_name = None

        has_person_target = bool(employee_id) or (
            bool(employee_name)
            and not is_self_placeholder_name(employee_name)
        )
        task_is_self = (
            bool(task.get("is_self", False))
            and not has_person_target
        ) or (
            question_is_self
            and not employee_id
            and not has_person_target
        )

        has_explicit_search_target = bool(
            employee_id
            or has_person_target
            or department
            or team
            or job_grade
            or position
            or filters
            or unknown_org_candidates
        )
        is_targetless_single_lookup = (
            intent in {"single_lookup", "condition_search", "unknown"}
            and safe_fields != ["unknown"]
            and any(field not in {"employee", "employee_name", "employee_id"} for field in safe_fields)
            and not requested_employee_collection
            and not requested_employee_count
            and not requested_category_list
            and not has_explicit_search_target
        )

        if is_targetless_single_lookup:
            intent = "single_lookup"
            task_is_self = True

        if requested_employee_collection:
            employee_name = None
            employee_id = None
            task_is_self = False

        # intent가 single_lookup이 아니어도,
        # 질문에 사람 이름이 있으면 코드에서 다시 이름을 보정한다.
        if (
            not requested_employee_collection
            and not task_is_self
            and not employee_name
            and not employee_id
        ):
            guessed_name = extract_employee_name(question)

            if guessed_name:
                employee_name = guessed_name

        # 이름이 있고, 조건 필터가 없고, 특정 필드를 물어본 경우는
        # condition_search가 아니라 특정 직원 단일 조회로 보정한다.
        if (
            employee_name
            and not requested_employee_collection
            and intent == "condition_search"
            and not filters
            and safe_fields != ["employee"]
        ):
            intent = "single_lookup"

        # LLM이 "인사부", "마케팅부" 같은 조직명을
        # employee_name으로 잘못 넣는 경우가 있다.
        #
        # 그런 경우 직원 이름이 아니므로 None으로 제거한다.
        if is_org_alias_text(employee_name):
            employee_name = None

        if is_limit_placeholder_name(employee_name):
            employee_name = None

        if (
            filters
            and intent == "single_lookup"
            and not employee_name
            and not employee_id
            and not task_is_self
        ):
            intent = "condition_search"

        if task_is_self:
            employee_name = None

            # 본인 단일 정보 조회는 requester_employee_id로 이미 대상이 정해져 있다.
            # 따라서 LLM이 잘못 만든 filters는 사용하지 않는다.
            #
            # 예:
            # - 나의 계좌번호 알려줘 -> target_fields=account, filters=[]
            # - 나의 입사일 알려줘 -> target_fields=hire_date, filters=[]
            # - 내 이메일 알려줘 -> target_fields=email, filters=[]
            if intent == "single_lookup":
                filters = []
            else:
                filters = [
                    item
                    for item in filters
                    if not is_self_placeholder_filter_value(
                        item.get("field"),
                        item.get("value"),
                    )
                ]

        # -------------------------
        # 6. 정규화된 task 추가
        # -------------------------

        # 위에서 안전하게 보정한 값들만 최종 task로 저장한다.

        normalized_tasks.append(
            {
                "intent": intent,
                "target_fields": safe_fields,
                "employee_name": employee_name,
                "employee_id": employee_id,
                "department": department,
                "team": team,
                "job_grade": job_grade,
                "position": position,
                "unknown_orgs": unknown_org_candidates,
                "filters": filters,
                "sort": sort,
                "is_self": task_is_self,
            }
        )

    # =========================
    # 7. task가 하나도 없을 때 기본 task 추가
    # =========================

    # LLM 응답이 비었거나,
    # 모든 task가 잘못된 형식이라서 건너뛰어진 경우
    # 빈 리스트를 반환하지 않고 DEFAULT_TASK를 넣어준다.
    #
    # 이렇게 해야 뒤쪽 process_task에서 최소한 unknown 질문으로 처리할 수 있다.
    if not normalized_tasks:
        normalized_tasks.append(DEFAULT_TASK.copy())

    # 최종 정규화된 task 목록 반환
    return normalized_tasks


