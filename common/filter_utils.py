from common.hr_fields import FIELD_RULES
from common.text_utils import compact_text


def unique_keep_order(items: list[str]) -> list[str]:
    """
    리스트의 기존 순서는 유지하면서 중복 값만 제거한다.
    """

    result = []

    for item in items:
        if item not in result:
            result.append(item)

    return result


def value_appears_in_question(question: str, value) -> bool:
    """
    필터 값이 실제 사용자 질문에 포함되어 있었는지 확인한다.
    LLM이 질문에 없는 조직/직급 조건을 만들어내는 경우를 걸러낼 때 쓴다.
    """

    if not question or not value:
        return False

    return compact_text(str(value)) in compact_text(question)


def add_filter_if_missing(
    filters: list[dict],
    field: str,
    value,
    op: str = "eq",
) -> list[dict]:
    """
    같은 field/value 조건이 아직 없을 때만 filters에 새 조건을 추가한다.
    """

    if not value:
        return filters

    for item in filters:
        if item.get("field") == field and item.get("value") == value:
            return filters

    filters.append(
        {
            "field": field,
            "op": op,
            "value": value,
        }
    )

    return filters


def get_filter_fields(filters: list[dict]) -> list[str]:
    """
    filters에 들어 있는 유효한 HR field 목록을 중복 없이 반환한다.
    권한 판단에 필요한 필드 목록을 만들 때 사용한다.
    """

    if not isinstance(filters, list):
        return []

    fields = []

    for item in filters:
        if not isinstance(item, dict):
            continue

        field = item.get("field")

        if field in FIELD_RULES:
            fields.append(field)

    return unique_keep_order(fields)
