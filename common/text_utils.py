import re


def compact_text(text: str) -> str:
    """
    질문이나 값 비교를 쉽게 하기 위해 모든 공백을 제거한다.
    예: "영업 관련 직원" -> "영업관련직원"
    """

    if not text:
        return ""

    return re.sub(r"\s+", "", text)


def to_none(value):
    """
    LLM이나 입력값에서 넘어온 빈 값/NULL 문자열을 Python None으로 통일한다.
    """

    if value is None:
        return None

    value = str(value).strip()

    if not value or value.upper() == "NULL":
        return None

    return value


def parse_bool(value) -> bool:
    """
    문자열 true/false 값을 bool 값으로 변환한다.
    """

    return str(value).strip().lower() == "true"
