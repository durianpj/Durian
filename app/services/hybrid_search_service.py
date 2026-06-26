import os
import re
import time
from dotenv import load_dotenv
from opensearchpy import OpenSearch
from transformers import AutoTokenizer, AutoModel
import torch

from app.services.question_service import extract_employee_name
from common.hr_fields import ACCESSIBLE_INDICES, FIELD_RULES
from common.hr_master_data import (
    DEPARTMENTS,
    TEAMS,
    JOB_GRADES,
    POSITIONS,
    TEAM_TO_DEPARTMENT,
)

DOC_TYPE_KEYWORDS = {
    "salary": [
        "연봉",
        "급여",
        "월급",
        "보상",
        "보수",
        "수당",
        "실수령",
        "계좌번호",
        "계좌",
        "은행",
    ],
    "performance": [
        "성과",
        "평가",
        "고과",
        "점수",
        "자격증",
        "토익",
        "TOEIC",
        "수상",
        "징계",
        "불이익",
        "인사조치",
    ],
    "basic": [
        "주소",
        "거주지",
        "사는 곳",
        "사는곳",
        "주민번호",
        "주민등록번호",
        "전화번호",
        "연락처",
        "휴대폰",
        "핸드폰",
        "이름",
        "성별",
        "나이",
        "생년월일",
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
        "이전직장명",
        "이전최종직급",
        "이전담당업무",
        "회사명",
        "사업장위치",
        "부서",
        "팀",
        "부서레벨",
        "직급",
        "직책",
        "직급레벨",
        "퇴직구분",
        "퇴직일자",
        "이메일",
        "메일",
        "사번",
        "사원번호",
    ],
}


# =========================
# OpenSearch 접속 설정
# =========================

# .env 파일에 있는 OpenSearch 접속 정보를 불러온다.
# 예: OPENSEARCH_HOST, OPENSEARCH_PORT, OPENSEARCH_USER, OPENSEARCH_PASSWORD
load_dotenv()

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST")
OPENSEARCH_PORT = os.getenv("OPENSEARCH_PORT")
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")


# 필수 환경변수가 하나라도 없으면 실행을 중단한다.
# OpenSearch 접속 정보가 없으면 검색 자체가 불가능하기 때문이다.
required_envs = [
    OPENSEARCH_HOST,
    OPENSEARCH_PORT,
    OPENSEARCH_USER,
    OPENSEARCH_PASSWORD,
]

if any(value is None for value in required_envs):
    raise ValueError(
        "OpenSearch 접속 환경변수가 설정되지 않았습니다. .env 파일을 확인하세요."
    )


# .env에서 가져온 값은 문자열이므로 port는 int로 변환한다.
OPENSEARCH_PORT = int(OPENSEARCH_PORT)


# Python 코드에서 OpenSearch에 검색 요청을 보내기 위한 클라이언트 객체(OpenSearch랑 Python을 연결)
client = OpenSearch(
    hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
    http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False,
)


# =========================
# 임베딩 모델 설정
# =========================

# 질문을 벡터로 바꾸기 위해 사용하는 임베딩 모델
# OpenSearch의 embedding_vector가 384차원으로 저장되어 있으므로
# 같은 모델을 사용해야 벡터 차원이 맞다.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# tokenizer: 문장을 모델이 이해할 수 있는 숫자 토큰으로 변환
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# embedding_model: 토큰을 실제 벡터로 변환하는 모델
embedding_model = AutoModel.from_pretrained(MODEL_NAME)


# =========================
# LLM 전달용 Context 생성 함수
# =========================
def extract_embedding_text_field(embedding_text: str, field_name: str) -> str:
    """
    embedding_text의 "필드명: 값" 형태에서 값을 꺼낸다.
    예:
    주소: 서울시 마포구 합정동
    퇴직일자: 2024-10-17

    field_name이 "퇴직일자"이면 "2024-10-17"을 반환한다.
    """

    if not embedding_text or not field_name:
        return ""

    for line in embedding_text.splitlines():
        line = line.strip()

        if not line.startswith(field_name):
            continue

        rest = line[len(field_name):].lstrip()

        if not rest.startswith(":"):
            continue

        return rest[1:].strip()

    # 추출 대상 필드 이름들을 저장할 집합(set)
    labels = set()

    # FIELD_RULES에 정의된 모든 규칙 순회
    for rule in FIELD_RULES.values():

        # label 값이 있으면 labels에 추가
        if rule.get("label"):
            labels.add(rule["label"])

        # embedding_label 값이 있으면 labels에 추가
        if rule.get("embedding_label"):
            labels.add(rule["embedding_label"])

    # 현재 찾고 있는 필드명은 제외
    # (예: "이름:" 값을 찾는데 다음 라벨 탐색에 "이름"이 포함되면 안 됨)
    labels.discard(field_name)

    # embedding_text 안에서
    # "필드명 :" 형태를 찾음
    #
    # 예:
    # 이름: 홍길동
    # 부서: 개발팀
    #
    # field_name = "이름" 이라면
    # "이름:" 위치를 찾음
    match = re.search(
        rf"(?:^|\s){re.escape(field_name)}\s*:\s*",
        embedding_text,
    )

    # 필드를 찾지 못한 경우 빈 문자열 반환
    if not match:
        return ""

    # 값이 시작하는 위치
    # 예: "이름: 홍길동" 에서 "홍길동" 시작 위치
    value_start = match.end()

    # 필드명 이후의 텍스트만 잘라냄
    tail = embedding_text[value_start:]

    # 기본적으로 끝까지를 값이라고 가정
    value_end = len(tail)

    # 다른 필드(label)가 나오는 위치를 찾음
    for label in sorted(labels, key=len, reverse=True):

        # 다음 필드 시작 위치 탐색
        #
        # 예:
        # 부서:
        # 직급:
        #
        next_match = re.search(
            rf"(?:^|\s){re.escape(label)}\s*:",
            tail,
        )

        # 더 가까운 다음 필드를 찾으면
        # 그 위치를 값의 끝으로 설정
        if next_match and next_match.start() < value_end:
            value_end = next_match.start()

    # 현재 필드의 값만 잘라서 반환
    #
    # 예:
    # 이름: 홍길동 부서: 개발팀
    #
    # field_name = "이름"
    # -> "홍길동"
    return tail[:value_end].strip()

    return ""

 
def build_list_summary(search_hits, question: str) -> str | None:
    # search_hits = OpenSearch에서 검색된 결과들
    # question = 사용자가 입력한 질문
    """
    부서/팀/직급/직책 종류 질문을 공통으로 처리한다.

    예:
    - 부서종류 알려줘
    - 팀종류 알려줘
    - 직급 목록 알려줘
    - 직책 리스트 알려줘
    """

    question = question.strip()

    is_list_question = any(word in question for word in ["종류", "목록", "리스트", "전체", "모두", "모든","뭐뭐","전부"])

    if not is_list_question:
        return None

    list_targets = {
        "부서": {
            "label": "부서",
            "field_key": "department",
        },
        "팀": {
            "label": "팀",
            "field_key": "team",
        },
        "직급": {
            "label": "직급",
            "field_key": "job_grade",
        },
        "직책": {
            "label": "직책",
            "field_key": "position",
        },
    }

    for keyword, config in list_targets.items():
        if keyword not in question:
            continue

        values = set()

        for hit in search_hits:
            source = hit.get("_source", {})
            embedding_text = source.get("embedding_text", "")

            value = get_allowed_field_value(
                source=source,
                embedding_text=embedding_text,
                field_key=config["field_key"],
            )

            if value:
                values.add(value)

        values = sorted(values)

        if not values:
            return (
                "[목록 질문 요약]\n"
                f"사용자는 {config['label']} 종류를 물었습니다.\n"
                f"검색 결과에서 확인 가능한 {config['label']} 정보가 없습니다."
            )

        return (
            "[목록 질문 요약]\n"
            f"사용자는 특정 직원 한 명의 정보가 아니라 {config['label']} 종류를 물었습니다.\n"
            "직원 이름은 답변하지 마세요.\n"
            f"아래 {config['label']} 값만 사용해서 답변하세요.\n\n"
            "[답변 형식]\n"
            f"검색 결과에서 확인되는 {config['label']} 종류는 다음과 같습니다.\n"
            + "\n".join([f"- {value}" for value in values])
        )

    return None

def build_full_list_answer(question: str, permission_level: int) -> str | None:
    """
    부서/팀/직급/직책 종류 질문은 검색 결과 size에 의존하지 않고,
    접근 가능한 기본정보 인덱스 전체에서 필요한 값만 모아 답변한다.
    """

    question = question.strip()

    is_list_question = any(word in question for word in ["종류", "목록", "리스트"])

    if not is_list_question:
        return None

    target = None

    if "부서" in question:
        target = "부서"
    elif "팀" in question:
        target = "팀"
    elif "직급" in question:
        target = "직급"
    elif "직책" in question:
        target = "직책"

    if not target:
        return None

    indices = get_basic_indices(permission_level)

    if not indices:
        return f"조회 가능한 {target} 정보가 없습니다."

    # 필요한 필드만 가져오기
    if target == "부서":
        query = {
            "size": 10000,
            "_source": ["department"],
            "query": {"match_all": {}},
        }

    elif target == "직급":
        query = {
            "size": 10000,
            "_source": ["job_grade"],
            "query": {"match_all": {}},
        }

    elif target == "팀":
        query = {
            "size": 10000,
            "_source": ["embedding_text"],
            "query": {
                "match_phrase": {
                    "embedding_text": "팀:"
                }
            },
        }

    elif target == "직책":
        query = {
            "size": 10000,
            "_source": ["embedding_text"],
            "query": {
                "match_phrase": {
                    "embedding_text": "직책:"
                }
            },
        }

    # 선택된 인덱스에서 query 조건에 맞는 검색 결과를 가져온다.
    response = client.search(index=indices, body=query)

    hits = response["hits"]["hits"]

    values = set()

    for hit in hits:
        source = hit.get("_source", {})
        embedding_text = source.get("embedding_text", "")

        field_key_by_label = {
            "부서": "department",
            "직급": "job_grade",
            "팀": "team",
            "직책": "position",
        }
        value = get_allowed_field_value(
            source=source,
            embedding_text=embedding_text,
            field_key=field_key_by_label.get(target, ""),
        )

        if value:
            value = value.strip()

            # "부서: 개발부" 같은 이상한 값 제거
            if ":" in value:
                continue

            # 팀 종류 질문이면 진짜 팀 이름만 남긴다.
            if target == "팀" and not value.endswith("팀"):
                continue

            # 부서 종류 질문이면 부서명만 남긴다.
            if target == "부서" and not value.endswith("부"):
                continue

            values.add(value)

    values = sorted(values)

    if not values:
        return f"조회 가능한 {target} 정보가 없습니다."

    return (
        f"현재 접근 권한 내에서 확인되는 {target} 종류는 다음과 같습니다.\n"
        + "\n".join([f"- {value}" for value in values])
    )
def get_allowed_field_value(source: dict, embedding_text: str, field_key: str):
    """
    FIELD_RULES 기준으로 허용된 필드 값만 꺼낸다.

    source_field가 있으면 _source에서 꺼내고,
    embedding_label이 있으면 embedding_text에서 해당 라벨만 추출한다.
    """

    rule = FIELD_RULES.get(field_key)

    if not rule:
        return None

    # 직원 기본 식별 정보
    if field_key == "employee":
        return {
            "employee_id": source.get("employee_id", ""),
            "employee_name": source.get("employee_name", ""),
        }

    # OpenSearch _source에 직접 있는 필드
    source_field = rule.get("source_field")

    if source_field:
        return source.get(source_field, "")

    # embedding_text 안에 들어있는 필드
    embedding_label = rule.get("embedding_label")

    if embedding_label:
        return extract_embedding_text_field(embedding_text, embedding_label)

    return None


def get_doc_type_label(index_name: str) -> str:
    """
    인덱스명 기준으로 문서 유형 라벨을 만든다.
    """

    if "salary" in index_name:
        return "급여정보"

    if "performance" in index_name:
        return "성과정보"

    return "기본정보"


def build_context(search_hits, question="", allowed_fields=None):
    """
    OpenSearch 검색 결과를 LLM에 전달할 context 문자열로 변환한다.

    보안 원칙:
    - allowed_fields에 포함된 필드만 context에 넣는다.
    - embedding_text 원문은 절대 그대로 넣지 않는다.
    - 권한 없는 필드는 LLM에게도 보여주지 않는다.
    """

    allowed_fields = set(allowed_fields or [])

    # 아무 필드도 허용되지 않으면 context를 만들지 않는다.
    if not allowed_fields:
        return ""

    context_list = []

    for hit in search_hits:
        index_name = hit["_index"]
        doc_id = hit["_id"]
        source = hit.get("_source", {})
        embedding_text = source.get("embedding_text", "")

        doc_type = get_doc_type_label(index_name)

        lines = [
            f"[출처: {index_name} / {doc_id}]",
            f"문서유형: {doc_type}",
        ]

        # 직원 식별 정보도 employee가 허용된 경우에만 넣는다.
        if "employee" in allowed_fields:
            employee_info = get_allowed_field_value(
                source=source,
                embedding_text=embedding_text,
                field_key="employee",
            )

            employee_id = employee_info.get("employee_id", "")
            employee_name = employee_info.get("employee_name", "")

            if employee_id:
                lines.append(f"사번: {employee_id}")

            if employee_name:
                lines.append(f"이름: {employee_name}")

        for field_key in allowed_fields:
            if field_key == "employee":
                continue

            rule = FIELD_RULES.get(field_key)

            if not rule:
                continue

            value = get_allowed_field_value(
                source=source,
                embedding_text=embedding_text,
                field_key=field_key,
            )
            if value:
                label = rule.get("label", field_key)
                lines.append(f"{label}: {value}")

        changed = source.get("changed")
        if changed and isinstance(changed, list):
            # change_history가 target_fields에 있으면 이력 전체를 보여준다.
            # (인덱스 레벨로 이미 권한 통제가 됨)
            # 그 외에는 allowed_fields 기준 라벨만 허용해 권한 밖 필드 이력을 숨긴다.
            show_all_history = "change_history" in (allowed_fields or [])
            allowed_labels = set()
            if not show_all_history:
                for fk in allowed_fields:
                    rule = FIELD_RULES.get(fk)
                    if rule:
                        if rule.get("label"):
                            allowed_labels.add(rule["label"])
                        if rule.get("embedding_label"):
                            allowed_labels.add(rule["embedding_label"])

            history_lines = ["변경이력:"]
            for entry in changed:
                timestamp = entry.get("timestamp", "")
                for f in entry.get("fields", []):
                    field_label = f.get("field", "")
                    if not show_all_history and allowed_labels and field_label not in allowed_labels:
                        continue
                    old_val = f.get("old", "") or "없음"
                    new_val = f.get("new", "") or "없음"
                    history_lines.append(
                        f"  {timestamp} | {field_label} | {old_val} → {new_val}"
                    )
            if len(history_lines) > 1:
                lines.extend(history_lines)

        context_list.append("\n".join(lines))

    return "\n\n".join(context_list)

# =========================
# 질문 임베딩 생성 함수
# =========================

def create_question_vector(question):
    """
    사용자 질문을 벡터 검색에 사용할 임베딩 벡터로 변환한다.

    원래 sentence_transformers를 쓰면 더 간단하지만,
    현재 PC에서 sklearn DLL 차단 문제가 있어서
    transformers + torch 방식으로 직접 임베딩을 만들었다.
    """

    # 질문 문장을 토큰 형태로 변환한다.
    inputs = tokenizer(question, padding=True, truncation=True, return_tensors="pt")

    # 검색용 임베딩 생성이므로 학습은 하지 않는다.
    # torch.no_grad()를 쓰면 메모리 사용이 줄어든다.
    # 지금은 학습 아니고 예측/변환만 할 거니까 가볍게 실행해라
    with torch.no_grad():
        outputs = embedding_model(**inputs)
        # **inputs는 딕셔너리를 함수 인자로 풀어주는 문법

    # 각 토큰별 임베딩 벡터
    token_embeddings = outputs.last_hidden_state

    # 실제 토큰과 padding 토큰을 구분하기 위한 값
    attention_mask = inputs["attention_mask"]

    # attention_mask의 차원을 token_embeddings와 맞춘다.
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )

    # 여러 토큰 벡터의 평균을 내서 문장 하나의 벡터로 만든다.
    # 이것을 mean pooling이라고 한다.
    sentence_embedding = torch.sum(
        token_embeddings * input_mask_expanded, dim=1
    ) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    # OpenSearch에 보낼 수 있도록 tensor를 Python list로 변환한다.
    vector = sentence_embedding[0].tolist()

    return vector


# =========================
# 사용자 권한 레벨 계산
# =========================

def get_user_permission_level(employee_id: str) -> int | None:
    """
    사용자 사번으로 권한 레벨을 자동 계산한다.

    현재 기준:
    - department_level = 부서 권한 레벨
    - job_grade_level = 직급 권한 레벨
    """

    # employee_id = employee_id.strip().upper()

    query = {
        "query": {
            "bool": {
                "filter": [
                    {"term": {"employee_id": employee_id}}
                ]
            }
        },
        "size": 1,
    }

    response = client.search(
        index=["hr_basic_1", "hr_basic_2", "hr_basic_3"],
        body=query,
    )

    hits = response["hits"]["hits"]
    # print("response:", response)

    if not hits:
        return None

    source = hits[0]["_source"]

    department_level = source.get("department_level", 1)
    job_grade_level = source.get("job_grade_level", 1)

    return max(department_level, job_grade_level)


def get_retired_employee_ids(employee_ids: list[str]) -> set[str]:
    """
    여러 사원의 퇴사 여부를 한 번의 OpenSearch 조회로 확인한다.

    hr_basic_3 문서 중 퇴직일자가 실제 값으로 입력된 사번만 반환한다.
    """

    normalized_ids = list(
        dict.fromkeys(
            str(employee_id).strip().upper()
            for employee_id in employee_ids
            if employee_id
        )
    )

    if not normalized_ids:
        return set()

    query = {
        "query": {
            "terms": {
                "employee_id": normalized_ids
            }
        },
        "size": min(max(len(normalized_ids) * 4, 100), 10000),
        "_source": ["employee_id", "embedding_text"],
    }

    response = client.search(
        index=["hr_basic_3"],
        body=query,
    )

    retired_employee_ids = set()
    missing_values = {"", "미입력", "NULL", "None", "none", "null", "-"}

    for hit in response["hits"]["hits"]:
        source = hit.get("_source", {})
        employee_id = source.get("employee_id")
        retirement_date = extract_embedding_text_field(
            embedding_text=source.get("embedding_text", ""),
            field_name="퇴직일자",
        )

        if (
            employee_id
            and retirement_date
            and retirement_date.strip() not in missing_values
        ):
            retired_employee_ids.add(str(employee_id).strip().upper())

    return retired_employee_ids


def is_retired_employee(employee_id: str) -> bool:
    """
    요청자 사번이 퇴사자이면 True를 반환한다.
    퇴직일자 값이 비어 있거나 '미입력'이면 재직자로 본다.
    """

    if not employee_id:
        return False

    normalized_id = employee_id.strip().upper()
    return normalized_id in get_retired_employee_ids([normalized_id])


def get_employee_ids_by_name(employee_name: str) -> list[str]:
    """
    정확히 일치하는 직원 이름의 사번 목록을 반환한다.

    퇴사자 안내를 필드 권한 안내보다 먼저 판단할 때 사용한다.
    """

    if not employee_name:
        return []

    query = {
        "query": {
            "match_phrase": {
                "employee_name": employee_name.strip()
            }
        },
        "size": 10000,
        "_source": ["employee_id"],
    }

    response = client.search(
        index=["hr_basic_1"],
        body=query,
    )

    return list(
        dict.fromkeys(
            hit.get("_source", {}).get("employee_id")
            for hit in response["hits"]["hits"]
            if hit.get("_source", {}).get("employee_id")
        )
    )


# =========================
# 질문 기반 검색 인덱스 선택 함수
# =========================

def get_doc_types_by_keywords(question: str) -> list[str]:
    """
    질문에 포함된 키워드를 기준으로 검색할 문서 유형을 판단한다.

    예:
    - 연봉, 급여 → salary
    - 성과, 평가 → performance
    - 이름, 부서, 직급 → basic
    """

    selected_doc_types = []

    for doc_type, keywords in DOC_TYPE_KEYWORDS.items():
        if any(keyword in question for keyword in keywords):
            selected_doc_types.append(doc_type)

    return selected_doc_types


def select_search_indices(question: str, permission_level: int) -> list[str]:
    """
    질문 내용과 권한 레벨을 기준으로 검색 대상 인덱스를 선택한다.

    권한 레벨로 먼저 접근 가능한 인덱스를 제한하고,
    그 안에서 질문 유형에 맞는 인덱스만 고른다.
    """

    indices = ACCESSIBLE_INDICES.get(permission_level, [])

    if not indices:
        return []

    selected_doc_types = get_doc_types_by_keywords(question)

    # 질문 유형을 못 잡으면 접근 가능한 전체 인덱스를 검색한다.
    if not selected_doc_types:
        return indices

    selected_indices = [
        index
        for index in indices
        if any(doc_type in index for doc_type in selected_doc_types)
    ]

    return selected_indices or indices


def build_field_exists_clause(rule: dict) -> dict | None:
    """
    FIELD_RULES의 source_field 또는 embedding_label 기준으로 값이 실제 입력된 문서를 찾는다.
    "미입력"은 값이 없는 상태로 본다.
    """

    source_field = rule.get("source_field")
    embedding_label = rule.get("embedding_label")

    if source_field:
        return {
            "bool": {
                "must": [
                    {"exists": {"field": source_field}},
                ],
                "must_not": [
                    {"term": {source_field: "미입력"}},
                ],
            }
        }

    if embedding_label:
        return {
            "bool": {
                "must": [
                    {
                        "match_phrase": {
                            "embedding_text": f"{embedding_label}:"
                        }
                    }
                ],
                "must_not": [
                    {
                        "match_phrase": {
                            "embedding_text": f"{embedding_label}: 미입력"
                        }
                    }
                ],
            }
        }

    return None


def build_opensearch_filter_clauses(filters: list[dict] | None) -> list[dict]:
    """
    task filters를 OpenSearch bool filter 조건으로 바꾼다.

    예:
    {"field": "retirement_date", "op": "contains", "value": "2024"}
    -> embedding_text에 "퇴직일자: 2024"가 포함된 문서 검색
    """

    if not filters:
        return []

    clauses = []

    for filter_item in filters:
        if not isinstance(filter_item, dict):
            continue

        field = filter_item.get("field")
        op = filter_item.get("op", "eq")
        value = filter_item.get("value")

        if field not in FIELD_RULES:
            continue

        if value is None or value == "":
            continue

        if isinstance(value, str) and value.strip().upper() == "NULL":
            continue

        rule = FIELD_RULES[field]

        if op == "exists":
            clause = build_field_exists_clause(rule)

            if clause:
                clauses.append(clause)

            continue

        # 우선 eq / contains만 OpenSearch 검색 조건으로 사용한다.
        # gte, lte, between 같은 숫자 비교는 기존 filter_hits_by_filters에서 후처리한다.
        if op not in {"eq", "contains"}:
            continue

        source_field = rule.get("source_field")
        embedding_label = rule.get("embedding_label")

        if source_field:
            clauses.append(
                {
                    "match_phrase": {
                        source_field: str(value)
                    }
                }
            )
            continue

        if embedding_label:
            clauses.append(
                {
                    "bool": {
                        "should": [
                            {
                                "match_phrase": {
                                    "embedding_text": f"{embedding_label}: {value}"
                                }
                            },
                            {
                                "match_phrase": {
                                    "embedding_text": str(value)
                                }
                            },
                        ],
                        "minimum_should_match": 1,
                    }
                }
            )

    return clauses







# =========================
# BM25 검색 함수
# =========================

def search_bm25(
    question,
    permission_level,
    employee_id=None,
    employee_name=None,
    extract_name=True,
    size=5,
    indices=None,
    filters=None,
):
    """
    embedding_text 필드를 대상으로 BM25 기반 키워드 검색을 수행한다.

    - employee_id가 있으면 employee_id 기준으로 검색한다.
    - employee_id가 없으면 질문 전체 + 핵심 키워드로 검색한다.
    - 이름이 있으면 employee_name 또는 embedding_text에서 이름을 찾는다.
    """

    indices = indices or select_search_indices(question, permission_level)
    filter_clauses = build_opensearch_filter_clauses(filters)
    if not indices:
        print("접근 가능한 인덱스가 없습니다.")
        return []

    # =========================
    # 1. 본인 / 특정 사번 조회
    # =========================
    if employee_id:
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"match_all": {}}
                    ],
                    "filter": [
                        {"term": {"employee_id": employee_id}},
                        *filter_clauses,
                    ],
                }
            },
            "size": size,
        }

    # =========================
    # 2. 일반 검색
    # =========================
    else:
        query = {
            "query": {
                "bool": {
                    "should": [
                        {"match": {"embedding_text": question}}
                    ],
                    "filter": filter_clauses,
                    "minimum_should_match": 1,
                }
            },
            "size": size,
        }

        # 질문 안에 들어간 핵심 필드를 따로 검색 조건에 추가
        field_keywords = [
            "부서",
            "팀",
            "직급",
            "직책",
            "이메일",
            "전화번호",
            "주소",
            "주민등록번호",
            "주민번호",
            "연봉",
            "계좌번호",
            "성과점수",
            "인사고과",
        ]

        for keyword in field_keywords:
            if keyword in question:
                query["query"]["bool"]["should"].append(
                    {"match_phrase": {"embedding_text": keyword}}
                )

                query["query"]["bool"]["should"].append(
                    {"match_phrase": {"embedding_text": f"{keyword}:"}}
                )
        if "팀" in question:
            query["query"]["bool"]["should"].append(
                {"match_phrase": {"embedding_text": "팀:"}}
            )

        is_list_question = any(word in question for word in ["종류", "목록", "리스트"])

        if is_list_question and "팀" in question:
            query["query"]["bool"]["filter"].append(
                {"match_phrase": {"embedding_text": "팀:"}}
            )

        has_department_filter = any(
            filter_item.get("field") == "department"
            for filter_item in (filters or [])
            if isinstance(filter_item, dict)
        )

        # 부서명이 직접 들어온 경우 department 필터 추가
        for department in DEPARTMENTS:
            if department in question and not has_department_filter:
                query["query"]["bool"]["filter"].append(
                    {"match_phrase": {"department": department}}
                )
                break

        # 직원 이름이 들어온 경우 employee_name 또는 embedding_text에서 검색한다.
        # task 분석에서 employee_name을 명시적으로 넘긴 경우 그 값만 사용한다.
        # condition_search 질문에서는 "계약직", "영업관련" 같은 단어가 이름으로
        # 오인될 수 있으므로 main.py에서 extract_name=False로 호출한다.
        target_name = employee_name

        if not target_name and extract_name:
            target_name = extract_employee_name(question)

        # 이름처럼 보이지만 실제로는 이름이 아닌 단어들
        not_person_names = [
            "종류",
            "목록",
            "리스트",
            "전체",
            "모두",
            "모든",
            "부서",
            "팀",
            "직급",
            "직책",
            "이메일",
            "메일",
            "사원번호",
            "사번",
            "주민등록번호",
            "주민번호",
            "연봉",
            "주소",
            "평가",
            "인사고과",
        ]

        if target_name and target_name not in not_person_names:
            query["query"]["bool"]["filter"].append(
                {
                    "bool": {
                        "should": [
                            {"match_phrase": {"employee_name": target_name}},
                            {"match_phrase": {"embedding_text": target_name}},
                            {"match_phrase": {"embedding_text": f"이름: {target_name}"}},
                        ],
                        "minimum_should_match": 1,
                    }
                }
            )
    print("[DEBUG] BM25 indices:", indices)
    print("[DEBUG] BM25 query:", query)

    response = client.search(
        index=indices,
        body=query,
    )

    hits = response["hits"]["hits"]

    print("[DEBUG] BM25 hits count:", len(hits))

    return hits
# =========================
# 벡터 검색 함수
# =========================

def cosine_similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = sum(a * a for a in vec1) ** 0.5
    norm_b = sum(b * b for b in vec2) ** 0.5
    return dot / (norm_a * norm_b + 1e-9)




def search_vector(
    question,
    question_vector,
    permission_level,
    employee_id=None,
    employee_name=None,
    extract_name=True,
    size=5,
    indices=None,
):
    """
    embedding_vector 필드를 대상으로 벡터 유사도 검색을 수행한다.

    중요:
    - employee_id가 있으면 먼저 OpenSearch에서 employee_id 필터를 건다.
    - 즉, 본인 질문이면 requester_employee_id 문서만 가져온다.
    - 그 안에서 Python으로 cosine similarity를 계산한다.
    """

    indices = indices or select_search_indices(question, permission_level)

    if not indices:
        print("접근 가능한 인덱스가 없습니다.")
        return []

    # =========================
    # 1. 본인 질문 / 특정 사번 질문
    # =========================
    # 보안상 중요:
    # 전체 벡터 검색 후 필터링하지 말고,
    # 먼저 employee_id로 검색 대상을 제한한다.
    # 사번이 있으면 이름보다 사번을 우선해서 검색 범위를 먼저 고정한다.
    if employee_id:
        employee_id = employee_id.strip().upper()

        query = {
            "size": 100,
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"employee_id": employee_id}}
                    ]
                }
            }
        }

        response = client.search(
            index=indices,
            body=query,
        )

        hits = response["hits"]["hits"]

        # employee_id로 제한된 문서들 안에서만 벡터 유사도 계산
        for hit in hits:
            source = hit.get("_source", {})
            embedding_vector = source.get("embedding_vector")

            if embedding_vector:
                hit["_score"] = cosine_similarity(question_vector, embedding_vector)
            else:
                hit["_score"] = 0

        hits = sorted(
            hits,
            key=lambda hit: hit.get("_score", 0),
            reverse=True,
        )

        return hits[:size]

    # =========================
    # 2. 일반 벡터 검색
    # =========================

    expanded_k = max(size * 10, 50)

    query = {
        "size": expanded_k,
        "query": {
            "knn": {
                "embedding_vector": {
                    "vector": question_vector,
                    "k": expanded_k,
                }
            }
        },
    }

    response = client.search(
        index=indices,
        body=query,
    )

    hits = response["hits"]["hits"]

    # employee_id가 없을 때만 이름 기반 필터를 사용한다.
    target_name = employee_name

    if not target_name and extract_name:
        target_name = extract_employee_name(question)

    # 사번이 없을 때만 이름 기반 필터를 붙인다.
    if target_name:
        hits = [
            hit
            for hit in hits
            if target_name in hit.get("_source", {}).get("employee_name", "")
            or target_name in hit.get("_source", {}).get("embedding_text", "")
        ]

    return hits[:size]

# =========================
# RRF 병합 함수
# =========================

def merge_rrf(
    bm25_hits,
    vector_hits,
    k=60,
    size=5,
    bm25_weight=1.0,
    vector_weight=1.0,
):
    """
    BM25 검색 결과와 벡터 검색 결과를 RRF 방식으로 병합한다.

    RRF는 BM25 점수와 벡터 점수를 직접 더하지 않는다.
    대신 각 검색 결과의 순위를 기준으로 최종 점수를 계산한다.

    공식:
    RRF 점수 = 가중치 * (1 / (k + 순위))

    k는 보통 60을 많이 사용한다.
    """

    merged = {}

    # BM25 검색 결과 순위를 RRF 점수로 반영한다.
    for rank, hit in enumerate(bm25_hits, start=1):
        # 같은 문서인지 구분하기 위해 인덱스명과 문서 ID를 함께 사용한다.
        doc_key = f"{hit['_index']}::{hit['_id']}"

        # 처음 나온 문서라면 저장 공간을 만든다.
        if doc_key not in merged:
            merged[doc_key] = {
                "hit": hit,
                "rrf_score": 0,
            }

        # 순위가 높을수록 더 큰 점수를 받는다.
        merged[doc_key]["rrf_score"] += bm25_weight * (1 / (k + rank))

    # 벡터 검색 결과 순위도 RRF 점수로 반영한다.
    for rank, hit in enumerate(vector_hits, start=1):
        doc_key = f"{hit['_index']}::{hit['_id']}"

        if doc_key not in merged:
            merged[doc_key] = {
                "hit": hit,
                "rrf_score": 0,
            }

        merged[doc_key]["rrf_score"] += vector_weight * (1 / (k + rank))

    # RRF 점수가 높은 순서대로 정렬한다.
    sorted_results = sorted(
        merged.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    results = []

    for idx, item in enumerate(sorted_results[:size], start=1):
        hit = item["hit"]
        source = hit.get("_source", {})

        # print(
        #     "[DEBUG] RRF hit",
        #     idx,
        #     {
        #         "index": hit.get("_index"),
        #         "id": hit.get("_id"),
        #         "rrf_score": item.get("rrf_score"),
        #         "original_score": hit.get("_score"),
        #         "employee_id": source.get("employee_id"),
        #         "employee_name": source.get("employee_name"),
        #         "department": source.get("department"),
        #         "source": source.get("source"),
        #     },
        # )

        results.append(hit)

    return results

# =========================
# 조직/부서/팀 직접 검색 함수
# =========================

def get_basic_indices(permission_level: int) -> list[str]:
    """
    조직/부서/팀/직급 검색은 기본정보 인덱스만 사용한다.
    """

    indices = ACCESSIBLE_INDICES.get(permission_level, [])

    return [
        index
        for index in indices
        if "basic" in index
    ]


def employee_name_exists(employee_name: str) -> bool:
    """
    hr_basic_1에서 직원 이름이 실제로 존재하는지 확인한다.
    이름 존재 여부는 기본 인사정보 기준으로 판단한다.
    """

    if not employee_name:
        return False

    query = {
        "size": 0,
        "query": {
            "bool": {
                "should": [
                    {"term": {"employee_name.keyword": employee_name}},
                    {"match_phrase": {"employee_name": employee_name}},
                    {"match_phrase": {"embedding_text": f"이름: {employee_name}"}},
                ],
                "minimum_should_match": 1,
            }
        },
    }

    response = client.search(index="hr_basic_1", body=query)
    total = response.get("hits", {}).get("total", {})

    if isinstance(total, dict):
        return int(total.get("value", 0)) > 0

    return int(total or 0) > 0


_employee_name_cache: list[str] | None = None


def get_employee_names() -> list[str]:
    """
    hr_basic_1에 있는 실제 직원명 목록을 가져온다.
    이름 추측보다 실제 직원명 매칭을 우선하기 위한 캐시다.
    """

    global _employee_name_cache

    if _employee_name_cache is not None:
        return _employee_name_cache

    query = {
        "size": 10000,
        "_source": ["employee_name"],
        "query": {"match_all": {}},
    }

    response = client.search(index="hr_basic_1", body=query)
    names = []

    for hit in response.get("hits", {}).get("hits", []):
        name = hit.get("_source", {}).get("employee_name")

        if name and name not in names:
            names.append(name)

    _employee_name_cache = sorted(names, key=len, reverse=True)

    return _employee_name_cache


def find_employee_name_in_question(question: str) -> str | None:
    """
    질문 원문에 실제 직원명이 포함되어 있으면 그 이름을 반환한다.
    """

    if not question:
        return None

    compact_question = re.sub(r"\s+", "", question)

    for name in get_employee_names():
        if name and name in compact_question:
            return name

    return None


def get_department_list(permission_level: int) -> list[str]:
    """
    조회 가능한 부서 목록을 중복 제거해서 반환한다.

    department.keyword를 쓰지 않고,
    문서를 직접 읽어서 department 값을 모은다.
    """

    indices = get_basic_indices(permission_level)

    if not indices:
        return []

    query = {
        "size": 1000,
        "_source": ["department"],
        "query": {
            "match_all": {}
        },
    }

    response = client.search(index=indices, body=query)

    hits = response["hits"]["hits"]

    departments = set()

    for hit in hits:
        source = hit.get("_source", {})
        department = source.get("department")

        if department:
            departments.add(department)

    return sorted(list(departments))


def get_category_values(field_key: str, permission_level: int) -> list[str]:
    """
    FIELD_RULES에 등록된 기본 분류 필드를 직접 조회/집계한다.
    category_list intent는 hybrid 검색보다 직접 집계가 정확하다.
    """

    if field_key not in {"department", "team", "job_grade", "position"}:
        return []

    policy_values = {
        "department": DEPARTMENTS,
        "team": TEAMS,
        "job_grade": JOB_GRADES,
        "position": POSITIONS,
    }

    if field_key in policy_values:
        return policy_values[field_key]

    indices = get_basic_indices(permission_level)

    if not indices:
        return []

    source_fields = ["embedding_text"]

    if field_key == "department":
        source_fields.append("department")

    if field_key == "job_grade":
        source_fields.append("job_grade")

    query = {
        "size": 10000,
        "_source": source_fields,
        "query": {"match_all": {}},
    }

    response = client.search(index=indices, body=query)
    hits = response["hits"]["hits"]
    values = set()

    for hit in hits:
        source = hit.get("_source", {})
        value = get_allowed_field_value(
            source=source,
            embedding_text=source.get("embedding_text", ""),
            field_key=field_key,
        )

        if value:
            values.add(value.strip())

    return sorted(values)


def search_employees_by_department_or_team(
    department_or_team: str,
    permission_level: int,
    size: int = 20,
):
    """
    특정 부서 또는 팀의 직원 목록을 검색한다.

    현재 데이터 기준:
    - department = 부서
    - team = 팀

    단, 데이터에 팀 정보가 없을 수도 있으므로
    '개발팀'으로 질문하면 '개발부'도 함께 검색한다.
    """

    indices = get_basic_indices(permission_level)


    if not indices:
        return []

    print("[DEBUG] 직원 검색 indices:", indices)
    print("[DEBUG] 원본 검색어:", department_or_team)

    # =========================
    # 1. 검색어 확장
    # =========================
    # 개발팀 → 개발팀, 개발부
    # 인사팀 → 인사팀, 인사부
    # 마케팅팀 → 마케팅팀, 마케팅부

    search_terms = [department_or_team]

    if department_or_team in TEAM_TO_DEPARTMENT:
        search_terms.append(TEAM_TO_DEPARTMENT[department_or_team])

    print("[DEBUG] 확장 검색어:", search_terms)

    # =========================
    # 2. 검색 조건 생성
    # =========================

    should_conditions = []

    for term in search_terms:
        should_conditions.extend(
            [
                {"match_phrase": {"department": term}},
                {"match_phrase": {"embedding_text": term}},
                {"match_phrase": {"embedding_text": f"부서: {term}"}},
                {"match_phrase": {"embedding_text": f"팀: {term}"}},
            ]
        )

    query = {
        "query": {
            "bool": {
                "should": should_conditions,
                "minimum_should_match": 1,
            }
        },
        "size": size,
    }

    response = client.search(index=indices, body=query)

    hits = response["hits"]["hits"]

    print("[DEBUG] 직원 검색 결과 수:", len(hits))
    print(f"[SEARCH] employee_search hits={len(hits)}")

    return hits

def search_employees_by_conditions(
    permission_level: int,
    department: str | None = None,
    team: str | None = None,
    job_grade: str | None = None,
    position: str | None = None,
    size: int = 50,
):
    """
    부서/팀/직급/직책 조건으로 직원 목록을 직접 검색한다.

    예:
    - 인사부 알려줘 -> department="인사부"
    - 백엔드팀 직원 알려줘 -> team="백엔드팀"
    - 대리 직원 알려줘 -> job_grade="대리"
    - 팀장 누구야 -> position="팀장"
    - 인사부 팀장 알려줘 -> department="인사부", position="팀장"
    """

    indices = get_basic_indices(permission_level)

    if not indices:
        return []

    filter_conditions = []

    if department:
        filter_conditions.append(
            {"match_phrase": {"department": department}}
        )

    if team:
        filter_conditions.append(
            {"match_phrase": {"embedding_text": f"팀: {team}"}}
        )

    if job_grade:
        filter_conditions.append(
            {"match_phrase": {"job_grade": job_grade}}
        )

    if position:
        filter_conditions.append(
            {"match_phrase": {"embedding_text": f"직책: {position}"}}
        )

    if not filter_conditions:
        return []

    query = {
        "query": {
            "bool": {
                "filter": filter_conditions
            }
        },
        "size": size,
    }

    print("[DEBUG] 조건 직원 검색 indices:", indices)
    print("[DEBUG] 조건 직원 검색 query:", query)

    response = client.search(
        index=indices,
        body=query,
    )

    hits = response["hits"]["hits"]

    print("[DEBUG] 조건 직원 검색 결과 수:", len(hits))
    print(f"[SEARCH] condition_employee_search hits={len(hits)}")

    return hits


def build_filter_search_clause(filter_item: dict) -> dict | None:
    """
    task filter를 OpenSearch 검색 조건으로 바꾼다.

    최종 검증은 task_processor_service.py의 filter_hits_by_filters에서 한 번 더 한다.
    여기서는 후보군을 정확하게 줄이는 역할만 한다.
    """

    field = filter_item.get("field")
    op = filter_item.get("op", "eq")
    value = filter_item.get("value")

    if field not in FIELD_RULES or value is None or value == "":
        return None

    if isinstance(value, str) and value.strip().upper() == "NULL":
        return None

    rule = FIELD_RULES[field]

    if op == "exists":
        if rule.get("embedding_label"):
            return None

        return build_field_exists_clause(rule)

    # 숫자 비교는 OpenSearch에서 직접 처리하지 않고,
    # 후보 조회 후 filter_hits_by_filters에서 검증한다.
    if op not in {"eq", "contains"}:
        return None

    source_field = rule.get("source_field")
    if source_field:
        return {
            "match_phrase": {
                source_field: str(value)
            }
        }

    # embedding_text 안에 있는 필드는 OpenSearch 라벨 매칭에 의존하지 않는다.
    # 후보는 관련 인덱스에서 넓게 가져오고, 실제 값 비교는 get_allowed_field_value()로 한다.
    if rule.get("embedding_label"):
        return None

    return None


def search_employees_by_filter_conditions(
    indices: list[str],
    filters: list[dict],
    size: int = 10000,
) -> list[dict]:
    """
    입사일/퇴직일자/계약형태 같은 조건 검색은
    hybrid 검색 전에 OpenSearch filter로 후보를 줄인다.
    """

    if not indices:
        return []

    filter_conditions = []

    for filter_item in filters:
        clause = build_filter_search_clause(filter_item)

        if clause:
            filter_conditions.append(clause)

    query = {
        "query": (
            {
                "bool": {
                    "should": filter_conditions,
                    "minimum_should_match": 1,
                }
            }
            if filter_conditions
            else {
                "match_all": {}
            }
        ),
        "size": size,
    }

    print("[DEBUG] 필터 조건 검색 indices:", indices)
    print("[DEBUG] 필터 조건 검색 query:", query)

    response = client.search(
        index=indices,
        body=query,
    )

    hits = response["hits"]["hits"]

    print("[DEBUG] 필터 조건 검색 결과 수:", len(hits))
    print(f"[SEARCH] filter_condition_search hits={len(hits)}")

    return hits


def search_hits_by_employee_ids(
    indices: list[str],
    employee_ids: list[str],
    size: int = 10000,
) -> list[dict]:
    """
    조건을 만족한 employee_id들의 관련 문서를 다시 가져온다.
    조건 필드와 정렬/답변 필드가 서로 다른 인덱스에 있을 때 사용한다.
    """

    if not indices or not employee_ids:
        return []

    query = {
        "size": size,
        "query": {
            "terms": {
                "employee_id": employee_ids
            }
        },
    }

    response = client.search(
        index=indices,
        body=query,
    )

    return response["hits"]["hits"]

def count_employees_by_conditions(
    permission_level: int,
    department: str | None = None,
    team: str | None = None,
    job_grade: str | None = None,
    position: str | None = None,
) -> int:
    """
    부서/팀/직급/직책 조건에 맞는 직원 수를 OpenSearch count로 계산한다.

    중요:
    - 직원 수는 목록 size=50으로 세면 안 된다.
    - hr_basic_1은 직원 기본 문서 기준으로 1명당 1문서라고 보고 count에 사용한다.
    - hr_basic_2, hr_basic_3까지 같이 세면 같은 직원이 중복 집계될 수 있다.
    """

    accessible_basic_indices = get_basic_indices(permission_level)

    indices = [
        index
        for index in accessible_basic_indices
        if index == "hr_basic_1"
    ]

    if not indices:
        return 0

    filter_conditions = []

    if department:
        filter_conditions.append(
            {"match_phrase": {"department": department}}
        )

    if team:
        filter_conditions.append(
            {"match_phrase": {"embedding_text": f"팀: {team}"}}
        )

    if job_grade:
        filter_conditions.append(
            {"match_phrase": {"job_grade": job_grade}}
        )

    if position:
        filter_conditions.append(
            {"match_phrase": {"embedding_text": f"직책: {position}"}}
        )

    if filter_conditions:
        query = {
            "query": {
                "bool": {
                    "filter": filter_conditions
                }
            }
        }
    else:
        query = {
            "query": {
                "match_all": {}
            }
        }

    print("[DEBUG] 직원 수 count indices:", indices)
    print("[DEBUG] 직원 수 count query:", query)

    # 관리자 권한이면 OpenSearch count API로 바로 직원 수를 계산한다.
    # 관리자는 퇴사자까지 볼 수 있으므로 별도 퇴사자 제외 처리를 하지 않는다.
    if permission_level >= 3:
        response = client.count(
            index=indices,
            body=query,
        )
        count = int(response.get("count", 0))
    # 일반 사용자 권한이면 퇴사자를 제외해야 하므로 count API를 바로 쓰지 않는다.
    else:
        # 먼저 조건에 맞는 직원의 employee_id만 조회한다.
        # _source를 employee_id로 제한해서 불필요한 개인정보 조회를 줄인다.
        search_query = {
            **query,
            "size": 10000,
            "_source": ["employee_id"],
        }
        response = client.search(
            index=indices,
            body=search_query,
        )

        # 검색 결과에서 employee_id만 꺼낸다.
        # dict.fromkeys를 사용해서 employee_id 중복을 제거하고 순서는 유지한다.
        employee_ids = list(
            dict.fromkeys(
                hit.get("_source", {}).get("employee_id")
                for hit in response["hits"]["hits"]
                if hit.get("_source", {}).get("employee_id")
            )
        )
        # 검색된 직원들 중 퇴사자 employee_id 목록을 가져온다.
        retired_employee_ids = get_retired_employee_ids(employee_ids)
        # 일반 사용자는 퇴사자를 볼 수 없으므로 퇴사자를 제외하고 count한다.
        count = sum(
            1
            for employee_id in employee_ids
            if employee_id not in retired_employee_ids
        )

    print("[DEBUG] 직원 수 count 결과:", count)
    print(f"[SEARCH] employee_count count={count}")

    return count

def search_managers(
    permission_level: int,
    department_or_team: str | None = None,
    size: int = 20,
):
    """
    팀장 목록 또는 특정 부서/팀의 팀장을 검색한다.

    현재 데이터에는 팀장 여부를 판단할 수 있는 필드가 없다.
    department = 부서
    team = 팀
    position = 직책
    job_grade = 직급
    따라서 팀장 여부는 확인할 수 없다.
    """

    return []


def format_employee_list_answer(hits) -> str:
    """
    직원 목록 검색 결과를 답변 문장으로 바꾼다.
    """

    if not hits:
        return "조건에 맞는 조회 결과가 없습니다."

    lines = []

    for hit in hits:
        source = hit.get("_source", {})
        embedding_text = source.get("embedding_text", "")

        employee_name = (
            get_allowed_field_value(source, embedding_text, "employee_name")
            or "이름 없음"
        )
        employee_id = (
            get_allowed_field_value(source, embedding_text, "employee_id")
            or "사번 없음"
        )
        department = (
            get_allowed_field_value(source, embedding_text, "department")
            or "부서 없음"
        )
        team = (
            get_allowed_field_value(source, embedding_text, "team")
            or "팀 없음"
        )
        position = (
            get_allowed_field_value(source, embedding_text, "position")
            or "직책 없음"
        )
        job_grade = (
            get_allowed_field_value(source, embedding_text, "job_grade")
            or "직급 없음"
        )
        lines.append(
            f"- {employee_name} ({employee_id}) / {department} / {team} / {position} / {job_grade}"
        )
    return "\n".join(lines)


def make_sources(hits, allowed_fields=None) -> list[dict]:
    """
    검색 결과를 API sources 형태로 변환한다.

    원칙:
    - allowed_fields에 포함된 필드만 sources에 넣는다.
    - 디버깅을 위해 검색된 문서의 embedding_text 원문도 함께 보여준다.
    """

    allowed_fields = set(allowed_fields or [])
    sources = []

    for hit in hits:
        source = hit.get("_source", {})
        embedding_text = source.get("embedding_text", "")

        source_item = {
            "index": hit["_index"],
            "_id": hit["_id"],
            "score": hit.get("_score"),
            "embedding_text": embedding_text,
        }

        if "employee" in allowed_fields:
            employee_info = get_allowed_field_value(
                source=source,
                embedding_text=embedding_text,
                field_key="employee",
            )

            source_item["employee_id"] = employee_info.get("employee_id")
            source_item["employee_name"] = employee_info.get("employee_name")

        for field_key in allowed_fields:
            if field_key == "employee":
                continue

            # change_history는 OpenSearch의 changed 배열을 그대로 노출한다.
            if field_key == "change_history":
                changed = source.get("changed")
                if changed:
                    source_item["change_history"] = changed
                continue

            rule = FIELD_RULES.get(field_key)

            if not rule:
                continue

            value = get_allowed_field_value(
                source=source,
                embedding_text=embedding_text,
                field_key=field_key,
            )

            if value:
                source_item[field_key] = value

        sources.append(source_item)

    return sources

def filter_hits_with_answer_values(hits, answer_fields: list[str]) -> list[dict]:
    """
    사용자가 실제로 요청한 필드 값이 있는 검색 결과만 남긴다.

    예:
    - 주소 질문이면 주소 값이 있는 문서만 남긴다.
    - 연봉 질문이면 연봉 값이 있는 문서만 남긴다.

    employee는 표시용 필드이므로 검사 대상에서 제외한다.
    """

    filtered_hits = []

    fields_to_check = [
        field
        for field in answer_fields
        if field != "employee"
    ]

    if not fields_to_check:
        return hits

    for hit in hits:
        source = hit.get("_source", {})
        embedding_text = source.get("embedding_text", "")

        has_value = False

        for field_key in fields_to_check:
            # change_history는 일반 필드가 아니라 OpenSearch의 changed 배열을 본다.
            if field_key == "change_history":
                if source.get("changed"):
                    has_value = True
                    break
                continue

            value = get_allowed_field_value(
                source=source,
                embedding_text=embedding_text,
                field_key=field_key,
            )

            if value:
                has_value = True
                break

        if has_value:
            filtered_hits.append(hit)

    return filtered_hits


# =========================
# 하이브리드 검색 함수
# =========================

def search_hybrid(
    question,
    permission_level,
    employee_id=None,
    employee_name=None,
    extract_name=True,
    size=20,
    indices=None,
    filters=None,
):
    """
    BM25 검색과 벡터 검색을 각각 수행한 뒤,
    RRF 방식으로 결과를 병합한다.
    """

    # start_time = time.perf_counter()

    if employee_id:
        employee_id = employee_id.strip().upper()

    # vector_create_start_time = time.perf_counter()
    question_vector = create_question_vector(question)
    # print(
    #     "[TIME] create_question_vector:",
    #     f"{time.perf_counter() - vector_create_start_time:.3f}s",
    # )

    # 바깥에서 indices를 넘기면 그 인덱스만 검색한다.
    # 안 넘기면 기존 방식대로 질문과 권한 기준으로 인덱스를 선택한다.
    indices = indices or select_search_indices(question, permission_level)

    # bm25_start_time = time.perf_counter()
    bm25_hits = search_bm25(
        question=question,
        permission_level=permission_level,
        employee_id=employee_id,
        employee_name=employee_name,
        extract_name=extract_name,
        size=size,
        indices=indices,
        filters=filters,
    )
    # print(
    #     "[TIME] search_bm25:",
    #     f"{time.perf_counter() - bm25_start_time:.3f}s",
    # )

    # print("[DEBUG] BM25 hits count:", len(bm25_hits))



    # vector_start_time = time.perf_counter()
    vector_hits = search_vector(
        question=question,
        question_vector=question_vector,
        permission_level=permission_level,
        employee_id=employee_id,
        employee_name=employee_name,
        extract_name=extract_name,
        size=size,
        indices=indices,
    )
    # print(
    #     "[TIME] search_vector:",
    #     f"{time.perf_counter() - vector_start_time:.3f}s",
    # )

    # print("[DEBUG] Vector hits count:", len(vector_hits))

 

    # rrf_start_time = time.perf_counter()

    # BM25 검색 결과와 벡터 검색 결과를 RRF 방식으로 병합한다. 호출 
    hybrid_hits = merge_rrf(
        bm25_hits=bm25_hits,        # 키워드 기반 검색 결과
        vector_hits=vector_hits,    # 의미 기반 벡터 검색 결과
        size=size,                  # 최종 검색 결과 개수

        # 정확한 조건 검색을 우선하기 위해 BM25 70%, 벡터 30% 비율로 RRF 점수를 계산한다.
        bm25_weight=0.7,
        vector_weight=0.3,
    )
    # print(
    #     "[TIME] merge_rrf:",
    #     f"{time.perf_counter() - rrf_start_time:.3f}s",
    # )

    print(
        f"[SEARCH] hybrid bm25={len(bm25_hits)} "
        f"vector={len(vector_hits)} rrf={len(hybrid_hits)}"
    )
    # print("[TIME] search_hybrid total:", f"{time.perf_counter() - start_time:.3f}s")

    return hybrid_hits
