import os
import re
from dotenv import load_dotenv
from opensearchpy import OpenSearch
from transformers import AutoTokenizer, AutoModel
import torch

from app.services.question_service import extract_employee_name


# =========================
# 권한별 접근 가능 인덱스 설정
# =========================

# 사용자 permission_level에 따라 검색 가능한 인덱스를 제한한다.
# 실제 권한 제한은 검색 쿼리 안에서 하는 것이 아니라,
# 애초에 검색 대상 인덱스를 제한하는 방식으로 처리한다.
ACCESSIBLE_INDICES = {
    1: ["hr_basic_1"],
    2: [
        "hr_basic_1",
        "hr_basic_2",
        "hr_performance_2",
        "hr_salary_2",
    ],
    3: [
        "hr_basic_1",
        "hr_basic_2",
        "hr_basic_3",
        "hr_performance_2",
        "hr_performance_3",
        "hr_salary_2",
        "hr_salary_3",
    ],
}


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
        "부서",
        "팀",
        "직급",
        "직책",
        "입사일",
        "이메일",
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


# Python 코드에서 OpenSearch에 검색 요청을 보내기 위한 클라이언트 객체
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
    예: "팀: 브랜드팀 직책: 팀원"에서 팀 값을 꺼내면 "브랜드팀"
    """

    if not embedding_text:
        return ""

    pattern = rf"(?:^|\s){re.escape(field_name)}:\s*(.*?)(?=\s+[가-힣A-Za-z0-9_]+:|$)"
    match = re.search(pattern, embedding_text)

    if not match:
        return ""

    return match.group(1).strip()


def build_list_summary(search_hits, question: str) -> str | None:
    """
    부서/팀/직급/직책 종류 질문을 공통으로 처리한다.

    예:
    - 부서종류 알려줘
    - 팀종류 알려줘
    - 직급 목록 알려줘
    - 직책 리스트 알려줘
    """

    question = question.strip()

    is_list_question = any(word in question for word in ["종류", "목록", "리스트"])

    if not is_list_question:
        return None

    list_targets = {
        "부서": {
            "label": "부서",
            "getter": lambda source, embedding_text: source.get("department", ""),
        },
        "팀": {
            "label": "팀",
            "getter": lambda source, embedding_text: extract_embedding_text_field(
                embedding_text,
                "팀",
            ),
        },
        "직급": {
            "label": "직급",
            "getter": lambda source, embedding_text: source.get("job_grade", ""),
        },
        "직책": {
            "label": "직책",
            "getter": lambda source, embedding_text: extract_embedding_text_field(
                embedding_text,
                "직책",
            ),
        },
    }

    for keyword, config in list_targets.items():
        if keyword not in question:
            continue

        values = set()

        for hit in search_hits:
            source = hit.get("_source", {})
            embedding_text = source.get("embedding_text", "")

            value = config["getter"](source, embedding_text)

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

    response = client.search(index=indices, body=query)

    hits = response["hits"]["hits"]

    values = set()

    for hit in hits:
        source = hit.get("_source", {})
        embedding_text = source.get("embedding_text", "")

        if target == "부서":
            value = source.get("department", "")
        elif target == "직급":
            value = source.get("job_grade", "")
        elif target == "팀":
            value = extract_embedding_text_field(embedding_text, "팀")
        elif target == "직책":
            value = extract_embedding_text_field(embedding_text, "직책")
        else:
            value = ""

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

def build_context(search_hits, question=""):
    """
    OpenSearch 검색 결과를 LLM에 전달할 context 문자열로 변환한다.

    목록 질문이면 먼저 목록 요약만 반환한다.
    일반 질문이면 검색 결과를 LLM이 읽기 쉬운 context로 변환한다.
    """

    list_summary = build_list_summary(search_hits, question)

    if list_summary:
        return list_summary

    context_list = []

    for hit in search_hits:
        index_name = hit["_index"]
        doc_id = hit["_id"]
        source = hit["_source"]

        embedding_text = source.get("embedding_text", "")

        employee_id = source.get("employee_id", "")
        employee_name = source.get("employee_name", "")
        department = source.get("department", "")
        job_grade = source.get("job_grade", "")

        team = extract_embedding_text_field(embedding_text, "팀")
        position = extract_embedding_text_field(embedding_text, "직책")
        email = extract_embedding_text_field(embedding_text, "이메일")
        phone = extract_embedding_text_field(embedding_text, "전화번호")
        address = extract_embedding_text_field(embedding_text, "주소")

        birth_date = extract_embedding_text_field(embedding_text, "생년월일")
        military = extract_embedding_text_field(embedding_text, "병역")
        education = extract_embedding_text_field(embedding_text, "학력")
        university = extract_embedding_text_field(embedding_text, "출신대학")

        salary = extract_embedding_text_field(embedding_text, "연봉")
        bank = extract_embedding_text_field(embedding_text, "급여은행")
        account = extract_embedding_text_field(embedding_text, "계좌번호")
        overtime = extract_embedding_text_field(embedding_text, "잔업시간")
        unused_vacation = extract_embedding_text_field(embedding_text, "미사용휴가일수")

        performance_score = extract_embedding_text_field(embedding_text, "성과점수")
        evaluation_2024 = extract_embedding_text_field(embedding_text, "인사고과_2024")
        certificate_allowance = extract_embedding_text_field(embedding_text, "자격증수당여부")

        rrn = extract_embedding_text_field(embedding_text, "주민등록번호")

        doc_type = "기본정보"

        if "salary" in index_name:
            doc_type = "급여정보"
        elif "performance" in index_name:
            doc_type = "성과정보"

        detail_lines = []

        field_values = {
            "팀": team,
            "직책": position,
            "이메일": email,
            "전화번호": phone,
            "주소": address,
            "생년월일": birth_date,
            "병역": military,
            "학력": education,
            "출신대학": university,
            "연봉": salary,
            "급여은행": bank,
            "계좌번호": account,
            "잔업시간": overtime,
            "미사용휴가일수": unused_vacation,
            "성과점수": performance_score,
            "인사고과_2024": evaluation_2024,
            "자격증수당여부": certificate_allowance,
            "주민등록번호": rrn,
        }

        for field_name, value in field_values.items():
            if value:
                detail_lines.append(f"{field_name}: {value}")

        detail_text = "\n".join(detail_lines)

        context = f"""
            [출처: {index_name} / {doc_id}]
            문서유형: {doc_type}
            사번: {employee_id}
            이름: {employee_name}
            부서: {department}
            직급: {job_grade}
            {detail_text}
            원문내용: {embedding_text}
            """.strip()

        context_list.append(context)

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
    with torch.no_grad():
        outputs = embedding_model(**inputs)

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

    employee_id = employee_id.strip().upper()

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

    if not hits:
        return None

    source = hits[0]["_source"]

    department_level = source.get("department_level", 1)
    job_grade_level = source.get("job_grade_level", 1)

    return max(department_level, job_grade_level)


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


# =========================
# BM25 검색 함수
# =========================

def search_bm25(question, permission_level, employee_id=None, size=5, indices=None):
    """
    embedding_text 필드를 대상으로 BM25 기반 키워드 검색을 수행한다.

    - employee_id가 있으면 employee_id 기준으로 검색한다.
    - employee_id가 없으면 질문 전체 + 핵심 키워드로 검색한다.
    - 이름이 있으면 employee_name 또는 embedding_text에서 이름을 찾는다.
    """

    indices = indices or select_search_indices(question, permission_level)

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
                        {"term": {"employee_id": employee_id}}
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
                    "filter": [],
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

        # 부서명이 직접 들어온 경우 department 필터 추가
        departments = [
            "마케팅부",
            "기획부",
            "인사부",
            "개발부",
            "영업부",
            "재무부",
        ]

        for department in departments:
            if department in question:
                query["query"]["bool"]["filter"].append(
                    {"match_phrase": {"department": department}}
                )
                break

        # 직원 이름이 들어온 경우 employee_name 또는 embedding_text에서 검색
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

    response = client.search(
        index=indices,
        body=query,
    )

    hits = response["hits"]["hits"]


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
    size=5,
    indices=None,
):
    """
    embedding_vector 필드를 대상으로 벡터 유사도 검색을 수행한다.

    벡터 검색은 단어가 정확히 같지 않아도
    의미가 비슷한 문서를 찾는 데 사용한다.
    """

    # 권한 레벨과 질문 내용에 따라 검색 가능한 인덱스를 선택한다.
    indices = indices or select_search_indices(question, permission_level)

    if not indices:
        print("접근 가능한 인덱스가 없습니다.")
        return []

    target_name = extract_employee_name(question)

    if target_name:
        candidate_query = {
            "size": max(size * 20, 50),
            "query": {
                "bool": {
                    "should": [
                        {"match_phrase": {"employee_name": target_name}},
                        {"match_phrase": {"embedding_text": target_name}},
                    ],
                    "minimum_should_match": 1,
                }
            },
        }

        response = client.search(index=indices, body=candidate_query)
        candidate_hits = response["hits"]["hits"]

        if candidate_hits:
            ranked_hits = []

            for hit in candidate_hits:
                embedding_vector = hit.get("_source", {}).get("embedding_vector")

                if not embedding_vector:
                    continue

                score = cosine_similarity(question_vector, embedding_vector)
                ranked_hits.append((score, hit))

            ranked_hits.sort(key=lambda item: item[0], reverse=True)

            return [hit for score, hit in ranked_hits[:size]]

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

    if employee_id:
        hits = [
            hit
            for hit in hits
            if hit.get("_source", {}).get("employee_id") == employee_id
        ]

    elif target_name:
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

def merge_rrf(bm25_hits, vector_hits, k=60, size=5):
    """
    BM25 검색 결과와 벡터 검색 결과를 RRF 방식으로 병합한다.

    RRF는 BM25 점수와 벡터 점수를 직접 더하지 않는다.
    대신 각 검색 결과의 순위를 기준으로 최종 점수를 계산한다.

    공식:
    RRF 점수 = 1 / (k + 순위)

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
        merged[doc_key]["rrf_score"] += 1 / (k + rank)

    # 벡터 검색 결과 순위도 RRF 점수로 반영한다.
    for rank, hit in enumerate(vector_hits, start=1):
        doc_key = f"{hit['_index']}::{hit['_id']}"

        if doc_key not in merged:
            merged[doc_key] = {
                "hit": hit,
                "rrf_score": 0,
            }

        merged[doc_key]["rrf_score"] += 1 / (k + rank)

    # RRF 점수가 높은 순서대로 정렬한다.
    sorted_results = sorted(
        merged.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    # 상위 size개만 반환한다.
    return [item["hit"] for item in sorted_results[:size]]


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

    # =========================
    # 1. 검색어 확장
    # =========================
    # 개발팀 → 개발팀, 개발부
    # 인사팀 → 인사팀, 인사부
    # 마케팅팀 → 마케팅팀, 마케팅부

    search_terms = [department_or_team]

    team_to_department = {
        "인사팀": "인사부",
        "마케팅팀": "마케팅부",
        "개발팀": "개발부",
        "영업팀": "영업부",
        "재무팀": "재무부",
        "기획팀": "기획부",
    }

    if department_or_team in team_to_department:
        search_terms.append(team_to_department[department_or_team])



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



    return hits


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
        return "조회 가능한 정보가 없습니다."

    lines = []

    for hit in hits:
        source = hit.get("_source", {})

        employee_name = source.get("employee_name", "이름 없음")
        employee_id = source.get("employee_id", "사번 없음")
        department = source.get("department", "부서 없음")
        embedding_text = source.get("embedding_text", "")
        team = extract_embedding_text_field(embedding_text, "팀") or "팀 없음"
        position = extract_embedding_text_field(embedding_text, "직책") or "직책 없음"
        job_grade = source.get("job_grade", "직급 없음")
        lines.append(
            f"- {employee_name} ({employee_id}) / {department} / {team} / {position} / {job_grade}"
        )
    return "\n".join(lines)


def make_sources(hits) -> list[dict]:
    """
    검색 결과를 API sources 형태로 변환한다.
    """

    sources = []

    for hit in hits:
        source = hit.get("_source", {})
        embedding_text = source.get("embedding_text", "")

        sources.append(
            {
                "index": hit["_index"],
                "_id": hit["_id"],
                "employee_id": source.get("employee_id"),
                "employee_name": source.get("employee_name"),
                "department": source.get("department"),
                "team": extract_embedding_text_field(embedding_text, "팀"),
                "position": extract_embedding_text_field(embedding_text, "직책"),
                "job_grade": source.get("job_grade"),
                "score": hit.get("_score"),
            }
        )

    return sources


# =========================
# 하이브리드 검색 함수
# =========================

def search_hybrid(question, permission_level, employee_id=None, size=20):
    """
    BM25 검색과 벡터 검색을 각각 수행한 뒤,
    RRF 방식으로 결과를 병합한다.

    최종적으로 RAG에 사용할 검색 결과를 반환한다.
    """

    if employee_id:
        employee_id = employee_id.strip().upper()

    # 사용자 질문을 벡터로 변환한다.
    question_vector = create_question_vector(question)

    # BM25와 벡터 검색이 같은 인덱스를 보도록 검색 대상 인덱스를 먼저 선택한다.
    indices = select_search_indices(question, permission_level)

    # 1. BM25 키워드 검색 실행
    bm25_hits = search_bm25(
        question=question,
        permission_level=permission_level,
        employee_id=employee_id,
        size=size,
        indices=indices,
    )

    # 2. 벡터 의미 검색 실행
    vector_hits = search_vector(
        question=question,
        question_vector=question_vector,
        permission_level=permission_level,
        employee_id=employee_id,
        size=size,
        indices=indices,
    )

    # 3. 두 검색 결과를 RRF 방식으로 병합
    hybrid_hits = merge_rrf(
        bm25_hits=bm25_hits,
        vector_hits=vector_hits,
        size=size,
    )

    return hybrid_hits
