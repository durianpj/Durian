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

POSITION_RANK = {
    "사원": 1,
    "대리": 2,
    "과장": 3,
    "차장": 4,
    "부장": 5,
    "이사": 6,
    "상무": 7,
    "전무": 8,
    "부사장": 9,
    "사장": 10,
}

# 현재 정제 데이터에는 department_level, job_grade_level 필드가 없을 수 있다.
# 그래서 position(직급)을 기준으로 챗봇 권한 레벨 1~3을 계산한다.
POSITION_PERMISSION_LEVEL = {
    "사원": 1,
    "대리": 1,
    "과장": 2,
    "차장": 2,
    "부장": 3,
    "이사": 3,
    "상무": 3,
    "전무": 3,
    "부사장": 3,
    "사장": 3,
}

DEPARTMENTS = ["마케팅부", "기획부", "인사부", "개발부", "영업부", "재무부"]


def get_search_indices(question, permission_level):
    """
    질문 내용과 권한 레벨에 따라 검색 대상 인덱스를 결정한다.
    BM25와 벡터 검색이 같은 범위를 보도록 공통 함수로 관리한다.
    """

    indices = ACCESSIBLE_INDICES.get(permission_level, [])

    if any(keyword in question for keyword in ["연봉", "급여", "월급", "계좌번호", "은행"]):
        return [index for index in indices if "salary" in index]

    if any(keyword in question for keyword in ["성과", "평가", "고과"]):
        return [index for index in indices if "performance" in index]

    if any(keyword in question for keyword in ["주소", "주민번호", "주민등록번호"]):
        return [index for index in indices if "basic_3" in index]

    return indices


def extract_employee_name_from_source(source):
    """
    OpenSearch 문서에서 직원 이름을 추출한다.
    새 embedding_text의 '이름:' 형식과 기존 첫 단어 형식을 모두 지원한다.
    """

    employee_name = source.get("employee_name") or source.get("name")

    if employee_name:
        return employee_name

    embedding_text = source.get("embedding_text", "")

    if "이름:" in embedding_text:
        name_part = embedding_text.split("이름:", 1)[1].strip()
        return name_part.split()[0] if name_part else ""

    return embedding_text.split()[0] if embedding_text else ""


def extract_department_from_question(question):
    """
    질문에 포함된 부서명을 추출한다.
    """

    for department in DEPARTMENTS:
        if department in question:
            return department

    return None


def extract_phone_number_from_source(source):
    """
    구조화 필드 또는 embedding_text에서 실제 전화번호 패턴만 추출한다.
    이메일을 번호로 오인하지 않도록 010-0000-0000 형태만 허용한다.
    """

    phone_number = (
        source.get("phone_number")
        or source.get("phone")
        or source.get("전화번호")
        or source.get("연락처")
        or source.get("휴대폰")
    )

    if phone_number:
        return phone_number

    embedding_text = source.get("embedding_text", "")
    match = re.search(r"01[016789]-\d{3,4}-\d{4}", embedding_text)

    return match.group(0) if match else ""


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

tokenizer = None
embedding_model = None


def get_embedding_model():
    """
    벡터 검색이 실제로 필요할 때만 임베딩 모델을 로드한다.

    앱 시작 시점이 아니라 하이브리드 검색 실행 시점에 모델을 로드한다.
    """

    global tokenizer, embedding_model

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if embedding_model is None:
        embedding_model = AutoModel.from_pretrained(MODEL_NAME)

    return tokenizer, embedding_model


# =========================
# LLM 전달용 Context 생성 함수
# =========================


def build_context(search_hits):
    """
    OpenSearch 검색 결과를 LLM에 전달할 context 문자열로 변환한다.

    기존 구조는 유지하되,
    인덱스명 기준으로 문서유형을 추가해서
    LLM이 기본정보/급여정보/성과정보를 구분할 수 있게 한다.
    """

    context_list = []

    for hit in search_hits:
        # 검색 결과가 나온 인덱스명
        index_name = hit["_index"]

        # OpenSearch 문서 ID
        doc_id = hit["_id"]

        # 실제 문서 데이터
        source = hit["_source"]

        # 검색과 LLM context에 사용할 텍스트
        embedding_text = source.get("embedding_text", "")

        # 공통 메타데이터
        employee_id = source.get("employee_id", "")
        department = source.get("department", "")
        position = source.get("position", "")
        address = source.get("address", "")
        address_line = f"주소: {address}\n" if address else ""

        # employee_name 필드가 아직 없으므로
        # embedding_text의 첫 번째 단어를 이름으로 임시 추정한다.
        employee_name = source.get("employee_name", "")

        if not employee_name and embedding_text:
            employee_name = embedding_text.split()[0]

        # 인덱스명 기준으로 문서 유형 표시
        # 기존 데이터 구조는 바꾸지 않고, LLM에게 설명만 추가한다.
        doc_type = "기본정보"

        if "salary" in index_name:
            doc_type = "급여정보"
        elif "performance" in index_name:
            doc_type = "성과정보"

        # 출처를 함께 넣어야 나중에 어떤 문서를 보고 답했는지 알 수 있다.
        context = f"""
[출처: {index_name} / {doc_id}]
문서유형: {doc_type}
사번: {employee_id}
부서: {department}
직급/직책: {position}
{address_line}내용: {embedding_text}
""".strip()

        context_list.append(context)

    # 여러 검색 결과를 빈 줄로 구분해서 하나의 context 문자열로 합친다.
    return "\n\n".join(context_list)


# =========================
# 질문 임베딩 생성 함수
# =========================


def create_question_vector(question):
    """
    사용자 질문을 벡터 검색에 사용할 임베딩 벡터로 변환한다.

    원래 sentence_transformers를 쓰면 더 간단하지만,
    현재 PC에서 sklearn DLL 차단 문제가 있어서
    transformers + torch 방식으로 직접 임베딩을 만든다.
    """

    tokenizer, embedding_model = get_embedding_model()

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


def get_user_permission_level(employee_id: str) -> int | None:
    """
    사용자 사번으로 권한 레벨을 자동 계산한다.

    1순위: 문서에 department_level/job_grade_level 필드가 있으면 그 값을 사용한다.
    2순위: 현재 정제 데이터처럼 level 필드가 없으면 position(직급)을 기준으로 계산한다.

    권한 레벨은 챗봇 정책상 1, 2, 3 중 하나만 반환한다.
    """

    query = {
        "query": {"bool": {"filter": [{"term": {"employee_id": employee_id}}]}},
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

    department_level = source.get("department_level") or source.get("부서레벨")
    job_grade_level = source.get("job_grade_level") or source.get("직급레벨")

    # level 필드가 있는 데이터라면 기존 방식대로 둘 중 더 높은 값을 사용한다.
    if department_level is not None or job_grade_level is not None:
        department_level = int(department_level or 1)
        job_grade_level = int(job_grade_level or 1)
        permission_level = max(department_level, job_grade_level)
        return min(max(permission_level, 1), 3)

    # 현재 정제 데이터에는 position 필드만 있으므로 직급 기준으로 권한을 계산한다.
    position = source.get("position") or source.get("job_grade") or source.get("직급")

    return POSITION_PERMISSION_LEVEL.get(position, 1)

def get_employee_basic_info(employee_id: str) -> dict | None:
    """
    사번 기준으로 기본 인사정보를 조회한다.
    """

    query = {
        "query": {"bool": {"filter": [{"term": {"employee_id": employee_id}}]}},
        "size": 1,
    }

    response = client.search(index=["hr_basic_1"], body=query)
    hits = response["hits"]["hits"]

    if not hits:
        return None

    hit = hits[0]
    source = hit["_source"]

    return {
        "index": hit["_index"],
        "_id": hit["_id"],
        "employee_id": source.get("employee_id"),
        "name": extract_employee_name_from_source(source),
        "department": source.get("department"),
        "position": source.get("position"),
        "score": hit.get("_score"),
    }


def get_department_types() -> list[str]:
    """
    기본 인사 인덱스에서 부서 고유값 목록을 조회한다.

    부서명은 1레벨 기본 정보이므로 hr_basic_1 기준으로 집계한다.
    """

    query = {
        "size": 0,
        "aggs": {
            "departments": {
                "terms": {
                    "field": "department.keyword",
                    "size": 100,
                    "order": {"_key": "asc"},
                }
            }
        },
    }

    response = client.search(index=["hr_basic_1"], body=query)
    buckets = response.get("aggregations", {}).get("departments", {}).get("buckets", [])

    departments = [bucket["key"] for bucket in buckets if bucket.get("key")]

    if departments:
        return departments

    query["aggs"]["departments"]["terms"]["field"] = "department"
    response = client.search(index=["hr_basic_1"], body=query)
    buckets = response.get("aggregations", {}).get("departments", {}).get("buckets", [])

    return [bucket["key"] for bucket in buckets if bucket.get("key")]


def get_department_members(question: str, size=1000) -> list[dict]:
    """
    질문에 포함된 부서의 구성원 목록을 조회한다.
    '팀원'이나 '팀장'이 질문에 있으면 embedding_text 기준으로 직책도 제한한다.
    """

    department = extract_department_from_question(question)

    if not department:
        return []

    filters = [{"term": {"department": department}}]

    query = {
        "query": {
            "bool": {
                "must": [{"match_all": {}}],
                "filter": filters,
            }
        },
        "size": size,
    }

    if "팀원" in question:
        query["query"]["bool"]["must"].append(
            {"match_phrase": {"embedding_text": "팀원"}}
        )
    elif "팀장" in question:
        query["query"]["bool"]["must"].append(
            {"match_phrase": {"embedding_text": "팀장"}}
        )

    response = client.search(index=["hr_basic_1"], body=query)
    members_by_employee_id = {}

    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        employee_id = source.get("employee_id") or hit["_id"]

        if employee_id in members_by_employee_id:
            continue

        members_by_employee_id[employee_id] = {
            "employee_id": source.get("employee_id"),
            "name": extract_employee_name_from_source(source),
            "department": source.get("department"),
            "position": source.get("position"),
            "index": hit["_index"],
            "_id": hit["_id"],
            "score": hit.get("_score"),
        }

    members = list(members_by_employee_id.values())
    return sorted(members, key=lambda item: (item["position"] or "", item["name"]))


def get_supervisors(employee_id: str) -> list[dict]:
    """
    요청자와 같은 부서에서 요청자보다 직급 순위가 높은 직원을 조회한다.
    """

    user_query = {
        "query": {"bool": {"filter": [{"term": {"employee_id": employee_id}}]}},
        "size": 1,
    }

    user_response = client.search(index=["hr_basic_1"], body=user_query)
    user_hits = user_response["hits"]["hits"]

    if not user_hits:
        return []

    user_source = user_hits[0]["_source"]
    department = user_source.get("department", "")
    user_position = user_source.get("position", "")
    user_rank = POSITION_RANK.get(user_position, 0)

    if not department or not user_rank:
        return []

    query = {
        "query": {
            "bool": {
                "must": [{"match_all": {}}],
                "filter": [{"term": {"department": department}}],
                "must_not": [{"term": {"employee_id": employee_id}}],
            }
        },
        "size": 100,
    }

    response = client.search(index=["hr_basic_1"], body=query)
    supervisors = []

    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        position = source.get("position", "")
        rank = POSITION_RANK.get(position, 0)

        if rank <= user_rank:
            continue

        supervisors.append(
            {
                "employee_id": source.get("employee_id"),
                "name": extract_employee_name_from_source(source),
                "department": source.get("department"),
                "position": position,
                "rank": rank,
                "index": hit["_index"],
                "_id": hit["_id"],
                "score": hit.get("_score"),
            }
        )

    return sorted(supervisors, key=lambda item: (-item["rank"], item["name"]))


# =========================
# BM25 검색 함수
# =========================


def search_bm25(question, permission_level, employee_id=None, size=5):
    """
    embedding_text 필드를 대상으로 BM25 기반 키워드 검색을 수행한다.

    - employee_id가 있으면 본인/특정 사번 조회로 보고 employee_id 필터 중심으로 조회한다.
    - employee_id가 없으면 질문 내용을 embedding_text에서 검색한다.
    - 질문에 부서명이 있으면 department 필터를 추가한다.
    - 질문에 직원 이름이 있으면 embedding_text에서 이름을 match_phrase로 제한한다.
    """

    # 권한 레벨과 질문 내용에 따라 검색 가능한 인덱스만 가져온다.
    indices = get_search_indices(question, permission_level)

    # 필터링 후 검색할 인덱스가 없으면 빈 결과 반환
    if not indices:
        print("접근 가능한 인덱스가 없습니다.")
        return []

    # =========================
    # 1. 기본 Query DSL 생성
    # =========================

    if employee_id:
        # 본인 조회 또는 특정 사번 조회
        # "내 부서 알려줘" 같은 질문은 embedding_text에 그대로 없을 수 있으므로
        # match 검색 대신 employee_id 필터로 정확 조회한다.
        query = {
            "query": {
                "bool": {
                    "must": [{"match_all": {}}],
                    "filter": [
                        {"term": {"employee_id": employee_id}}
                    ],
                }
            },
            "size": size,
        }
    else:
        # 일반 검색
        # 예: "마케팅부 직원 찾아줘", "성과 좋은 직원 찾아줘"
        query = {
            "query": {
                "bool": {
                    "must": [{"match": {"embedding_text": question}}],
                    "filter": [],
                }
            },
            "size": size,
        }

        # =========================
        # 2. 부서 필터 추가
        # =========================
        # "마케팅부 직원 찾아줘"처럼 부서명이 있으면
        # department 필드로 정확히 제한한다.

        department_matched = False

        for department in DEPARTMENTS:
            if department in question:
                query["query"]["bool"]["filter"].append(
                    {"term": {"department": department}}
                )
                department_matched = True
                break

        # =========================
        # 3. 직원 이름 필터 추가
        # =========================
        # 현재 데이터에 employee_name 필드가 없기 때문에
        # 임시로 embedding_text에 이름이 포함된 문서만 검색한다.

        target_name = None

        if not department_matched:
            target_name = extract_employee_name(question)

        if target_name:
            query["query"]["bool"]["filter"].append(
                {"match_phrase": {"embedding_text": target_name}}
            )

    # =========================
    # 4. OpenSearch 검색 실행
    # =========================

    response = client.search(
        index=indices,
        body=query,
    )

    return response["hits"]["hits"]


# =========================
# 벡터 검색 함수
# =========================


def search_vector(question, question_vector, permission_level, employee_id=None, size=5):
    """
    embedding_vector 필드를 대상으로 벡터 유사도 검색을 수행한다.

    벡터 검색은 단어가 정확히 같지 않아도
    의미가 비슷한 문서를 찾는 데 사용한다.
    """

    # 권한 레벨과 질문 내용에 따라 검색 가능한 인덱스만 가져온다.
    indices = get_search_indices(question, permission_level)

    if not indices:
        print("접근 가능한 인덱스가 없습니다.")
        return []

    # OpenSearch k-NN 벡터 검색 쿼리
    query = {
        "size": size,
        "query": {
            "bool": {
                "filter": [],
                # must 안에서 embedding_vector와 질문 벡터를 비교한다.
                "must": [
                    {
                        "knn": {
                            "embedding_vector": {"vector": question_vector, "k": size}
                        }
                    }
                ],
            }
        },
    }

    # 본인 조회 또는 특정 사번 조회라면 벡터 검색도 반드시 employee_id로 제한한다.
    # 이 필터가 없으면 "내 연봉" 질문에서 다른 사람 문서가 섞일 수 있다.
    if employee_id:
        query["query"]["bool"]["filter"].append(
            {"term": {"employee_id": employee_id}}
        )

    # OpenSearch에 벡터 검색 요청을 보낸다.
    response = client.search(
        index=indices,
        body=query,
    )

    return response["hits"]["hits"]


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
            merged[doc_key] = {"hit": hit, "rrf_score": 0}

        # 순위가 높을수록 더 큰 점수를 받는다.
        merged[doc_key]["rrf_score"] += 1 / (k + rank)

    # 벡터 검색 결과 순위도 RRF 점수로 반영한다.
    for rank, hit in enumerate(vector_hits, start=1):
        doc_key = f"{hit['_index']}::{hit['_id']}"

        if doc_key not in merged:
            merged[doc_key] = {"hit": hit, "rrf_score": 0}

        merged[doc_key]["rrf_score"] += 1 / (k + rank)

    # RRF 점수가 높은 순서대로 정렬한다.
    sorted_results = sorted(
        merged.values(), key=lambda item: item["rrf_score"], reverse=True
    )

    # 상위 size개만 반환한다.
    return [item["hit"] for item in sorted_results[:size]]


# =========================
# 하이브리드 검색 함수
# =========================


def search_hybrid(question, permission_level, employee_id=None, size=5):
    """
    BM25 검색과 벡터 검색을 각각 수행한 뒤,
    RRF 방식으로 결과를 병합한다.

    최종적으로 RAG에 사용할 검색 결과를 반환한다.
    """

    question_vector = create_question_vector(question)

    bm25_hits = search_bm25(
        question=question,
        permission_level=permission_level,
        employee_id=employee_id,
        size=size,
    )

    vector_hits = search_vector(
        question=question,
        question_vector=question_vector,
        permission_level=permission_level,
        employee_id=employee_id,
        size=size,
    )

    hybrid_hits = merge_rrf(
        bm25_hits=bm25_hits,
        vector_hits=vector_hits,
        size=size,
    )

    return hybrid_hits
