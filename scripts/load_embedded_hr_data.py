import json
import os
from pathlib import Path

from dotenv import load_dotenv
from opensearchpy import OpenSearch

load_dotenv()

DATA_DIR = Path("data")
VECTOR_DIM = 384

# 전체 적재
MAX_COUNT = None

# 기존 인덱스 삭제 후 다시 만들고 싶으면 True
# 처음 실험할 때 기존 인덱스가 꼬였으면 True로 변경
RECREATE_INDEXES = True


client = OpenSearch(
    hosts=[
        {
            "host": os.getenv("OPENSEARCH_HOST", "localhost"),
            "port": int(os.getenv("OPENSEARCH_PORT", 9200)),
        }
    ],
    http_auth=(
        os.getenv("OPENSEARCH_USER", "admin"),
        os.getenv("OPENSEARCH_PASSWORD", "admin"),
    ),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False,
)


def find_all_jsonl_files():
    """
    data 폴더 안의 모든 jsonl 파일을 찾는다.
    여러 파일을 전부 읽어서 여러 인덱스에 적재한다.
    """
    jsonl_files = sorted(DATA_DIR.glob("*.jsonl"))

    if not jsonl_files:
        raise FileNotFoundError(
            "data 폴더 안에 .jsonl 파일이 없습니다. "
            "받은 임베딩 파일들을 C:\\dev\\Durian\\data 폴더에 넣어주세요."
        )

    print("적재 대상 jsonl 파일 목록:")
    for file_path in jsonl_files:
        print("-", file_path)

    return jsonl_files


def to_int(value, default=1):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def source_to_doc_type(source: str) -> str:
    if source == "기본인사정보":
        return "basic"
    if source == "급여정보":
        return "salary"
    if source == "역량성과":
        return "performance"

    return "unknown"


def get_source_min_level(doc_type: str) -> int:
    if doc_type == "basic":
        return 1

    if doc_type in ["salary", "performance"]:
        return 2

    return 1


def calculate_permission_level(doc: dict, doc_type: str) -> int:
    department_level = to_int(doc.get("department_level"), 1)
    position_level = to_int(doc.get("position_level"), 1)
    source_min_level = get_source_min_level(doc_type)

    permission_level = max(department_level, position_level, source_min_level)

    # 우리 프로젝트는 권한 1, 2, 3까지만 사용
    if permission_level < 1:
        permission_level = 1

    if permission_level > 3:
        permission_level = 3

    return permission_level


def create_index_if_not_exists(index_name: str):
    if client.indices.exists(index=index_name):
        return

    body = {
        "settings": {
            "index": {
                "knn": True
            },
            "analysis": {
                "analyzer": {
                    "korean_analyzer": {
                        "type": "custom",
                        "tokenizer": "nori_tokenizer"
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "employee_id": {"type": "keyword"},
                "employee_name": {
                    "type": "text",
                    "analyzer": "korean_analyzer"
                },
                "department": {"type": "keyword"},
                "department_level": {"type": "integer"},
                "position": {"type": "keyword"},
                "position_level": {"type": "integer"},
                "permission_level": {"type": "integer"},
                "doc_type": {"type": "keyword"},
                "source": {"type": "keyword"},
                "embedding_text": {
                    "type": "text",
                    "analyzer": "korean_analyzer"
                },
                "embedding_vector": {
                    "type": "knn_vector",
                    "dimension": VECTOR_DIM
                },
                "timestamp": {
                    "type": "date",
                    "format": "yyyy-MM-dd HH:mm:ss||strict_date_optional_time"
                },
                "changed": {"type": "keyword"}
            }
        }
    }

    client.indices.create(index=index_name, body=body)
    print(f"인덱스 생성 완료: {index_name}")

def prepare_indexes():
    index_names = [
        "hr_basic_1",
        "hr_basic_2",
        "hr_basic_3",
        "hr_salary_2",
        "hr_salary_3",
        "hr_performance_2",
        "hr_performance_3",
    ]

    if RECREATE_INDEXES:
        for index_name in index_names:
            if client.indices.exists(index=index_name):
                client.indices.delete(index=index_name)
                print(f"기존 인덱스 삭제: {index_name}")

    for index_name in index_names:
        create_index_if_not_exists(index_name)


def validate_doc(doc: dict, file_path: Path, line_no: int) -> bool:
    required_fields = [
        "employee_id",
        "employee_name",
        "department",
        "department_level",
        "position",
        "position_level",
        "embedding_text",
        "embedding_vector",
        "source",
    ]

    for field in required_fields:
        if field not in doc:
            print(f"{file_path.name} / {line_no}번째 줄 스킵: {field} 없음")
            return False

    vector = doc.get("embedding_vector")

    if not isinstance(vector, list):
        print(f"{file_path.name} / {line_no}번째 줄 스킵: embedding_vector가 list가 아님")
        return False

    if len(vector) != VECTOR_DIM:
        print(
            f"{file_path.name} / {line_no}번째 줄 스킵: 벡터 차원 불일치 "
            f"현재={len(vector)}, 필요={VECTOR_DIM}"
        )
        return False

    doc_type = source_to_doc_type(doc.get("source", ""))

    if doc_type == "unknown":
        print(
            f"{file_path.name} / {line_no}번째 줄 스킵: "
            f"알 수 없는 source={doc.get('source')}"
        )
        return False

    return True


def load_one_file(file_path: Path, total_count: int, index_count: dict):
    count = 0
    skip_count = 0

    print(f"\n========== 파일 적재 시작: {file_path.name} ==========")

    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"{file_path.name} / {line_no}번째 줄 JSON 파싱 실패: {e}")
                skip_count += 1
                continue

            if not validate_doc(doc, file_path, line_no):
                skip_count += 1
                continue

            source = doc.get("source", "")
            doc_type = source_to_doc_type(source)
            permission_level = calculate_permission_level(doc, doc_type)

            # 문자열 숫자를 정수로 변환해서 저장
            doc["department_level"] = to_int(doc.get("department_level"), 1)
            doc["position_level"] = to_int(doc.get("position_level"), 1)

            # 없는 필드 추가
            doc["doc_type"] = doc_type
            doc["permission_level"] = permission_level

            index_name = f"hr_{doc_type}_{permission_level}"

        

            # 파일명까지 넣어서 ID 충돌 방지
            doc_id = f"{file_path.stem}_{doc['employee_id']}_{doc_type}_{line_no}"

            client.index(
                index=index_name,
                id=doc_id,
                body=doc,
            )

            count += 1
            total_count += 1
            index_count[index_name] = index_count.get(index_name, 0) + 1

            print(
                f"\r적재중... {total_count}건",
                end="",
                flush=True
            )

            if MAX_COUNT is not None and total_count >= MAX_COUNT:
                return count, skip_count, total_count

    return count, skip_count, total_count


def load_all_files():
    prepare_indexes()

    jsonl_files = find_all_jsonl_files()

    total_count = 0
    total_skip_count = 0
    file_count = {}
    index_count = {}

    for file_path in jsonl_files:
        loaded_count, skip_count, total_count = load_one_file(
            file_path=file_path,
            total_count=total_count,
            index_count=index_count,
        )

        file_count[file_path.name] = loaded_count
        total_skip_count += skip_count

        if MAX_COUNT is not None and total_count >= MAX_COUNT:
            break
    print()
    print("\n========== 전체 적재 결과 ==========")
    print(f"총 적재 문서 수: {total_count}")
    print(f"총 스킵 문서 수: {total_skip_count}")

    print("\n파일별 적재 수:")
    for file_name, num in sorted(file_count.items()):
        print(f"{file_name}: {num}건")

    print("\n인덱스별 적재 수:")
    for index_name, num in sorted(index_count.items()):
        print(f"{index_name}: {num}건")


if __name__ == "__main__":
    load_all_files()