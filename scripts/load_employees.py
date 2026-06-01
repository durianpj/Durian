import json
from pathlib import Path
import os
from dotenv import load_dotenv

from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer


# =========================
# 기본 설정
# =========================


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "employees_sample.jsonl"

load_dotenv()
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", 9200))
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# =========================
# OpenSearch 클라이언트 생성
# =========================

client = OpenSearch(
    hosts=[
        {
            "host": OPENSEARCH_HOST,
            "port": OPENSEARCH_PORT,
        }
    ],
    http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False,
)


# =========================
# 임베딩 모델 로드
# =========================

model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def load_jsonl(file_path: Path):
    """
    JSONL 파일을 한 줄씩 읽어서 dict로 변환
    """
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            yield json.loads(line)

def get_index_name(doc):
    """
    문서의 doc_type과 security_level을 기준으로 저장할 OpenSearch 인덱스 이름 결정
    """
    doc_type = doc.get("doc_type")
    security_level = doc.get("security_level")

    if doc_type == "basic":
        return f"hr_basic_{security_level}"

    if doc_type == "performance":
        return f"hr_performance_{security_level}"

    if doc_type == "salary":
        return f"hr_salary_{security_level}"

    raise ValueError(f"알 수 없는 doc_type입니다: {doc_type}")



def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"JSONL 파일을 찾을 수 없습니다: {DATA_PATH}")

    success_count = 0

    for doc in load_jsonl(DATA_PATH):
        # OpenSearch 문서 ID 처리
        # _id가 있으면 _id 사용, 없으면 doc_id 사용
        doc_id = doc.pop("_id", None) or doc.pop("doc_id", None)

        if not doc_id:
            print("문서 ID가 없어 건너뜀:", doc)
            continue

        embedding_text = doc.get("embedding_text")

        if not embedding_text:
            print(f"embedding_text가 없어 건너뜀: {doc_id}")
            continue

        # embedding_text를 벡터로 변환
        embedding_vector = model.encode(embedding_text).tolist()

        # OpenSearch에 저장할 문서에 벡터 추가
        doc["embedding_vector"] = embedding_vector

        target_index = get_index_name(doc)

        client.index(
            index=target_index,
            id=doc_id,
            body=doc,
        )

        success_count += 1
        print(f"저장 완료: {target_index} / {doc_id}")

    print(f"\n총 {success_count}개 문서 적재 완료")


if __name__ == "__main__":
    main()