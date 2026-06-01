import json
from pathlib import Path

from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer


# =========================
# 기본 설정
# =========================

INDEX_NAME = "employees"

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "employees_sample.jsonl"

OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
OPENSEARCH_USER = "durian"
OPENSEARCH_PASSWORD = "admin123!"

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

        # OpenSearch employees 인덱스에 저장
        client.index(
            index=INDEX_NAME,
            id=doc_id,
            body=doc,
        )

        success_count += 1
        print(f"저장 완료: {doc_id}")

    print(f"\n총 {success_count}개 문서 적재 완료")


if __name__ == "__main__":
    main()