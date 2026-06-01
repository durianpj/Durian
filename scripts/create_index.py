from opensearchpy import OpenSearch

client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=("durian", "admin123!"),
    use_ssl=True,
    verify_certs=False,
)

index_name = "employees"

index_body = {
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
            "employee_name": {"type": "text"},
            "department": {"type": "keyword"},
            "team": {"type": "keyword"},
            "job_grade": {"type": "keyword"},
            "employment_status": {"type": "keyword"},
            "department_level": {"type": "integer"},
            "job_grade_level": {"type": "integer"},
            "permission_level": {"type": "integer"},
            "embedding_text": {
                "type": "text",
                "analyzer": "korean_analyzer"
            },
            "embedding_vector": {
                "type": "knn_vector",
                "dimension": 384
            }
        }
    }
}

if client.indices.exists(index=index_name):
    client.indices.delete(index=index_name)
    print("기존 employees 인덱스 삭제 완료")

client.indices.create(index=index_name, body=index_body)
print("employees 인덱스 생성 완료")