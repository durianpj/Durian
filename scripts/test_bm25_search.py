import os
from dotenv import load_dotenv
from opensearchpy import OpenSearch


ACCESSIBLE_INDICES = {
    1: ["hr_basic_1"],
    2: ["hr_basic_1", "hr_basic_2", "hr_performance_2", "hr_salary_2"],
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


load_dotenv()

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", 9200))
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")


client = OpenSearch(
    hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
    http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False,
)


def build_context(search_hits):
    context_list = []

    for hit in search_hits:
        index_name = hit["_index"]
        doc_id = hit["_id"]
        source = hit["_source"]
        embedding_text = source.get("embedding_text", "")

        context = f"[출처: {index_name} / {doc_id}]\n{embedding_text}"
        context_list.append(context)

    return "\n\n".join(context_list)


def search_bm25(question, permission_level, employee_id=None):
    indices = ACCESSIBLE_INDICES.get(permission_level, [])

    if not indices:
        print("접근 가능한 인덱스가 없습니다.")
        return

    query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "embedding_text": question
                        }
                    }
                ],
                "filter": []
            }
        },
        "size": 5,
    }

    if employee_id:
        query["query"]["bool"]["filter"].append(
            {
                "term": {
                    "employee_id": employee_id
                }
            }
        )

    response = client.search(
        index=indices,
        body=query,
    )

    total = response["hits"]["total"]["value"]
    print(f"검색 대상 인덱스: {indices}")
    print(f"검색 결과 수: {total}")
    print("-" * 50)

    context = build_context(response["hits"]["hits"])

    print("LLM에 전달할 Context:")
    print(context)
    print("=" * 50)

    for hit in response["hits"]["hits"]:
        print("인덱스:", hit["_index"])
        print("점수:", hit["_score"])
        print("문서ID:", hit["_id"])
        print("사번:", hit["_source"].get("employee_id"))
        print("내용:", hit["_source"].get("embedding_text"))
        print("-" * 50)


if __name__ == "__main__":
    search_bm25(
        question="인사관리 정보 알려줘",
        permission_level=2,
        employee_id="EMP1086",
    )