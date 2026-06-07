import json

file_path = "data/기본인사정보_정제.jsonl"

with open(file_path, "r", encoding="utf-8") as f:
    first_line = f.readline()
    doc = json.loads(first_line)

print("=== 필드 목록 ===")
print(doc.keys())

print("\n=== 문서 예시 ===")
for key, value in doc.items():
    if key == "embedding_vector":
        print(key, f"벡터 길이: {len(value)}")
        print("앞 5개 값:", value[:5])
    else:
        print(key, value)

print("\n=== 필수 확인 ===")
print("employee_id 있음:", "employee_id" in doc)
print("doc_type 있음:", "doc_type" in doc)
print("permission_level 있음:", "permission_level" in doc)
print("embedding_text 있음:", "embedding_text" in doc)
print("embedding_vector 있음:", "embedding_vector" in doc)

if "embedding_vector" in doc:
    print("임베딩 차원:", len(doc["embedding_vector"]))