from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

text = "김도윤은 인사부 인사운영팀 소속 사원이며 현재 재직 중입니다."

embedding = model.encode(text)

print(f"벡터 차원: {len(embedding)}")
print(embedding[:5])