import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"


def build_rag_prompt(question: str, context: str) -> str:
    """
    사용자 질문과 검색 결과 Context를 합쳐
    LLM에 전달할 프롬프트를 생성한다.
    """

    prompt = f"""
        당신은 인사 데이터를 기반으로 답변하는 RAG 챗봇입니다.

        아래 [검색 결과]에 있는 내용만 근거로 답변하세요.
        검색 결과에 없는 내용은 추측하지 말고 "확인 가능한 정보가 없습니다."라고 답변하세요.
        민감 정보는 권한이 허용된 경우에만 답변한다고 가정합니다.
        답변은 반드시 한국어로 작성해라.
        답변은 짧고 정확하게 작성해라.
        금액은 천 단위 쉼표와 원 단위를 포함해서 작성해라.

        [검색 결과]
        {context}

        [사용자 질문]
        {question}

        [답변]
    """
    return prompt.strip()


def generate_answer(question: str, context: str) -> str:
    """
    Ollama gemma3:4b 모델을 호출하여
    Context 기반 답변을 생성한다.
    """

    prompt = build_rag_prompt(question, context)

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
        },
        timeout=60,
    )

    response.raise_for_status()

    result = response.json()

    return result.get("response", "").strip()
