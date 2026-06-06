import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"


def build_rag_prompt(question: str, context: str) -> str:
    """
    사용자 질문과 검색 결과 Context를 합쳐
    LLM에 전달할 프롬프트를 생성한다.
    """

    prompt = f"""
        당신은 인사(HR) 데이터를 기반으로 답변하는 RAG 챗봇입니다.

        [답변 규칙]

        반드시 [검색 결과]에 포함된 내용만 근거로 답변합니다.
        검색 결과에 없는 내용은 추측하거나 생성하지 않습니다.
        검색 결과가 비어 있거나 질문과 관련된 정보가 없는 경우 다음과 같이 답변합니다.
        "조회된 데이터에서 확인할 수 없습니다."
        사용자의 권한 범위를 벗어나는 정보는 제공하지 않습니다.
        연봉, 급여, 보상 등 금액 정보는 천 단위 쉼표와 원 단위를 사용합니다.
        예: 45,000,000원
        답변은 한국어로 작성합니다.
        답변은 짧고 명확하게 작성합니다.
        가능하면 답변 마지막에 출처를 표시합니다.
        검색 결과만으로 답변이 불가능한 경우 추가 추론이나 가정은 하지 않습니다.

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
