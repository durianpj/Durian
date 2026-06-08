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
        질문한 항목만 말합니다.
        부서 질문이면 "OO님의 부서는 OO입니다." 형식으로 답변하세요.
        직급 질문이면 "OO님의 직급은 OO입니다." 형식으로 답변하세요.
        답변에서 "OO님"이라고 말할 때는 반드시 [검색 결과]의 이름 값을 사용합니다.
        이름 값이 없으면 사번을 이름처럼 사용하지 말고 "해당 사원"이라고 표현합니다.    
        사용자가 질문한 항목 하나만 답변합니다.
        질문하지 않은 항목은 검색 결과에 있어도 절대 답변하지 않습니다.
        예를 들어 주소를 물으면 주소만 답변하고, 전화번호/이메일/계좌번호는 말하지 않습니다.
        예를 들어 계좌번호를 물으면 계좌번호만 답변하고, 급여/은행/주소는 말하지 않습니다.

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
