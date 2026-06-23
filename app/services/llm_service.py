import requests
import time
import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_active_llm_model() -> str:
    # 운영 환경에서 어떤 모델을 쓸지 한 곳에서만 결정한다.
    if LLM_PROVIDER == "openai":
        return OPENAI_MODEL

    return OLLAMA_MODEL


def get_active_llm_label() -> str:
    # 디버그 로그에 provider/model을 같이 남길 때 사용한다.
    return f"{LLM_PROVIDER}:{get_active_llm_model()}"


def call_openai_chat_completion(
    prompt: str,
    temperature: float = 0.1,
    max_tokens: int | None = None,
    timeout: int = 60,
) -> str:
    """
    OpenAI GPT API를 호출해서 텍스트 응답을 반환한다.
    """

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": temperature,
    }

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    response = requests.post(
        OPENAI_API_URL,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()

    result = response.json()
    choices = result.get("choices", [])

    if not choices:
        return ""

    return choices[0].get("message", {}).get("content", "").strip()


def call_ollama_completion(
    prompt: str,
    temperature: float = 0.1,
    max_tokens: int | None = None,
    timeout: int = 60,
) -> str:
    """
    Ollama Gemma 모델을 호출해서 텍스트 응답을 반환한다.
    """

    options = {
        "temperature": temperature,
    }

    if max_tokens is not None:
        options["num_predict"] = max_tokens

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": options,
        },
        timeout=timeout,
    )
    response.raise_for_status()

    return response.json().get("response", "").strip()


def call_llm_completion(
    prompt: str,
    temperature: float = 0.1,
    max_tokens: int | None = None,
    timeout: int = 60,
) -> str:
    """
    운영 기본값은 Ollama/Gemma이고, 비교가 필요할 때만 LLM_PROVIDER=openai로 GPT를 사용한다.
    """

    # 호출 지점에서는 provider 차이를 신경 쓰지 않게 숨긴다.
    if LLM_PROVIDER == "openai":
        return call_openai_chat_completion(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    return call_ollama_completion(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def build_rag_prompt(question: str, context: str) -> str:
    """
    사용자 질문과 검색 결과 Context를 합쳐
    LLM에 전달할 프롬프트를 생성한다.
    """

    # RAG 답변은 검증된 context만 넣고, 모델의 임의 추측을 줄인다.
    prompt = f"""
        당신은 인사(HR) 데이터를 기반으로 답변하는 RAG 챗봇입니다.

        [가장 중요한 원칙]

        반드시 [검색 결과]에 포함된 내용만 근거로 답변합니다.
        검색 결과에 없는 내용은 추측하거나 생성하지 않습니다.
        검색 결과만으로 답변이 불가능하면 반드시 다음 문장으로 답변합니다.
        "조건에 맞는 조회 결과가 없습니다."

        [검색 결과 해석 규칙]

        1. 같은 사번(employee_id)을 가진 여러 문서는 같은 직원의 정보로 보고 함께 판단합니다.
        예를 들어 한 문서에는 팀이 있고, 다른 문서에는 주소가 있을 수 있습니다.
        이 경우 사번이 같으면 같은 직원의 정보로 합쳐서 이해합니다.

        2. 검색 결과의 빈 값은 없는 정보로 판단합니다.
        예를 들어 "팀:" 뒤에 값이 없으면 팀 정보가 확인되지 않은 것입니다.

        3. 사용자가 조건을 여러 개 말하면, 모든 조건을 만족하는 직원만 답변합니다.
        예:
        - "브랜드팀 직원 중 이메일이 naver인 사람"
        → 팀이 브랜드팀이고, 이메일에 naver가 포함된 직원만 답변합니다.

        4. 조건을 만족하는 직원이 검색 결과 안에서 확인되지 않으면 다음과 같이 답변합니다.
        "조건에 맞는 조회 결과가 없습니다."

        5. 사용자가 질문한 항목만 답변합니다.
        질문하지 않은 항목은 검색 결과에 있어도 답변하지 않습니다.
        예:
        - 주소를 물으면 주소만 답변합니다.
        - 계좌번호를 물으면 계좌번호만 답변합니다.
        - 이메일을 물으면 이메일만 답변합니다.

        [개인정보 및 권한 규칙]

        사용자의 권한 범위를 벗어나는 정보는 제공하지 않습니다.
        검색 결과에 포함되지 않은 민감정보는 절대 추측하지 않습니다.
        주민등록번호, 계좌번호, 연봉, 주소, 전화번호 등은 질문한 경우에만 답변합니다.

        [답변 형식 규칙]

        절대 인사말을 하지 않습니다.
        절대 자기소개를 하지 않습니다.
        절대 사용자의 질문을 다시 출력하지 않습니다.
        절대 "질문:", "답변:" 같은 제목을 출력하지 않습니다.
        절대 마크다운 제목이나 굵은 글씨를 사용하지 않습니다.
        최종 답변 문장만 출력합니다.
        사용자가 여러 항목을 함께 물으면 한 문장으로 자연스럽게 답변합니다.
        질문의 순서대로 답합니다

        예:
        질문: "내 이름과 부서와 직책과 직급 알려줘"
        답변: "이름은 오민호, 부서는 인사부, 직책은 팀원, 직급은 차장입니다."

        답변은 한국어로 작성합니다.
        답변은 짧고 명확하게 작성합니다.

        부서 질문이면 다음 형식을 사용합니다.
        "OO님의 부서는 OO입니다."

        팀 질문이면 다음 형식을 사용합니다.
        "OO님의 팀은 OO입니다."

        직급 질문이면 다음 형식을 사용합니다.
        "OO님의 직급은 OO입니다."

        이메일 질문이면 다음 형식을 사용합니다.
        "OO님의 이메일은 OO입니다."

        주소 질문이면 다음 형식을 사용합니다.
        "OO님의 주소는 OO입니다."

        내 이름만 묻는 질문이면 다음 형식을 사용합니다.
        "OO입니다."
        

        답변에서 "OO님"이라고 말할 때는 반드시 [검색 결과]의 이름 값을 사용합니다.
        이름 값이 없으면 사번을 이름처럼 사용하지 말고 "해당 사원"이라고 표현합니다.

        연봉, 급여, 보상 등 금액 정보는 천 단위 쉼표와 원 단위를 사용합니다.
        예: 45,000,000원

        부서 종류, 팀 종류, 직급 종류, 직책 종류처럼 "종류/목록/리스트"를 묻는 질문은
        특정 직원 한 명의 정보로 답하지 않습니다.

        검색 결과에 나온 값들을 중복 제거해서 목록으로 답합니다.

        예:
        - 부서 종류 질문 → 검색 결과의 "부서" 값만 모아서 답변
        - 팀 종류 질문 → 검색 결과의 "팀" 값만 모아서 답변
        - 직급 종류 질문 → 검색 결과의 "직급" 값만 모아서 답변
        - 직책 종류 질문 → 검색 결과의 "직책" 값만 모아서 답변

        검색 결과에 여러 직원이 있어도 직원 이름 중심으로 답하지 않습니다.

        부서 종류, 팀 종류, 직급 종류, 직책 종류처럼 "종류/목록/리스트"를 묻는 질문은 특정 직원 한 명의 정보로 답하지 않습니다.

        [목록 질문 요약]이 있으면 그 내용을 우선 사용합니다.

        부서 종류 질문이면 부서명만 중복 없이 목록으로 답변합니다.

        예:
        검색 결과에서 확인되는 부서 종류는 다음과 같습니다.
        - 개발부
        - 영업부
        - 기획부
        - 마케팅부

        "개발"처럼 단어 하나로만 답하지 말고, 반드시 완성된 문장과 목록으로 답변합니다.
        [목록 질문 요약]이 있으면 그 내용만 사용해서 답변합니다.
        직원 이름으로 답하지 말고, 목록에 있는 항목만 답변합니다.

        [답변 문체]
                - 같은 사람 이름과 호칭을 문장마다 반복하지 말고, 첫 줄에 한 번만 쓰십시오.
                - 가능하면 "이름 / 필드: 값" 형태로 짧게 정리하십시오.
                - 기본정보/인사정보 질문은 회사명, 사업장위치 같은 부가 항목보다 핵심 인사 항목 위주로 답하십시오.


        [검색 결과]
        {context}

        [사용자 질문]
        {question}

        [답변]
        """
    return prompt.strip()


def generate_answer(question: str, context: str) -> str:
    """
    설정된 LLM Provider로 Context 기반 답변을 생성한다.
    """

    # start_time = time.perf_counter()
    prompt = build_rag_prompt(question, context)
    # print("[TIME] build_rag_prompt:", f"{time.perf_counter() - start_time:.3f}s")

    # request_start_time = time.perf_counter()
    answer = call_llm_completion(
        prompt=prompt,
        temperature=0.1,
        timeout=60,
    )

    # print(
    #     f"[TIME] generate_answer {LLM_PROVIDER}:",
    #     f"{time.perf_counter() - request_start_time:.3f}s",
    # )

    # print("[TIME] generate_answer total:", f"{time.perf_counter() - start_time:.3f}s")

    return answer
