import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.services.llm_service import generate_answer

app = FastAPI(title="Durian HR RAG Chatbot")


class ChatRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "Durian RAG API Running"}


@app.post("/chat")
def chat(request: ChatRequest):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma3:4b",
            "prompt": request.question,
            "stream": False,
        },
    )

    result = response.json()

    return {
        "question": request.question,
        "answer": result["response"],
        "model": "gemma3:4b",
    }

class RagChatRequest(BaseModel):
    question: str
    employee_id: str | None = None
    permission_level: int = 1


@app.post("/rag-chat")
def rag_chat(request: RagChatRequest):
    """
    RAG 챗봇 API 1차 버전
    현재는 OpenSearch 검색 결과 대신 임의 Context로 LLM 응답 생성 흐름을 검증한다.
    """

    # 1. 질문 미입력 예외 처리
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="질문을 입력해주세요."
        )
    # 권한 레벨 입력값 검증
    if request.permission_level not in [1, 2, 3]:
        raise HTTPException(
            status_code=400,
            detail="permission_level은 1, 2, 3 중 하나여야 합니다."
        )

    # 질문 내용 기준 필요 권한 레벨 확인
    required_level = get_required_level(request.question)

    # 사용자 권한보다 높은 레벨의 정보 요청 시 차단
    if request.permission_level < required_level:
        return {
            "question": request.question,
            "answer": "해당 정보에 접근할 권한이 없습니다.",
            "employee_id": request.employee_id,
            "permission_level": request.permission_level,
            "required_level": required_level,
            "sources": [],
            "model": "gemma3:4b"
        }

    # 2. 권한 레벨 예외 처리
    if request.permission_level not in [1, 2, 3]:
        raise HTTPException(
            status_code=400,
            detail="permission_level은 1, 2, 3 중 하나여야 합니다."
        )

    # 3. 임의 Context
    # TODO: 이후 OpenSearch 검색 결과 Context로 교체 예정
    context = """
[출처: hr_basic_1 / BAS1_01086]
사번: EMP1086 / 이름: 구다은 / 부서: 인사부 / 직급: 대리 / 이메일: test@example.com
"""

    # 4. 검색 결과 없음 예외 처리용
    if not context.strip():
        return {
            "question": request.question,
            "answer": "관련 정보를 찾을 수 없습니다.",
            "employee_id": request.employee_id,
            "permission_level": request.permission_level,
            "sources": [],
            "model": "gemma3:4b"
        }

    answer = generate_answer(
        question=request.question,
        context=context
    )

    return {
        "question": request.question,
        "answer": answer,
        "employee_id": request.employee_id,
        "permission_level": request.permission_level,
        "sources": [
            {
                "index": "hr_basic_1",
                "doc_id": "BAS1_01086"
            }
        ],
        "model": "gemma3:4b"
    }


def get_required_level(question: str) -> int:
    """
    질문 내용에 따라 필요한 데이터 보안 레벨을 간단히 판단한다.
    실제 검색 연동 전까지는 키워드 기준으로 권한 검증을 수행한다.
    """

    level_3_keywords = [
        "주소", "주민번호", "주민등록번호", "계좌번호", "징계", "퇴직"
    ]

    level_2_keywords = [
        "연봉", "급여", "성과", "평가", "TOEIC", "토익", "자격증", "포상"
    ]

    for keyword in level_3_keywords:
        if keyword in question:
            return 3

    for keyword in level_2_keywords:
        if keyword in question:
            return 2

    return 1