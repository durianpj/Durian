import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.services.llm_service import call_llm_completion, get_active_llm_label
from app.services.rag_chat_service import handle_rag_chat


app = FastAPI(title="Durian HR RAG Chatbot")
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    question: str


class RagChatRequest(BaseModel):
    question: str
    employee_id: str


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    예상하지 못한 서버 오류를 통일된 JSON 응답으로 처리한다.
    HTTPException(400 등)은 FastAPI 기본 처리 흐름을 따른다.
    """

    logger.exception(
        "Unhandled exception during request: %s %s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "answer": "",
            "permission": {
                "allowed": False,
                "permission_level": None,
                "required_level": None,
            },
            "sources": [],
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "서버 처리 중 오류가 발생했습니다.",
            },
        },
    )


@app.post("/rag-chat")
def rag_chat(request: RagChatRequest):
    """
    HR RAG 챗봇의 메인 엔드포인트다.
    main.py는 요청 검증과 서비스 호출만 맡고,
    실제 처리 흐름은 rag_chat_service로 넘긴다.

    실행:
    uvicorn main:app --reload --port 8000

    요청 주소:
    localhost:8000/rag-chat
    """

    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="질문을 입력해주세요.",
        )

    if not request.employee_id or not request.employee_id.strip():
        raise HTTPException(
            status_code=400,
            detail="employee_id를 입력해주세요.",
        )

    return handle_rag_chat(
        question=request.question,
        employee_id=request.employee_id,
    )


@app.get("/")
def root():
    return {"message": "Durian RAG API Running"}


@app.post("/chat")
def chat(request: ChatRequest):
    """
    검색 없이 현재 설정된 LLM에 바로 질문하는 단순 테스트용 엔드포인트다.
    실제 HR RAG 흐름은 /rag-chat에서 처리한다.
    """

    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="질문을 입력해주세요.",
        )

    answer = call_llm_completion(
        prompt=request.question,
        temperature=0.1,
    )

    return {
        "question": request.question,
        "answer": answer,
        "model": get_active_llm_label(),
    }