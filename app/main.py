import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.services.rag_chat_service import handle_rag_chat


app = FastAPI(title="Durian HR RAG Chatbot")


class ChatRequest(BaseModel):
    question: str


class RagChatRequest(BaseModel):
    question: str
    employee_id: str


@app.get("/")
def root():
    return {"message": "Durian RAG API Running"}


@app.post("/chat")
def chat(request: ChatRequest):
    """
    검색 없이 Ollama 모델에 바로 질문하는 단순 테스트용 엔드포인트다.
    실제 HR RAG 흐름은 /rag-chat에서 처리한다.
    """

    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="질문을 입력해주세요.",
        )

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


@app.post("/rag-chat")
def rag_chat(request: RagChatRequest):
    """
    HR RAG 챗봇의 메인 엔드포인트다.
    main.py는 요청 검증과 서비스 호출만 맡고, 실제 처리 흐름은 rag_chat_service로 넘긴다.
    uvicorn main:app --reload --port 8000
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
