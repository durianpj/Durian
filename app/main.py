import requests
from fastapi import FastAPI
from pydantic import BaseModel

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