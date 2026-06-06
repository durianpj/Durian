import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.services.llm_service import generate_answer
from app.services.question_service import (
    is_department_list_question,
    is_department_members_question,
    is_phone_number_question,
    is_self_question,
    is_supervisor_question,
)
from app.services.hybrid_search_service import (
    build_context,
    extract_employee_name_from_source,
    extract_phone_number_from_source,
    get_department_members,
    get_department_types,
    get_supervisors,
    get_user_permission_level,
    search_hybrid,
)


# =========================
# FastAPI 앱 생성
# =========================

app = FastAPI(title="Durian HR RAG Chatbot")


# =========================
# LLM 단독 테스트 요청 모델
# =========================

class ChatRequest(BaseModel):
    # 사용자가 입력한 질문
    question: str


# =========================
# RAG 챗봇 요청 모델
# =========================

class RagChatRequest(BaseModel):
    # 사용자가 입력한 질문
    question: str

    # 요청한 사용자의 사번
    # 이 사번을 기준으로 권한 레벨을 자동 계산한다.
    employee_id: str


# =========================
# 기본 상태 확인 API
# =========================

@app.get("/")
def root():
    return {"message": "Durian RAG API Running"}


# =========================
# LLM 단독 호출 API
# =========================

@app.post("/chat")
def chat(request: ChatRequest):
    """
    RAG 검색 없이 gemma3:4b 모델만 단독으로 호출하는 테스트 API이다.
    """

    # 질문이 비어 있으면 예외 처리
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="질문을 입력해주세요.",
        )

    # Ollama gemma3:4b 모델 호출
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma3:4b",
            "prompt": request.question,
            "stream": False,
        },
    )

    # Ollama 응답을 JSON으로 변환
    result = response.json()

    return {
        "question": request.question,
        "answer": result["response"],
        "model": "gemma3:4b",
    }


# =========================
# RAG 챗봇 API
# =========================

@app.post("/rag-chat")
def rag_chat(request: RagChatRequest):
    """
    권한 기반 RAG 챗봇 API이다.

    처리 흐름:
    1. 질문과 employee_id 입력값을 검증한다.
    2. employee_id 기준으로 사용자 권한 레벨을 자동 계산한다.
    3. 질문 내용 기준으로 필요한 정보 레벨을 판단한다.
    4. 사용자 권한이 부족하면 차단한다.
    5. 본인 질문이면 employee_id로 검색 대상을 제한한다.
    6. OpenSearch에서 BM25 + 벡터 검색을 수행한다.
    7. RRF 방식으로 병합된 검색 결과를 Context로 만든다.
    8. Context와 질문을 gemma3:4b에 전달한다.
    9. 답변과 출처 데이터를 반환한다.
    """

    # =========================
    # 1. 질문 미입력 예외 처리
    # =========================

    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="질문을 입력해주세요.",
        )

    # =========================
    # 2. 사번 미입력 예외 처리
    # =========================

    if not request.employee_id or not request.employee_id.strip():
        raise HTTPException(
            status_code=400,
            detail="employee_id를 입력해주세요.",
        )

    # =========================
    # 3. employee_id 기준 권한 레벨 계산
    # =========================
    # get_user_permission_level() 내부에서
    # department_level과 job_grade_level 중 더 높은 값을 permission_level로 사용한다.

    permission_level = get_user_permission_level(request.employee_id)

    # 사번으로 사용자를 찾지 못한 경우
    if permission_level is None:
        raise HTTPException(
            status_code=404,
            detail="사용자 사번을 찾을 수 없습니다.",
        )

    # 계산된 권한 레벨이 1, 2, 3 중 하나인지 확인
    if permission_level not in [1, 2, 3]:
        raise HTTPException(
            status_code=400,
            detail="계산된 permission_level이 올바르지 않습니다.",
        )

    # =========================
    # 4. 질문 내용 기준 필요 권한 레벨 판단
    # =========================
    # 예:
    # - 기본 정보: 1
    # - 연봉, 성과, 평가: 2
    # - 주소, 계좌번호, 징계: 3

    required_level = get_required_level(request.question)

    if is_department_list_question(request.question):
        departments = get_department_types()

        if departments:
            answer = "부서 종류는 " + ", ".join(departments) + "입니다."
        else:
            answer = "조회된 데이터에서 확인할 수 없습니다."

        return {
            "success": True,
            "answer": answer,
            "permission": {
                "allowed": True,
                "employee_id": request.employee_id,
                "permission_level": permission_level,
                "required_level": required_level,
            },
            "sources": [],
            "model_type": "rule-based",
        }

    if is_department_members_question(request.question):
        members = get_department_members(request.question)

        if members:
            member_names = [
                f"{member['name']} {member['position']}"
                for member in members
            ]
            answer = (
                f"해당 구성원은 총 {len(members)}명입니다. "
                + ", ".join(member_names)
                + "입니다."
            )
        else:
            answer = "조회된 데이터에서 확인할 수 없습니다."

        return {
            "success": True,
            "answer": answer,
            "permission": {
                "allowed": True,
                "employee_id": request.employee_id,
                "permission_level": permission_level,
                "required_level": required_level,
            },
            "sources": [
                {
                    "index": member["index"],
                    "_id": member["_id"],
                    "employee_id": member["employee_id"],
                    "department": member["department"],
                    "position": member["position"],
                    "score": member["score"],
                }
                for member in members
            ],
            "model_type": "rule-based",
        }

    # =========================
    # 5. 권한 부족 시 차단
    # =========================

    # 7. 본인 질문 여부 판단
    # "내 연봉", "나의 주소"처럼 본인 데이터 조회면 본인 조회로 본다.
    is_self = is_self_question(request.question)

    if is_self and is_supervisor_question(request.question):
        supervisors = get_supervisors(request.employee_id)

        if supervisors:
            supervisor_names = [
                f"{supervisor['name']} {supervisor['position']}"
                for supervisor in supervisors
            ]
            answer = "상사는 " + ", ".join(supervisor_names) + "입니다."
        else:
            answer = "조회된 데이터에서 확인할 수 없습니다."

        return {
            "success": True,
            "answer": answer,
            "permission": {
                "allowed": True,
                "employee_id": request.employee_id,
                "permission_level": permission_level,
                "required_level": required_level,
            },
            "sources": [
                {
                    "index": supervisor["index"],
                    "_id": supervisor["_id"],
                    "employee_id": supervisor["employee_id"],
                    "department": supervisor["department"],
                    "position": supervisor["position"],
                    "score": supervisor["score"],
                }
                for supervisor in supervisors
            ],
            "model_type": "rule-based",
        }

    # 8. 사용자 권한보다 높은 정보 요청 시 차단
    # 단, 본인 정보 조회는 본인이 조회 가능하도록 허용한다.
    if not is_self and permission_level < required_level:
        return {
            "success": False,
            "answer": "해당 정보에 접근할 권한이 없습니다.",
            "permission": {
                "allowed": False,
                "employee_id": request.employee_id,
                "permission_level": permission_level,
                "required_level": required_level,
            },
            "sources": [],
            "model_type": "gemma3:4b",
        }

    # =========================
    # 6. 본인 정보 조회 여부 판단
    # =========================
    # "내 부서 알려줘", "내 연봉 알려줘" 같은 질문이면
    # 요청한 employee_id로 검색 결과를 제한한다.

    search_employee_id = None

    if is_self:
        search_employee_id = request.employee_id


    # =========================
    # 7. 검색에 사용할 권한 레벨 결정
    # =========================
    # 기본적으로는 요청자의 permission_level을 사용한다.
    # 단, 본인 조회인 경우에는 본인 데이터 조회를 허용하기 위해
    # 질문에서 필요한 required_level까지 검색 범위를 넓힌다.
    #
    # 예:
    # permission_level = 1
    # required_level = 2
    # 질문 = "내 연봉 알려줘"
    # → 본인 조회이므로 search_permission_level = 2

    search_permission_level = permission_level

    if is_self:
        search_permission_level = max(permission_level, required_level)


    # =========================
    # 8. OpenSearch 하이브리드 검색 실행
    # =========================

    search_hits = search_hybrid(
        question=request.question,
        permission_level=search_permission_level,
        employee_id=search_employee_id,
        size=5,
    )

    if is_phone_number_question(request.question):
        phone_answer = None

        for hit in search_hits:
            source = hit.get("_source", {})
            phone_number = extract_phone_number_from_source(source)

            if phone_number:
                employee_name = extract_employee_name_from_source(source)
                phone_answer = f"{employee_name}님의 전화번호는 {phone_number}입니다."
                break

        if phone_answer is None:
            phone_answer = "조회된 데이터에서 확인할 수 없습니다."

        return {
            "success": True,
            "answer": phone_answer,
            "permission": {
                "allowed": True,
                "employee_id": request.employee_id,
                "permission_level": permission_level,
                "required_level": required_level,
            },
            "sources": [
                {
                    "index": hit["_index"],
                    "_id": hit["_id"],
                    "employee_id": hit.get("_source", {}).get("employee_id"),
                    "department": hit.get("_source", {}).get("department"),
                    "position": hit.get("_source", {}).get("position"),
                    "score": hit.get("_score"),
                }
                for hit in search_hits
            ],
            "model_type": "rule-based",
        }
    # =========================
    # 9. 검색 결과를 LLM Context로 변환
    # =========================

    context = build_context(search_hits)

    # =========================
    # 10. gemma3:4b 답변 생성
    # =========================

    answer = generate_answer(
        question=request.question,
        context=context,
    )

    # =========================
    # 11. 응답에 포함할 출처 데이터 구성
    # =========================
    # 검색 결과의 OpenSearch 인덱스, 문서 ID, 사번, 부서, 직급/직책, 점수를 반환한다.

    sources = []

    for hit in search_hits:
        source = hit.get("_source", {})

        sources.append(
            {
                "index": hit["_index"],
                "_id": hit["_id"],
                "employee_id": source.get("employee_id"),
                "department": source.get("department"),
                "position": source.get("position"),
                "score": hit.get("_score"),
            }
        )

    # =========================
    # 12. 최종 응답 반환
    # =========================

    return {
        "success": True,
        "answer": answer,
        "permission": {
            "allowed": True,
            "employee_id": request.employee_id,
            "permission_level": permission_level,
            "required_level": required_level,
        },
        "sources": sources,
        "model_type": "gemma3:4b",
    }


# =========================
# 질문 내용 기준 필요 권한 레벨 판단
# =========================

def get_required_level(question: str) -> int:
    """
    질문에 포함된 키워드를 기준으로 필요한 권한 레벨을 간단히 판단한다.

    실제 접근 제어는 OpenSearch 검색 대상 인덱스 제한과
    검색 필터를 통해 수행한다.

    이 함수는 민감 정보 요청을 사전에 차단하기 위한 보조 검증이다.
    """

    # 3레벨 권한이 필요한 민감 정보 키워드
    level_3_keywords = [
        "주소",
        "주민번호",
        "주민등록번호",
        "계좌번호",
        "은행",
        "급여은행",
        "징계",
        "퇴직",
    ]

    # 2레벨 권한이 필요한 정보 키워드
    level_2_keywords = [
        "연봉",
        "급여",
        "성과",
        "평가",
        "TOEIC",
        "토익",
        "자격증",
        "포상",
    ]

    # 3레벨 키워드가 질문에 있으면 3 반환
    for keyword in level_3_keywords:
        if keyword in question:
            return 3

    # 2레벨 키워드가 질문에 있으면 2 반환
    for keyword in level_2_keywords:
        if keyword in question:
            return 2

    # 일반 기본 정보는 1레벨로 판단
    return 1
