from app.services.question_service import extract_employee_name, is_self_question


conversation_sessions = {}


def resolve_question_with_session(
    question: str,
    employee_id: str,
    session_id: str | None,
) -> str:
    """
    세션에 저장된 이전 조회 대상을 사용해 후속 질문을 보정한다.
    """

    if not session_id:
        return question

    session_key = f"{employee_id}:{session_id}"
    session = conversation_sessions.setdefault(session_key, {})
    target_name = extract_employee_name(question)

    if target_name:
        session["target_name"] = target_name
        return question

    previous_target_name = session.get("target_name")

    if previous_target_name and not is_self_question(question):
        return f"{previous_target_name} {question}"

    return question
