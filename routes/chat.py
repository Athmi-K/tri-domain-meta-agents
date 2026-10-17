"""
app/routes/chat.py

The core conversational endpoint. Flow per request:
  1. Resolve/create the conversation
  2. Save the user's message
  3. Detect domain (if "auto")
  4. Run the selected LangChain domain agent with the stored profile
  5. Save the assistant's reply
  6. Extract + save any new long-term memory from the user's message
  7. Return the answer
"""
from typing import List
import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.database import get_db
from core.security import get_current_user
from models.user import User
from schemas.chat import ChatRequest, ChatResponse, ConversationOut, ConversationSummary
from services.conversation_service import (
    create_conversation,
    get_conversation,
    save_message,
    get_conversation_history,
    get_recent_conversations,
)
from langchain_agents.meta_lc_agent import run_chat_agent, run_chat_agents
from core.intent_detector import detect_intent
from services.memory_service import extract_and_save_memory

router = APIRouter(prefix="/chat", tags=["chat"])


def _chat_domain_label(
    requested_domain: str, detected_domains: list[str] | None, result: dict
) -> str:
    """Return the stable domain label exposed by the chat API."""
    status = result.get("status")
    if status == "blocked":
        return "blocked"
    if status == "out_of_scope":
        return "general"
    if detected_domains and len(detected_domains) > 1:
        return "multidomain"
    return requested_domain


def _answer_from_agent_result(result: dict) -> str:
    """Convert LangChain or legacy agent output into the chat answer string."""
    candidate = (
        result.get("recommendation")
        or result.get("message")
        or result.get("summary")
        or result.get("content")
    )
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    if candidate is not None:
        return json.dumps(candidate, ensure_ascii=False)
    raise HTTPException(
        status_code=502,
        detail="The domain agent returned no answer. Please try again.",
    )


def _is_income_recall_query(query: str) -> bool:
    normalized = query.lower()
    return (
        "what income did i tell you" in normalized
        or "what did i tell you about my income" in normalized
        or "what income do you remember" in normalized
    )


def _income_recall_result(history) -> dict:
    for message in reversed(history):
        if message.role != "user":
            continue
        normalized = message.content.lower()
        if any(
            phrase in normalized
            for phrase in (
                "no income",
                "dont have income",
                "don't have income",
                "do not have income",
                "without income",
                "not earning",
            )
        ):
            return {
                "status": "success",
                "message": (
                    "You told me that you currently have no income."
                ),
                "reason": "Recalled from the current conversation",
                "confidence": 1.0,
                "confidence_level": "High",
                "tools_used": [],
            }
    return {
        "status": "success",
        "message": (
            "I do not have a previous message in this conversation stating "
            "your income."
        ),
        "reason": "No matching income statement was found in conversation history",
        "confidence": 0.5,
        "confidence_level": "Medium",
        "tools_used": [],
    }


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 1. Resolve domain
    domain = request.domain
    detected_domains = None
    if domain == "auto":
        detected_domains = detect_intent(request.query).get("domains", [])
        domain = detected_domains[0] if detected_domains else "general"

    # 2. Resolve/create conversation
    if request.conversation_id:
        conversation = get_conversation(db, request.conversation_id)
        if not conversation or conversation.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        conversation = create_conversation(db, user_id=current_user.id, domain=domain)

    previous_history = get_conversation_history(db, conversation.id)

    # 3. Save user message
    save_message(db, conversation.id, role="user", content=request.query)

    # 4. Run the official LangChain domain agent
    if _is_income_recall_query(request.query):
        agent_result = _income_recall_result(previous_history)
    elif detected_domains and len(detected_domains) > 1:
        agent_result = run_chat_agents(
            db=db,
            user_id=current_user.id,
            domains=detected_domains,
            query=request.query,
        )
    else:
        agent_result = run_chat_agent(
            db=db,
            user_id=current_user.id,
            domain=domain,
            query=request.query,
        )

    response_domain = _chat_domain_label(domain, detected_domains, agent_result)
    if conversation.domain != response_domain:
        conversation.domain = response_domain
        db.commit()
    answer = _answer_from_agent_result(agent_result)

    # 5. Save assistant reply
    save_message(db, conversation.id, role="assistant", content=answer)

    # 6. Load the full conversation history so the client can use backend timestamps
    history = get_conversation_history(db, conversation.id)

    # 7. Extract + save long-term memory from the user's message (best-effort)
    memory_saved = []
    if agent_result.get("status") != "blocked":
        try:
            memory = extract_and_save_memory(db, current_user.id, request.query)
            if memory:
                memory_saved.append(memory.memory_text)
        except Exception:
            pass  # memory extraction must never break the chat response

    return ChatResponse(
        conversation_id=conversation.id,
        domain=response_domain,
        answer=answer,
        reason=agent_result.get("reason"),
        confidence=agent_result.get("confidence"),
        confidence_level=agent_result.get("confidence_level"),
        memory_saved=memory_saved,
        sources=agent_result.get("sources", []),
        tools_used=agent_result.get("tools_used", []),
        tool_outputs=agent_result.get("tool_outputs"),
        explainability=agent_result.get("explainability"),
        messages=history,
    )


@router.get("/history", response_model=List[ConversationSummary])
def chat_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_recent_conversations(db, current_user.id)


@router.get("/conversation/{conversation_id}", response_model=ConversationOut)
def chat_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conversation = get_conversation(db, conversation_id)
    if not conversation or conversation.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation
