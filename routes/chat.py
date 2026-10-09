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
from langchain_agents.meta_lc_agent import run_chat_agent
from services.memory_service import extract_and_save_memory
from utils.intent_detector import detect_domain

router = APIRouter(prefix="/chat", tags=["chat"])


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


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 1. Resolve domain
    domain = request.domain
    if domain == "auto":
        domain = detect_domain(request.query)

    # 2. Resolve/create conversation
    if request.conversation_id:
        conversation = get_conversation(db, request.conversation_id)
        if not conversation or conversation.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        conversation = create_conversation(db, user_id=current_user.id, domain=domain)

    # 3. Save user message
    save_message(db, conversation.id, role="user", content=request.query)

    # 4. Run the official LangChain domain agent
    agent_result = run_chat_agent(
        db=db,
        user_id=current_user.id,
        domain=domain,
        query=request.query,
    )

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
        domain=domain,
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
