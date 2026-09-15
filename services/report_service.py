"""
app/services/report_service.py

Generates a PDF report summarizing a user's profile + recent advice for a
domain (optionally scoped to one conversation), saves the file under
REPORTS_DIR, and records the path in the `reports` table.
"""
from typing import Optional
from sqlalchemy.orm import Session
from datetime import datetime
import json
import time

from core.config import settings
from models.report import Report
from services.profile_service import get_full_profile
from services.memory_service import retrieve_memory
from services.conversation_service import get_conversation_history
from utils.pdf_generator import generate_pdf_report
from core.llm_client import call_llm


REPORT_PROMPT_TEMPLATE = """You are an expert AI Advisory Assistant for the {domain} domain.
Your task is to analyze the provided user profile, memories, and recent conversation context to generate a comprehensive advisory report.

You MUST respond with a valid JSON object matching this exact structure (no markdown fences):
{{
    "ai_recommendation": "The primary, most important recommendation for the user (2-3 sentences max).",
    "detailed_analysis": {{
        "Analysis Area 1 (e.g. Budget Analysis)": "Detailed paragraph analyzing this area...",
        "Analysis Area 2 (e.g. Savings Analysis)": "Detailed paragraph analyzing this area..."
    }},
    "explainability": {{
        "Why this recommendation?": "...",
        "Calculations Performed": "...",
        "Decision Reasoning": "..."
    }},
    "key_metrics": {{
        "Metric 1": "Value",
        "Metric 2": "Value"
    }},
    "action_plan": [
        "First actionable step.",
        "Second actionable step."
    ],
    "risks": [
        "Potential risk 1",
        "Potential risk 2"
    ],
    "assumptions": [
        "Assumption made by AI 1"
    ],
    "missing_information": [
        "Data that would improve this report 1"
    ],
    "confidence_level": "High/Medium/Low",
    "confidence_reason": "Reason for the confidence level.",
    "disclaimer": "Domain specific disclaimer."
}}

CONTEXT:
Profile Data:
{profile_data}

Key Facts on File:
{memory_data}

Recent Advice Given:
{history_data}
"""


def _normalize_report_output(llm_output, domain: str, profile_snapshot: dict) -> dict:
    """Adapt model responses to the stable schema consumed by the PDF renderer."""
    output = llm_output if isinstance(llm_output, dict) else {}
    recommendation = output.get("ai_recommendation") or output.get("recommendation") or output.get("summary")

    if not recommendation or len(str(recommendation).strip()) < 10:
        recommendation = (
            f"Use the available {domain} profile information to define a focused next step, "
            "then review and update this report as more information becomes available."
        )

    detailed_analysis = output.get("detailed_analysis") or {}
    if not isinstance(detailed_analysis, dict):
        detailed_analysis = {"Analysis": str(detailed_analysis)}
    if not detailed_analysis:
        detailed_analysis = {
            "Profile-based analysis": (
                "The report was generated from the profile information currently on file. "
                "Add more domain-specific details to make the next analysis more precise."
            )
        }

    explainability = output.get("explainability") or {}
    if not isinstance(explainability, dict):
        explainability = {"Reasoning": str(explainability)}
    if not explainability:
        explainability = {"Reasoning": "Recommendations are based on the supplied profile and available context."}

    normalized = dict(output)
    normalized.update({
        "ai_recommendation": str(recommendation),
        "detailed_analysis": detailed_analysis,
        "explainability": explainability,
        "key_metrics": output.get("key_metrics") if isinstance(output.get("key_metrics"), dict) else profile_snapshot,
        "action_plan": output.get("action_plan") if isinstance(output.get("action_plan"), list) else [
            "Review the recommendation against your current goals.",
            "Add missing profile details and regenerate the report for a more specific plan.",
        ],
        "risks": output.get("risks") if isinstance(output.get("risks"), list) else [],
        "assumptions": output.get("assumptions") if isinstance(output.get("assumptions"), list) else [
            "The report uses only the profile and context available at generation time."
        ],
        "missing_information": output.get("missing_information") if isinstance(output.get("missing_information"), list) else [],
        "confidence_level": output.get("confidence_level") or "Low",
        "confidence_reason": output.get("confidence_reason") or "Some report sections were not returned by the model.",
    })
    return normalized

def generate_report(
    db: Session, user_id: str, user_name: str, domain: str, conversation_id: Optional[str] = None
) -> Report:
    start_time = time.time()
    profile = get_full_profile(db, user_id)
    memories = retrieve_memory(db, user_id, category=domain, limit=10)

    # 1. Gather Context
    domain_profile = profile.get(domain)
    profile_snapshot = {}
    profile_data_str = "No data on file."
    
    if domain_profile:
        profile_lines = []
        for key, value in vars(domain_profile).items():
            if key.startswith("_") or key in ("id", "user_id", "updated_at") or value is None:
                continue
            display_key = key.replace('_', ' ').title()
            profile_snapshot[display_key] = value
            profile_lines.append(f"{display_key}: {value}")
        if profile_lines:
            profile_data_str = "\\n".join(profile_lines)

    memory_data_str = "No key facts recorded."
    if memories:
        memory_data_str = "\\n".join(f"- {m.memory_text}" for m in memories)

    history_data_str = "No advice recorded yet."
    if conversation_id:
        history = get_conversation_history(db, conversation_id)
        assistant_turns = [m.content for m in history if m.role == "assistant"]
        if assistant_turns:
            history_data_str = "\\n\\n".join(assistant_turns[-5:])

    # 2. Call LLM for Report Content
    system_prompt = REPORT_PROMPT_TEMPLATE.format(
        domain=domain.capitalize(),
        profile_data=profile_data_str,
        memory_data=memory_data_str,
        history_data=history_data_str
    )
    
    # We pass a simple user message to trigger generation
    llm_output = call_llm(
        system_prompt,
        "Generate the advisory report based on the provided context.",
        temperature=0.5,
        max_tokens=2500,
    )

    # 3. Prepare Report Data Structure
    end_time = time.time()
    response_time = f"{(end_time - start_time):.2f}s"
    now = datetime.now()

    report_data = {
        "report_id": f"REP-{int(time.time())}",
        "username": user_name,
        "user_id": user_id,
        "domain": domain,
        "intent_detected": f"{domain.capitalize()} Advisory Generation",
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "response_time": response_time,
        "version": "1.1",
        "ai_model": "llama-3.3-70b-versatile",
        "conversation_id": conversation_id,
        "profile_snapshot": profile_snapshot,
        "llm_output": _normalize_report_output(llm_output, domain, profile_snapshot)
    }

    # 4. Generate PDF
    file_path = generate_pdf_report(
        output_dir=settings.REPORTS_DIR,
        report_data=report_data
    )

    report = Report(
        user_id=user_id,
        report_name=f"{domain.capitalize()} Advisory Report",
        file_path=file_path,
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def list_reports(db: Session, user_id: str):
    return (
        db.query(Report)
        .filter(Report.user_id == user_id)
        .order_by(Report.generated_at.desc())
        .all()
    )


def get_report(db: Session, report_id: str, user_id: str):
    return (
        db.query(Report)
        .filter(Report.id == report_id, Report.user_id == user_id)
        .first()
    )


def delete_report(db: Session, report_id: str, user_id: str) -> bool:
    import os
    report = get_report(db, report_id, user_id)
    if not report:
        return False
    try:
        if os.path.exists(report.file_path):
            os.remove(report.file_path)
    except Exception:
        pass
    db.delete(report)
    db.commit()
    return True
