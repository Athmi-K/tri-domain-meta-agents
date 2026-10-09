"""
Meta LangChain Agent
Routes queries to the correct LangChain domain agents.
Drop-in replacement for the existing meta_agent in main.py.
"""
import agents.career_agent as career_original
import agents.health_agent as health_original
import agents.finance_agent as finance_original
from sqlalchemy.orm import Session
from types import SimpleNamespace
from langchain_agents.career_lc_agent import run as career_run
from langchain_agents.health_lc_agent import run as health_run
from langchain_agents.finance_lc_agent import run as finance_run
from core.intent_detector import detect_intent
from core.safety_layer import check_safety, check_relevance
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=3)

FALLBACK_AGENTS = {
    "career": career_original.run,
    "health": health_original.run,
    "finance": finance_original.run,
}


def _build_chat_explainability(domain: str, request, result: dict) -> dict:
    """Build useful explainability when a LangChain agent returns no details."""
    data_used = ["Current user query"]
    decision_factors = []

    if domain == "career":
        profile_fields = {
            "Target role": getattr(request, "target_role", None),
            "Current skills": getattr(request, "current_skills", None),
            "Experience level": getattr(request, "experience_level", None),
            "Location": getattr(request, "location", None),
            "Timeline": getattr(request, "timeline_months", None),
        }
        data_used.extend(
            f"{label}: {value}"
            for label, value in profile_fields.items()
            if value not in (None, "", [], {})
        )
        next_steps = [
            "Compare the recommendation with your target role and current skills.",
            "Provide your resume or a specific target role for a more tailored analysis.",
        ]
    elif domain == "health":
        profile_fields = {
            "Age": getattr(request, "age", None),
            "Weight": getattr(request, "weight_kg", None),
            "Height": getattr(request, "height_cm", None),
            "Fitness goal": getattr(request, "fitness_goal", None),
            "Sleep hours": getattr(request, "sleep_hours", None),
        }
        data_used.extend(
            f"{label}: {value}"
            for label, value in profile_fields.items()
            if value not in (None, "", [], {})
        )
        next_steps = [
            "Use the recommendation alongside your current health profile.",
            "Provide specific symptoms, goals, or medical context for a more tailored answer.",
        ]
    else:
        data_used.append("Persisted finance profile data available to the finance agent")
        next_steps = [
            "Review the recommendation against your income, expenses, and financial goals.",
            "Update your finance profile if any values are missing or outdated.",
        ]

    reason = result.get("reason")
    if reason:
        decision_factors.append(str(reason))
    else:
        decision_factors.append(
            f"The {domain} agent evaluated the query using the available profile fields."
        )

    confidence = result.get("confidence")
    confidence_text = (
        f"The {domain} agent reported confidence of {confidence}."
        if confidence is not None
        else f"The {domain} agent did not report a confidence score."
    )
    return {
        "data_used": data_used,
        "decision_factors": decision_factors,
        "next_steps": next_steps,
        "confidence_explanation": confidence_text,
    }


def _build_chat_request(db: Session, user_id: str, query: str) -> SimpleNamespace:
    """Adapt persisted profile data to the LangChain agent request contract."""
    from services.profile_service import get_full_profile

    profile = get_full_profile(db, user_id)
    general = profile.get("general")
    career = profile.get("career")
    health = profile.get("health")
    finance = profile.get("finance")

    request = SimpleNamespace(
        name="User",
        age=getattr(general, "age", 25) or 25,
        query=query,
        current_skills=getattr(career, "current_skills", None) or [],
        target_role=getattr(career, "target_role", None) or "data scientist",
        location=getattr(general, "location", None) or "Bangalore",
        experience_level=getattr(career, "experience_level", None) or "beginner",
        years_experience=0,
        timeline_months=6,
        current_level=getattr(career, "experience_level", None) or "beginner",
        weight_kg=getattr(general, "weight_kg", None) or 70,
        height_cm=getattr(general, "height_cm", None) or 170,
        sleep_hours=getattr(health, "sleep_hours", None) or 7,
        sleep_quality=getattr(health, "sleep_quality", None) or 7,
        stress_level=getattr(health, "stress_level", None) or 5,
        mood_score=getattr(health, "mood_score", None) or 7,
        anxiety_level=getattr(health, "anxiety_level", None) or 4,
        fitness_goal=getattr(health, "fitness_goal", None) or "general fitness",
        available_days=getattr(health, "active_days_per_week", None) or 3,
        fitness_experience="beginner",
        monthly_income=getattr(finance, "monthly_income", None),
        monthly_expenses=getattr(finance, "monthly_expenses", None),
        current_savings=getattr(finance, "current_savings", None),
        risk_tolerance=getattr(finance, "risk_appetite", None),
        investment_experience=getattr(finance, "investment_experience", None),
        financial_goals=getattr(finance, "financial_goals", None),
        portfolio=getattr(finance, "portfolio", None) or {},
        debts=getattr(finance, "debts", None) or [],
        total_debt=getattr(finance, "total_debt", None),
        monthly_debt_payment=getattr(finance, "monthly_debt_payment", None),
        retirement_age=getattr(finance, "retirement_age", None),
        retirement_savings=getattr(finance, "retirement_savings", None),
        monthly_contribution=getattr(finance, "monthly_contribution", None),
        annual_income=getattr(finance, "annual_income", None),
        tax_deductions=getattr(finance, "tax_deductions", None) or {},
    )
    return request


def run_chat_agent(db: Session, user_id: str, domain: str, query: str) -> dict:
    """Run the official LangChain domain agent for an authenticated chat."""
    safety = check_safety(query)
    if not safety["is_safe"]:
        return {
            "status": "blocked",
            "domain": domain,
            "tools_used": [],
            "message": safety["message"],
            "reason": safety["reason"],
            "confidence": 1.0,
            "confidence_level": "High",
        }

    request = _build_chat_request(db, user_id, query)
    agent = {
        "career": career_run,
        "health": health_run,
        "finance": finance_run,
    }.get(domain)
    if agent is None:
        raise ValueError(f"Unsupported chat domain: {domain}")
    result = agent(request)
    if not isinstance(result, dict):
        raise TypeError(f"{domain} agent returned an invalid response")

    # Keep the shared chat contract consistent when a LangChain agent does not
    # provide the richer explainability block used by the tool pipelines.
    if not result.get("explainability"):
        result["explainability"] = _build_chat_explainability(
            domain, request, result
        )
    return result

async def meta_lc_agent(request) -> dict:
    """
    LangChain powered meta agent.
    Same interface as existing meta_agent() in main.py.
    """
    # Safety check
    safety = check_safety(request.query)
    if not safety["is_safe"]:
        return {
            "status":            "blocked",
            "reason":            safety["reason"],
            "message":           safety["message"],
            "domains_activated": []
        }

    # Relevance check
    relevance = check_relevance(request.query)
    if not relevance["is_relevant"]:
        return {
            "status":            "out_of_scope",
            "message":           relevance["message"],
            "domains_activated": []
        }

    # Intent detection
    if request.domain == "auto":
        intent  = detect_intent(request.query)
        domains = intent["domains"]
    else:
        intent = {
            "domains":    [request.domain],
            "confidence": 1.0,
            "reasoning":  "Manual domain selection"
        }
        domains = [request.domain]

    # Route to LangChain agents
    agent_map = {
        "career":  career_run,
        "health":  health_run,
        "finance": finance_run,
    }

    loop = asyncio.get_event_loop()

    def run_one(domain):
        fn = agent_map.get(domain)
        if fn:
            return fn(request)
        return {"domain": domain, "message": "Unknown domain"}

    futures = [
        loop.run_in_executor(executor, run_one, domain)
        for domain in domains if domain in agent_map
    ]

    results = await asyncio.gather(*futures, return_exceptions=True)
    responses = []
    for domain, result in zip(domains, results):
        if isinstance(result, Exception):
            print(f"[LangChain Meta] {domain} agent failed: {result}")
            fallback_fn = FALLBACK_AGENTS.get(domain)
            if fallback_fn:
                fallback = fallback_fn(request)
                fallback["agent_type"] = "fallback"
                responses.append(fallback)
            else:
                responses.append({
                    "domain": domain,
                    "error": "agent_failure",
                    "message": f"{domain} agent failed and no fallback is available."
                })
        else:
            responses.append(result)

    return {
        "status":            "success",
        "intent":            intent,
        "responses":         responses,
        "domains_activated": domains,
        "agent_framework":   "langchain"
    }