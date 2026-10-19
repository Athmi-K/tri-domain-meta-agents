"""
Career LangChain Agent — LangChain 1.x + LangGraph
Uses create_react_agent from langgraph.prebuilt
"""
import os
import json
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from core.config import settings

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model=settings.GROQ_MODEL,
    temperature=0.7
)

# ── Tools ─────────────────────────────────────────────────────
@tool
def skill_gap_tool(current_skills: list, target_role: str) -> dict:
    """Analyzes skill gap between current skills and target role.
    Returns match percentage and missing skills.
    Use when user wants to switch roles or learn new skills."""
    from tools.calculators import skill_gap_analyzer
    return skill_gap_analyzer(current_skills, target_role)

@tool
def job_search_tool(current_skills: list, location: str, experience_level: str) -> dict:
    """Finds job matches based on skills, location and experience.
    Use when user asks about job opportunities or market demand."""
    from tools.calculators import job_search
    return job_search(current_skills, location, experience_level)

@tool
def salary_benchmark_tool(target_role: str, location: str, years_experience: int) -> dict:
    """Returns salary benchmarks for a role in a location.
    Use when user asks about salary expectations or negotiation."""
    from tools.calculators import salary_benchmark
    return salary_benchmark(target_role, location, years_experience)

@tool
def learning_path_tool(target_role: str, current_level: str, timeline_months: int) -> dict:
    """Generates structured learning path to reach a target role.
    Use when user asks how to learn or transition into a role."""
    from tools.calculators import learning_path_generator
    return learning_path_generator(target_role, current_level, timeline_months)

# ── System prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a specialist career advisor inside a TriDomain AI system.
You have tools for skill gap analysis, job search, salary benchmarking and learning paths.

RULES:
1. Answer the user's query directly.
2. For technical concept questions, explain the requested concept and do not
   call career tools or add an unrelated roadmap.
3. Use a tool only when the query requires that tool. Do not claim a tool ran
   unless its result is present in the conversation.
4. Never invent profile values, achievements, job listings, sources, or
   confidence scores.
5. Never recommend quitting without a financial safety plan.
6. Tailor advice to the Indian job market only when location or market advice
   is relevant.
7. If the user requests a four-week daily schedule, produce 28 daily entries
   with approximately the requested daily time. Do not choose C++, Java, or
   another language unless the user explicitly requests it or validated profile
   data requires it. Keep weekly topics and durations consistent with the
   requested schedule.

After using tools, respond ONLY with valid JSON:
{
    "recommendation": "specific actionable advice",
    "reason": "why this fits this user based on tool results"
}"""

# ── Agent ─────────────────────────────────────────────────────
tools = [skill_gap_tool, job_search_tool, salary_benchmark_tool, learning_path_tool]
career_agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)


def _build_career_explainability(request, result: dict) -> dict:
    """Build Career Explainability strictly from the execution record."""
    query = getattr(request, "query", "")
    tools_used = result.get("tools_used") or []
    tool_outputs = result.get("tool_outputs") or {}
    profile_fields = {
        "Target role": getattr(request, "target_role", None),
        "Current skills": getattr(request, "current_skills", None),
        "Experience level": getattr(request, "experience_level", None),
        "Location": getattr(request, "location", None),
        "Timeline": getattr(request, "timeline_months", None),
    }
    tool_profile_fields = {
        "skill_gap_tool": {"Current skills", "Target role"},
        "skill_gap_analyzer": {"Current skills", "Target role"},
        "job_search_tool": {"Current skills", "Location", "Experience level"},
        "job_search": {"Current skills", "Location", "Experience level"},
        "salary_benchmark_tool": {"Target role", "Location"},
        "salary_benchmark": {"Target role", "Location"},
        "learning_path_tool": {"Target role", "Experience level", "Timeline"},
        "learning_path_generator": {"Target role", "Experience level", "Timeline"},
        "resume_optimizer": {"Target role"},
    }
    fields_used = set().union(
        *(tool_profile_fields.get(tool_name, set()) for tool_name in tools_used)
    )
    profile_values_used = {
        label: value
        for label, value in profile_fields.items()
        if label in fields_used and value not in (None, "", [], {})
    }
    failed_tools = [
        tool_name
        for tool_name, output in tool_outputs.items()
        if isinstance(output, dict) and output.get("status") == "failed"
    ]
    sources = result.get("sources") or []
    query_facts = _extract_career_query_facts(query)
    data_used = [f"Original query: {query}"]
    data_used.extend(f"{label}: {value}" for label, value in query_facts.items())
    data_used.extend(
        f"{label}: {value}" for label, value in profile_values_used.items()
    )
    limitations = []
    retrieval_status = result.get("retrieval_status", "not_attempted")
    retrieval_error = result.get("retrieval_error")
    if not tools_used:
        limitations.append("No Career tools were executed for this answer.")
    if retrieval_status == "failed":
        limitations.append("Career knowledge retrieval failed; the answer used the LLM fallback.")
        if retrieval_error:
            limitations.append(f"Retrieval error: {retrieval_error}")
    elif retrieval_status == "empty":
        limitations.append("Career knowledge retrieval returned no documents.")
    elif retrieval_status == "not_used":
        limitations.append(
            "Career knowledge was retrieved before LangChain fallback, "
            "but was not passed to the fallback agent."
        )
    elif retrieval_status == "not_attempted" and not sources:
        limitations.append("No retrieved Career sources were recorded.")
    if failed_tools:
        limitations.append(
            f"Tool execution failed: {', '.join(failed_tools)}."
        )
    if retrieval_status == "success":
        decision_factors = [
            "The answer was generated by the LangChain Career LLM using retrieved Career documents."
        ]
    elif tools_used:
        decision_factors = [
            "The answer reflects the Career tools recorded in the execution trace."
        ]
    else:
        decision_factors = [
            "The answer was generated directly by the LangChain Career LLM."
        ]
    decision_factors.extend(
        f"{label} was taken from the user's query."
        for label in query_facts
    )
    if failed_tools:
        decision_factors.append("Failed tool outputs were not treated as successful evidence.")

    missing_profile_fields = [
        label
        for label, value in profile_fields.items()
        if value in (None, "", [], {})
        and label in (
            {"Target role", "Current skills", "Experience level"}
            if query_facts.get("Placement goal")
            else fields_used
        )
    ]
    explanation = {
        "original_query": query,
        "selected_agent": result.get("agent_type", "langchain"),
        "domain": "career",
        "data_used": data_used,
        "profile_values_used": profile_values_used,
        "decision_factors": decision_factors,
        "tools_invoked": tools_used,
        "tool_statuses": {
            tool_name: output.get("status")
            for tool_name, output in tool_outputs.items()
            if isinstance(output, dict)
        },
        "retrieved_sources": list(dict.fromkeys(sources)),
        "retrieval_status": retrieval_status,
        "missing_profile_fields": missing_profile_fields,
        "limitations": limitations,
        "next_steps": [
            "Review the recommendation against the original Career question."
        ],
        "confidence_explanation": (
            "No confidence score was reported."
            if result.get("confidence") is None
            else f"The Career agent reported confidence of {result['confidence']}."
        ),
    }
    processing_steps = [
        f"Used {label.lower()} from the query."
        for label in query_facts
    ]
    if processing_steps:
        explanation["processing_steps"] = processing_steps
    confidence_level = result.get("confidence_level")
    if confidence_level not in (None, ""):
        explanation["confidence_level"] = confidence_level
    calculation_steps = {
        tool_name: output["calculation_steps"]
        for tool_name, output in tool_outputs.items()
        if isinstance(output, dict) and output.get("calculation_steps")
    }
    if calculation_steps:
        explanation["calculation_steps"] = calculation_steps
    return explanation


def _extract_career_query_facts(query: str) -> dict[str, str]:
    """Extract explicit Career constraints used by the prompt, not model claims."""
    facts: dict[str, str] = {}
    lowered = query.casefold()
    duration = re.search(r"\b(\d+)\s*-\s*week(?:s)?\b", lowered)
    days = re.search(r"\b(?:next|over the next|within)\s+(\d+)\s+days?\b", lowered)
    daily_time = re.search(
        r"\b(\d+(?:\.\d+)?|one|a)\s*(?:hours?|hrs?)"
        r"(?:\s+of\s+\w+)?\s*(?:per|a)\s*day\b",
        lowered,
    )
    if duration:
        facts["Requested duration"] = f"{duration.group(1)} weeks"
    elif days:
        facts["Requested duration"] = f"{days.group(1)} days"
    if daily_time:
        amount = "1" if daily_time.group(1) in {"one", "a"} else daily_time.group(1)
        unit = "hour" if amount == "1" else "hours"
        facts["Daily study limit"] = f"{amount} {unit} per day"
    if "placement" in lowered:
        facts["Placement goal"] = "software engineering placements"
    language = re.search(
        r"\b(?:learning|learn|using|use)\s+"
        r"(java|python|javascript|typescript|c\+\+|c#|go|rust)\b",
        lowered,
    )
    if language:
        language_name = language.group(1)
        facts["Requested language"] = language_name
    else:
        basic_language = re.search(
            r"\bbasic\s+(java|python|javascript|typescript|c\+\+|c#|go|rust)\b",
            lowered,
        )
        if basic_language:
            facts["Requested language"] = basic_language.group(1)
    return facts


def _execution_reason(result: dict) -> str:
    """Describe observable execution when the model supplied a generic reason."""
    reason = str(result.get("reason") or "")
    generic_reason = "no structured execution metadata was returned"
    if generic_reason not in reason.casefold():
        return reason

    retrieval_status = result.get("retrieval_status", "not_attempted")
    tools_used = result.get("tools_used") or []
    if retrieval_status == "success":
        evidence = "retrieved Career documents"
        if tools_used:
            evidence += " and recorded Career tool execution"
        else:
            evidence += "; no Career tools were executed"
        return f"The answer used {evidence}."
    if tools_used:
        return "The answer used recorded Career tool execution; no retrieval was used."
    if retrieval_status == "empty":
        return "The answer used the Career LLM fallback after retrieval returned no documents."
    if retrieval_status == "failed":
        return "The answer used the Career LLM fallback because Career retrieval failed."
    return "The answer was generated directly by the Career LLM; no tools or retrieval were used."


def _should_retrieve_career_context(query: str) -> bool:
    """Retrieve KB context for planning questions, not standalone concepts."""
    lowered = query.casefold()
    planning_terms = (
        "placement",
        "prepare for",
        "preparation",
        "study plan",
        "daily plan",
        "roadmap",
        "learning plan",
        "coding problems",
        "aptitude",
        "technical interview",
    )
    return any(term in lowered for term in planning_terms)


def _retrieve_career_context(query: str) -> dict:
    """Return retrieval evidence without hiding empty or failed execution."""
    if not _should_retrieve_career_context(query):
        return {
            "status": "not_attempted",
            "context": "",
            "sources": [],
        }
    try:
        from rag.retriever import retrieve

        documents = retrieve(query, domain="career", top_k=5) or []
        sources = list(dict.fromkeys(
            document.get("source")
            for document in documents
            if isinstance(document, dict) and document.get("source")
        ))
        context = "\n\nRELEVANT CAREER KNOWLEDGE:\n" + "\n\n".join(
            f"[Source: {document.get('source', 'unknown')}]\n"
            f"{document.get('text', '')}"
            for document in documents
            if isinstance(document, dict) and document.get("text")
        )
        return {
            "status": "success" if documents else "empty",
            "context": context if documents else "",
            "sources": sources,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "context": "",
            "sources": [],
            "error": str(exc),
        }


# ── Run ───────────────────────────────────────────────────────
def run(request) -> dict:
    retrieval = _retrieve_career_context(getattr(request, "query", ""))
    profile_lines = []
    profile_values = (
        ("Current skills", getattr(request, "current_skills", None)),
        ("Target role", getattr(request, "target_role", None)),
        ("Location", getattr(request, "location", None)),
        ("Experience level", getattr(request, "experience_level", None)),
        ("Years of experience", getattr(request, "years_experience", None)),
        ("Timeline", getattr(request, "timeline_months", None)),
    )
    for label, value in profile_values:
        if value not in (None, "", [], {}):
            profile_lines.append(f"- {label}: {value}")

    profile_text = "\n".join(profile_lines) or "- No validated profile values provided"
    user_input = f"""Answer this query directly:
{request.query}

Validated profile values, to use only when relevant:
{profile_text}

Use a career tool only when the query requires it. For a daily schedule,
use the explicit duration and daily time in the query rather than a default
timeline.

Career knowledge retrieval status: {retrieval["status"]}
{retrieval["context"] or "No Career knowledge context was retrieved."}"""

    try:
        result = career_agent.invoke({
            "messages": [{"role": "user", "content": user_input}]
        })

        # Get last message content
        last_msg = result["messages"][-1].content
        if isinstance(last_msg, list):
            last_msg = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in last_msg
            )
        last_msg = str(last_msg)

        # Extract JSON
        match = re.search(r'\{[^{}]*"recommendation"[^{}]*\}', last_msg, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
        else:
            parsed = {
                "recommendation": last_msg,
                "reason": (
                    "The answer was generated directly by the Career LLM; "
                    "no structured execution metadata was returned."
                ),
            }
        # The model has no calibrated basis for a numeric confidence score.
        parsed.pop("confidence", None)

        tool_messages = [
            message for message in result.get("messages", [])
            if getattr(message, "type", "") == "tool"
        ]
        tools_used = list(dict.fromkeys(
            getattr(message, "name", None)
            for message in tool_messages
            if getattr(message, "name", None)
        ))
        tool_outputs = {}
        sources = []
        for message in tool_messages:
            tool_name = getattr(message, "name", "unknown")
            status = getattr(message, "status", None) or "completed"
            content = getattr(message, "content", "")
            if isinstance(content, dict) and content.get("error"):
                status = "failed"
            if isinstance(content, dict):
                message_sources = content.get("sources", content.get("source", []))
                if isinstance(message_sources, str):
                    message_sources = [message_sources]
                if isinstance(message_sources, list):
                    sources.extend(
                        source for source in message_sources if isinstance(source, str)
                    )
            tool_outputs[tool_name] = {
                "status": status,
                "output": content,
            }

        response_sources = list(dict.fromkeys(
            [*retrieval["sources"], *sources]
        ))
        response = {
            "domain":      "career",
            "agent_type":  "langchain",
            "tools_used":  tools_used,
            "tool_outputs": tool_outputs,
            "sources": response_sources,
            "retrieval_status": retrieval["status"],
            "retrieval_error": retrieval.get("error"),
            **parsed
        }
        response["reason"] = _execution_reason(response)
        response["explainability"] = _build_career_explainability(
            request, response
        )
        return response

    except Exception as e:
        print(f"[LangChain Career] Error: {e} — falling back")
        import agents.career_agent as original
        result = original.run(request)
        result["agent_type"] = "fallback"
        result.setdefault(
            "retrieval_status",
            "not_used" if retrieval["status"] == "success" else retrieval["status"],
        )
        if retrieval.get("error"):
            result.setdefault("retrieval_error", retrieval["error"])
        return result