from __future__ import annotations

import re
from typing import Any

from core.llm_client import call_llm

from tools.calculators import (
    salary_benchmark,
)

from tools.career_tools import (
    skill_gap_analyzer,
    job_search,
    resume_optimizer,
    learning_path_generator,
)


_CAREER_ANALYSIS_TERMS = (
    "career",
    "job",
    "role",
    "salary",
    "resume",
    "cv",
    "interview",
    "placement",
    "skill gap",
    "roadmap",
    "learning plan",
    "upskill",
    "switch",
    "promotion",
    "hire",
)
_KNOWN_LOCATIONS = {
    "ahmedabad",
    "bengaluru",
    "bangalore",
    "bhubaneswar",
    "chandigarh",
    "chennai",
    "coimbatore",
    "delhi",
    "gurgaon",
    "hyderabad",
    "jaipur",
    "kochi",
    "kolkata",
    "mumbai",
    "noida",
    "pune",
    "remote",
    "india",
    "anywhere",
}


def _should_run_career_analysis(query: str, resume_text: str) -> bool:
    """Return whether the request needs the career-analysis tools."""
    if resume_text:
        return True
    if not query.strip():
        return True
    normalized_query = query.casefold()
    return any(term in normalized_query for term in _CAREER_ANALYSIS_TERMS)


def _as_optional_text(value: Any) -> str | None:
    """Return usable text without turning missing profile data into a fact."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _validated_location(value: Any) -> str | None:
    location = _as_optional_text(value)
    if location and location.casefold() in _KNOWN_LOCATIONS:
        return location
    return None


def _schedule_constraints(query: str) -> str:
    """Describe explicit schedule constraints for the summary model."""
    normalized = query.casefold()
    weeks_match = re.search(r"\b(\d+)\s*[- ]?\s*week", normalized)
    hours_match = re.search(
        r"\b(?:only\s*)?(\d+(?:\.\d+)?)\s*hours?\s*(?:per|a|/)\s*day\b",
        normalized,
    )
    constraints = []
    if weeks_match:
        weeks = int(weeks_match.group(1))
        constraints.append(
            f"Use exactly {weeks} weeks ({weeks * 7} daily entries) "
            "when a daily schedule is requested."
        )
    if hours_match:
        constraints.append(
            f"Plan for approximately {hours_match.group(1)} hour(s) per day."
        )
    language_match = re.search(
        r"\b(?:in|using|with)\s+(java|python|c\+\+|javascript|typescript|go|rust)\b"
        r"|\b(java|python|c\+\+|javascript|typescript|go|rust)\s+"
        r"(?:placement|schedule|roadmap|plan)\b",
        normalized,
    )
    if language_match:
        language = next(
            group for group in language_match.groups() if group is not None
        )
        constraints.append(f"Use the requested language: {language}.")
    else:
        constraints.append(
            "Do not choose or force a programming language unless the user "
            "requests one or validated profile data makes it necessary."
        )
    return " ".join(constraints)


def _requested_timeline_months(query: str) -> int | None:
    match = re.search(r"\b(\d+)\s*[- ]?\s*week", query.casefold())
    if not match:
        return None
    return max(1, round(int(match.group(1)) / 4))


def _get_request_value(
    request: Any,
    field_name: str,
    default: Any,
) -> Any:
    """
    Safely read a value from either an object or dictionary.
    """

    if isinstance(request, dict):
        value = request.get(field_name, default)
    else:
        value = getattr(request, field_name, default)

    return default if value is None else value


def _format_skills(skills: list) -> str:
    """
    Convert the skills list into readable text.
    """

    if not skills:
        return "No skills provided"

    return ", ".join(str(skill) for skill in skills)


def _format_top_job(jobs_data: dict) -> str:
    """
    Format the highest-ranked job returned by job_search().

    FIX: distinguishes "search ran, found nothing" from "search failed"
    so the LLM (and eventually the user) isn't told the same generic
    message for both cases.
    """

    if jobs_data.get("error"):
        return f"Job search could not be completed: {jobs_data['error']}"

    jobs = jobs_data.get("jobs", [])

    if not jobs:
        return "No matching jobs found"

    top_job = jobs[0]

    title = top_job.get("title", "Unknown role")
    company = top_job.get("company", "Unknown company")
    location = top_job.get("location", "Location not specified")
    score = top_job.get("embedding_match_score", 0)

    return (
        f"{title} at {company}, "
        f"Location: {location}, "
        f"Embedding match score: {score}%"
    )


def _format_resume_result(resume_data: dict | None) -> str:
    """
    Format the resume_optimizer() result.
    """

    if not resume_data:
        return "Resume analysis was not requested"

    if resume_data.get("error"):
        return f"Resume analysis could not be completed: {resume_data['error']}"

    semantic_score = resume_data.get("semantic_match_score", 0)

    required_skills = resume_data.get("required_skills", [])
    skills_not_evidenced = resume_data.get("skills_not_evidenced", [])
    suggestions = resume_data.get("suggestions", [])

    return (
        f"Resume semantic match score: {semantic_score}%\n"
        f"Required skills: {_format_skills(required_skills)}\n"
        f"Skills not evidenced: {_format_skills(skills_not_evidenced)}\n"
        f"Suggestions: {suggestions}"
    )


CAREER_ADVISOR_SYSTEM_PROMPT = """
You are a practical and honest assistant in the Career domain.

Answer the user's current query directly and prioritize it over any generic
career-readiness workflow. For technical or educational questions, explain
the requested topic and provide requested examples without adding a career
roadmap. For career-guidance questions, use the supplied profile and career
analysis data to give relevant advice. Do not invent facts, job listings,
salary figures, profile details, or confidence scores.

Return ONLY valid JSON.
Return exactly one key named "summary".
The value of "summary" must be a concise answer to the user's current query.
Do not return Markdown outside the JSON object.
Do not add extra keys.
"""


def _build_summary_prompt(
    current_query: str,
    target_role: str,
    current_skills: list,
    gap_data: dict,
    jobs_data: dict,
    salary_data: dict,
    path_data: dict,
    resume_data: dict | None,
) -> str:
    """
    Build the user-message prompt sent to the language model.

    The LLM is instructed to return only one valid JSON object
    containing a single "summary" field.
    """

    top_job_text = _format_top_job(jobs_data)
    resume_text = _format_resume_result(resume_data)

    prompt = f"""
    Answer the user's current query directly and concisely.
    Treat the current query as the primary instruction. Do not provide
    unrelated career recommendations or force a career roadmap.

    Current user query:
    {current_query}

    Explicit schedule constraints:
    {_schedule_constraints(current_query)}

    Use the following career information only when it is relevant to the
    current query. For technical questions, answer the requested topic and
    include requested examples. For career-guidance questions, use the
    profile and analysis data below. Use only information supported by the
    provided data; do not invent job listings, salary figures, profile
    details, or confidence scores.

Target role:
{target_role}

Current skills:
{_format_skills(current_skills)}

Skill-gap analysis:
{gap_data}

Top matching job:
{top_job_text}

Salary information:
{salary_data}

Learning path:
{path_data}

Resume analysis:
{resume_text}

When discussing resume improvements, state only achievements supported by the
resume analysis. If an achievement, certification, metric, or experience is
missing, label it as missing and phrase any example conditionally ("if true,
add evidence") rather than as a fact.

For career-guidance questions, cover the relevant parts of the user's
career position, missing skills, learning order, job readiness, job
opportunities, and resume improvements. For other questions, omit these
career details unless the user asks for them.

Return ONLY valid JSON in this exact format:

{{
  "summary": "A concise advisory summary."
}}

The summary must be no more than 350 words.
Use short sections and concise dash-prefixed bullets.
Do not repeat all input data.
"""

    return prompt.strip()


def run(request: Any, constraints: str = "") -> dict:
    """
    Run the complete career advisory workflow.

    Expected request fields:

    - current_skills: list
    - target_role: str
    - location: str
    - experience_level: str
    - years_experience: int or float
    - current_level: str
    - timeline_months: int
    - resume_text: str
    """
    
    current_query = _get_request_value(
        request,
        "current_query",
        _get_request_value(request, "query", ""),
    )

    current_skills = _get_request_value(
        request,
        "current_skills",
        [],
    )

    target_role = _get_request_value(request, "target_role", None)

    location = _get_request_value(
        request,
        "location",
        None,
    )

    experience_level = _get_request_value(
        request,
        "experience_level",
        None,
    )

    years_experience = _get_request_value(
        request,
        "years_experience",
        0,
    )

    current_level = _get_request_value(
        request,
        "current_level",
        None,
    )

    timeline_months = _get_request_value(
        request,
        "timeline_months",
        None,
    )

    resume_text = _get_request_value(
        request,
        "resume_text",
        "",
    )

    if not isinstance(current_skills, list):
        current_skills = [str(current_skills)]

    current_skills = [
        str(skill).strip()
        for skill in current_skills
        if str(skill).strip()
    ]

    target_role = _as_optional_text(target_role)
    location = _validated_location(location)
    experience_level = _as_optional_text(experience_level)
    current_level = _as_optional_text(current_level)
    resume_text = str(resume_text).strip()
    current_query = str(current_query).strip()
    requested_timeline = _requested_timeline_months(current_query)
    if requested_timeline is not None:
        timeline_months = requested_timeline

    analysis_requested = _should_run_career_analysis(current_query, resume_text)

    if not analysis_requested:
        gap_data = {}
        jobs_data = {}
        salary_data = {}
        path_data = {}
    else:
        # ---------------------------------------------------------
        # 1. Analyze skill gaps
        # ---------------------------------------------------------
        gap_data = skill_gap_analyzer(
            current_skills=current_skills,
            target_role=target_role or "",
        )

        # ---------------------------------------------------------
        # 2. Search and rank jobs
        # ---------------------------------------------------------
        jobs_data = job_search(
            target_role=target_role or "",
            skills=current_skills,
            location=location or "",
            experience_level=experience_level or "",
        )

        # ---------------------------------------------------------
        # 3. Get salary information
        # ---------------------------------------------------------
        salary_data = salary_benchmark(
            role=target_role or "",
            location=location or "",
            years_experience=years_experience or 0,
        )

        # ---------------------------------------------------------
        # 4. Generate learning path
        # ---------------------------------------------------------
        path_data = learning_path_generator(
            target_role=target_role or "",
            current_level=current_level or "",
            timeline_months=timeline_months or 0,
            missing_skills=(
                gap_data.get("missing_skills", [])
                if isinstance(gap_data, dict)
                else []
            ),
        )

    # ---------------------------------------------------------
    # 5. Analyze resume if supplied
    # ---------------------------------------------------------

    resume_data = None

    if resume_text:
        resume_data = resume_optimizer(
            resume_text=resume_text,
            target_job=target_role or "",
        )

    tool_outputs = {
        "skill_gap_analyzer": gap_data,
        "job_search": jobs_data,
        "salary_benchmark": salary_data,
        "learning_path_generator": path_data,
    }
    if resume_data is not None:
        tool_outputs["resume_optimizer"] = resume_data
    tools_used = [
        name for name, output in tool_outputs.items()
        if isinstance(output, dict)
    ]
    sources = []
    for output in tool_outputs.values():
        if not isinstance(output, dict):
            continue
        source_values = output.get("sources", output.get("source", []))
        if isinstance(source_values, str):
            source_values = [source_values]
        if isinstance(source_values, list):
            sources.extend(
                source for source in source_values if isinstance(source, str)
            )

    # ---------------------------------------------------------
    # 6. Build the language-model prompt
    # ---------------------------------------------------------

    summary_prompt = _build_summary_prompt(
        current_query=current_query,
        target_role=target_role,
        current_skills=current_skills,
        gap_data=gap_data,
        jobs_data=jobs_data,
        salary_data=salary_data,
        path_data=path_data,
        resume_data=resume_data,
    )

    if constraints:
        summary_prompt += f"""

Additional constraints:
{constraints}
"""

        # ---------------------------------------------------------
    # 7. Generate final career advice
    # ---------------------------------------------------------

    try:
        llm_result = call_llm(
            CAREER_ADVISOR_SYSTEM_PROMPT,
            summary_prompt,
            temperature=0.4,
            max_tokens=2500,
        )
        print("DEBUG - LLM RESULT:")
        print(llm_result)

        if not isinstance(llm_result, dict):
            llm_response = str(llm_result)

        elif llm_result.get("summary"):
            llm_response = str(llm_result["summary"])

        elif llm_result.get("recommendation"):
            llm_response = str(llm_result["recommendation"])

        elif llm_result.get("error"):
            llm_response = (
                "Career analysis was completed, but the final advisory "
                f"response could not be generated: {llm_result['error']}"
            )

        else:
            llm_response = (
                "Career analysis was completed, but the model returned "
                "an unexpected response format."
            )

    except Exception as exc:
        import traceback

        traceback.print_exc()

        llm_response = (
            "Career analysis was completed, but the final advisory "
            f"response could not be generated: {exc}"
        )

    # ---------------------------------------------------------
    # 8. Return structured result
    # ---------------------------------------------------------

    return {
        "target_role": target_role,
        "location": location,
        "experience_level": experience_level,
        "current_skills": current_skills,
        "skill_gap": gap_data,
        "jobs": jobs_data,
        "salary": salary_data,
        "learning_path": path_data,
        "resume_analysis": resume_data,
        "summary": llm_response,
        "tools_used": tools_used if analysis_requested else [],
        "tool_outputs": (
            {
                name: {
                    "status": "failed" if output.get("error") else "completed",
                    "output": output,
                }
                for name, output in tool_outputs.items()
            }
            if analysis_requested
            else {}
        ),
        "sources": list(dict.fromkeys(sources)) if analysis_requested else [],
    }