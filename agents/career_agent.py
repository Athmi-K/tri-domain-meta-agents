from __future__ import annotations

from typing import Any

from core.llm_client import call_llm

from tools.calculators import (
    salary_benchmark,
    learning_path_generator,
)

from tools.career_tools import (
    skill_gap_analyzer,
    job_search,
    resume_optimizer,
)


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
You are a practical and honest career advisor.

Return ONLY valid JSON.
Return exactly one key named "summary".
The value of "summary" must be a concise career advisory string.
Do not return Markdown outside the JSON object.
Do not add extra keys.
"""


def _build_summary_prompt(
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
Analyze the following career information and provide concise, practical,
and realistic career-readiness advice.

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

The summary must cover:
1. The user's current career position.
2. The most important missing skills.
3. A realistic learning order.
4. Specific ways to improve job readiness.
5. The top job opportunity, if available.
6. Resume improvements, if resume analysis is available.
7. Only information supported by the provided data.

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

    current_skills = _get_request_value(
        request,
        "current_skills",
        [],
    )

    target_role = _get_request_value(
        request,
        "target_role",
        "data analyst",
    )

    location = _get_request_value(
        request,
        "location",
        "Bangalore",
    )

    experience_level = _get_request_value(
        request,
        "experience_level",
        "entry",
    )

    years_experience = _get_request_value(
        request,
        "years_experience",
        0,
    )

    current_level = _get_request_value(
        request,
        "current_level",
        "beginner",
    )

    timeline_months = _get_request_value(
        request,
        "timeline_months",
        6,
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

    target_role = str(target_role).strip()
    location = str(location).strip()
    experience_level = str(experience_level).strip()
    current_level = str(current_level).strip()
    resume_text = str(resume_text).strip()

    # ---------------------------------------------------------
    # 1. Analyze skill gaps
    # ---------------------------------------------------------

    gap_data = skill_gap_analyzer(
        current_skills=current_skills,
        target_role=target_role,
    )

    # ---------------------------------------------------------
    # 2. Search and rank jobs
    # ---------------------------------------------------------

    jobs_data = job_search(
        target_role=target_role,
        skills=current_skills,
        location=location,
        experience_level=experience_level,
    )

    # ---------------------------------------------------------
    # 3. Get salary information
    # ---------------------------------------------------------

    salary_data = salary_benchmark(
        role=target_role,
        location=location,
        years_experience=years_experience,
    )

    # ---------------------------------------------------------
    # 4. Generate learning path
    # ---------------------------------------------------------

    path_data = learning_path_generator(
        goal=target_role,
        current_level=current_level,
        timeline_months=timeline_months,
    )

    # ---------------------------------------------------------
    # 5. Analyze resume if supplied
    # ---------------------------------------------------------

    resume_data = None

    if resume_text:
        resume_data = resume_optimizer(
            resume_text=resume_text,
            target_job=target_role,
        )

    # ---------------------------------------------------------
    # 6. Build the language-model prompt
    # ---------------------------------------------------------

    summary_prompt = _build_summary_prompt(
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
    }