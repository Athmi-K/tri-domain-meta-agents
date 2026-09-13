"""
Job preprocessing for the Career Agent.

Pipeline:

JSearch results
      ↓
Deduplication
      ↓
Profile filtering
      ↓
Job matcher
"""

import re
from datetime import datetime
from difflib import SequenceMatcher


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def _normalize_text(text: str) -> str:
    if not text:
        return ""

    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# LOCATION NORMALIZATION
# ============================================================

LOCATION_ALIASES = {
    "bangalore": "bengaluru",
    "bengaluru": "bengaluru",

    "bombay": "mumbai",
    "mumbai": "mumbai",

    "madras": "chennai",
    "chennai": "chennai",

    "calcutta": "kolkata",
    "kolkata": "kolkata",

    "gurgaon": "gurugram",
    "gurugram": "gurugram",
}


def _normalize_location(location: str) -> str:
    normalized = _normalize_text(location)

    for alias, canonical in LOCATION_ALIASES.items():
        normalized = re.sub(
            rf"\b{re.escape(alias)}\b",
            canonical,
            normalized,
        )

    return normalized


# ============================================================
# DATE PARSING
# ============================================================

def _parse_posted_date(value):
    if not value:
        return None

    if isinstance(value, datetime):
        return value

    value = str(value).strip()

    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except ValueError:
        pass

    formats = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
    ]

    for date_format in formats:
        try:
            return datetime.strptime(
                value,
                date_format,
            )
        except ValueError:
            continue

    return None


# ============================================================
# DUPLICATION
# ============================================================

def _titles_similar(
    title_a: str,
    title_b: str,
    threshold: float = 0.88,
) -> bool:

    if not title_a or not title_b:
        return False

    return (
        SequenceMatcher(
            None,
            title_a,
            title_b,
        ).ratio()
        >= threshold
    )


def _locations_compatible(
    location_a: str,
    location_b: str,
) -> bool:

    if not location_a or not location_b:
        return True

    if location_a == location_b:
        return True

    return (
        location_a in location_b
        or location_b in location_a
    )


def _completeness_score(job: dict) -> int:

    score = 0

    if job.get("description"):
        score += 3

    if job.get("apply_link"):
        score += 2

    if job.get("location"):
        score += 1

    if job.get("posted_at"):
        score += 1

    if job.get("company"):
        score += 1

    return score


def _is_better_duplicate(
    candidate: dict,
    existing: dict,
) -> bool:

    candidate_score = _completeness_score(candidate)
    existing_score = _completeness_score(existing)

    if candidate_score > existing_score:
        return True

    if candidate_score < existing_score:
        return False

    candidate_date = _parse_posted_date(
        candidate.get("posted_at")
    )

    existing_date = _parse_posted_date(
        existing.get("posted_at")
    )

    if candidate_date and existing_date:
        return candidate_date > existing_date

    if candidate_date and not existing_date:
        return True

    return False


def deduplicate_jobs(
    jobs: list[dict],
    title_threshold: float = 0.88,
) -> list[dict]:

    seen = []

    for job in jobs:

        if not isinstance(job, dict):
            continue

        norm_title = _normalize_text(
            job.get("title", "")
        )

        norm_company = _normalize_text(
            job.get("company", "")
        )

        norm_location = _normalize_location(
            job.get("location", "")
        )

        match_index = None

        for index, existing in enumerate(seen):

            same_company = (
                existing["_norm_company"]
                == norm_company
            )

            similar_title = _titles_similar(
                norm_title,
                existing["_norm_title"],
                title_threshold,
            )

            compatible_location = _locations_compatible(
                norm_location,
                existing["_norm_location"],
            )

            if (
                same_company
                and similar_title
                and compatible_location
            ):
                match_index = index
                break

        if match_index is None:

            entry = dict(job)

            entry["_norm_title"] = norm_title
            entry["_norm_company"] = norm_company
            entry["_norm_location"] = norm_location

            seen.append(entry)

        else:

            if _is_better_duplicate(
                job,
                seen[match_index],
            ):

                entry = dict(job)

                entry["_norm_title"] = norm_title
                entry["_norm_company"] = norm_company
                entry["_norm_location"] = norm_location

                seen[match_index] = entry

    return [
        {
            key: value
            for key, value in job.items()
            if not key.startswith("_")
        }
        for job in seen
    ]


# ============================================================
# EXPERIENCE FILTER
# ============================================================

SENIOR_KEYWORDS = {
    "senior",
    "sr",
    "lead",
    "principal",
    "staff",
    "director",
    "head",
    "manager",
    "architect",
}


ENTRY_LEVEL_EXPERIENCE = {
    "fresher",
    "entry level",
    "entry-level",
    "0-1 years",
    "0 years",
    "intern",
    "internship",
}


def _experience_level_ok(
    job_title: str,
    experience: str,
) -> bool:

    title = _normalize_text(job_title)
    user_experience = _normalize_text(experience)

    if user_experience in ENTRY_LEVEL_EXPERIENCE:

        for keyword in SENIOR_KEYWORDS:

            if re.search(
                rf"\b{re.escape(keyword)}\b",
                title,
            ):
                return False

    return True


# ============================================================
# LOCATION FILTER
# ============================================================

def _location_ok(
    job_location: str,
    preferred_locations: list[str],
) -> bool:

    if not preferred_locations:
        return True

    normalized_job_location = _normalize_location(
        job_location
    )

    # Missing location shouldn't automatically
    # eliminate a potentially useful job.
    if not normalized_job_location:
        return True

    normalized_preferences = [
        _normalize_location(location)
        for location in preferred_locations
        if location
    ]

    # Remote jobs
    if "remote" in normalized_job_location:

        return any(
            "remote" in location
            for location in normalized_preferences
        )

    for preferred_location in normalized_preferences:

        if preferred_location == "remote":
            continue

        if (
            preferred_location in normalized_job_location
            or normalized_job_location in preferred_location
        ):
            return True

    return False


# ============================================================
# PROFILE FILTER
# ============================================================

def filter_by_profile(
    jobs: list[dict],
    profile: dict,
) -> list[dict]:

    experience = profile.get(
        "experience",
        "",
    )

    preferred_locations = profile.get(
        "locations"
    )

    if not preferred_locations:

        location = profile.get("location")

        preferred_locations = (
            [location]
            if location
            else []
        )

    filtered_jobs = []

    for job in jobs:

        if not isinstance(job, dict):
            continue

        if not _experience_level_ok(
            job.get("title", ""),
            experience,
        ):
            continue

        if not _location_ok(
            job.get("location", ""),
            preferred_locations,
        ):
            continue

        filtered_jobs.append(job)

    return filtered_jobs


# ============================================================
# COMPLETE PREPROCESSING
# ============================================================

def prepare_jobs_for_ranking(
    jobs: list[dict],
    profile: dict,
) -> list[dict]:

    deduplicated = deduplicate_jobs(jobs)

    filtered = filter_by_profile(
        deduplicated,
        profile,
    )

    return filtered