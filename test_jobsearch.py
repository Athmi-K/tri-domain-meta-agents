"""
test_job_search.py

Standalone test for job_search() in tools/career_tools.py - isolates it
from the full career_agent.run() pipeline (skill_gap_analyzer, salary,
learning_path, resume_optimizer, LLM summary) so failures here can only
be about JSearch/RapidAPI or the ranking logic, nothing else.

Run from your project root (same place you'd run your backend from):
    python test_job_search.py

Adjust the import path below if your project layout differs from
`tools.career_tools` (e.g. if you renamed the file or moved it).
"""

import os
import sys

# Load .env the same way the rest of your app does, so RAPIDAPI_KEY is
# actually available to this standalone script too.
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    print("[warn] python-dotenv not installed - relying on already-exported env vars only.")

from tools.career_tools import job_search


def print_section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def check_env():
    print_section("ENVIRONMENT CHECK")

    rapidapi_key = os.getenv("RAPIDAPI_KEY")
    if rapidapi_key:
        print(f"RAPIDAPI_KEY: found (starts with '{rapidapi_key[:8]}...', length {len(rapidapi_key)})")
    else:
        print("RAPIDAPI_KEY: NOT SET - job_search will fail immediately without this.")
        sys.exit(1)


def run_test(label: str, **kwargs):
    print_section(f"TEST: {label}")
    print(f"Calling job_search({kwargs})")

    result = job_search(**kwargs)

    if result.get("error"):
        print(f"\n[FAILED] error: {result['error']}")
        print(f"total_matches: {result.get('total_matches')}")
        return False

    jobs = result.get("jobs", [])
    print(f"\n[OK] total_matches: {result.get('total_matches')}")
    print(f"Returned {len(jobs)} job(s):\n")

    for i, job in enumerate(jobs, 1):
        print(f"  {i}. {job.get('title')} @ {job.get('company')}")
        print(f"     Location: {job.get('location')}")
        print(f"     Match score: {job.get('embedding_match_score')}")
        print(f"     Skills required: {job.get('required_skills')}")
        print()

    return True


if __name__ == "__main__":
    check_env()

    results = []

    # Basic case - matches what's been failing in your actual app
    results.append(run_test(
        "Frontend Developer in Bangalore, junior",
        target_role="Frontend Developer",
        skills=["React", "JavaScript", "HTML", "CSS"],
        location="Bangalore",
        experience_level="junior",
    ))

    # A second role/location, to confirm it's not specific to one query
    results.append(run_test(
        "Data Analyst in Mumbai, entry level",
        target_role="Data Analyst",
        skills=["SQL", "Excel", "Python"],
        location="Mumbai",
        experience_level="entry",
    ))

    # Deliberately bad input - target_role missing - should return a
    # clean error dict, not raise an exception
    results.append(run_test(
        "Missing target_role (should fail gracefully, not crash)",
        target_role="",
        skills=[],
        location="Bangalore",
        experience_level="junior",
    ))

    print_section("SUMMARY")
    passed = sum(1 for r in results if r)
    print(f"{passed}/{len(results)} test cases completed without raising an exception.")
    print("(Check each section above - a clean 'error' key is expected for the")
    print(" missing-target_role case; the other two should show real job results")
    print(" if RAPIDAPI_KEY/JSearch access is working correctly.)")