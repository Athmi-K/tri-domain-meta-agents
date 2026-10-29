"""
evaluate.py
-----------
Evaluates the TriDomain Meta-Agent intent routing and safety classification accuracy.

Scope of this evaluator:
- Target endpoint: POST /query
- Validates: HTTP 200, status alignment, domain activation matching, safety blocks, out-of-scope rejections.
- Output: Non-destructive, timestamped provenance-bearing reports in benchmark_reports/.
- NOTE: This evaluator assesses routing and safety firewall precision; it does not measure RAG response quality or tool execution factual correctness.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure UTF-8 stdout/stderr on Windows terminals to safely render box-drawing characters
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import requests

from test_queries import (
    ADDITIONAL_TEST_QUERIES,
    COMBINED_TEST_QUERIES,
    DATASETS,
    TEST_QUERIES,
)

API = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_SECONDS = 60

BASE_PROFILE = {
    "name": "TestUser",
    "age": 28,
    "weight_kg": 75,
    "height_cm": 175,
    "monthly_income": 50000,
    "monthly_expenses": 35000,
    "current_skills": ["python", "sql"],
    "target_role": "data scientist",
}


def get_provenance(
    dataset_path: str = "test_queries.py",
    dataset_name: str = "baseline",
    query_count: int = 0,
    endpoint: str = API,
) -> dict[str, Any]:
    """Capture environment and code revision metadata without inventing values."""
    git_commit = "unavailable"
    git_branch = "unavailable"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        pass

    try:
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        pass

    model_configured = "unavailable"
    try:
        from core.config import settings

        model_configured = getattr(settings, "GROQ_MODEL", "unavailable")
    except Exception:
        pass

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_branch": git_branch,
        "dataset_path": dataset_path,
        "dataset_name": dataset_name,
        "query_count": query_count,
        "endpoint": endpoint,
        "python_version": sys.version.split()[0],
        "system_platform": platform.platform(),
        "model_configured": model_configured,
    }


def evaluate_query(
    test: dict,
    api_url: str = API,
    base_profile: dict | None = None,
    session: requests.Session | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """
    Evaluate a single test query against the API.
    Strictly checks HTTP status, response schema, and expected classification.
    Keeps latency tightly synchronized with the query result.
    """
    expected_status = test.get("expected_status", "success")
    expected_domains = sorted(test.get("expected_domains", []))
    payload = {**(base_profile or BASE_PROFILE), "query": test.get("query", "")}
    http_caller = session.post if session else requests.post

    start = time.time()
    try:
        res = http_caller(f"{api_url}/query", json=payload, timeout=timeout)
        end = time.time()
        latency = round((end - start) * 1000)
    except requests.exceptions.Timeout as exc:
        end = time.time()
        latency = round((end - start) * 1000)
        return {
            "id": test.get("id"),
            "category": test.get("category"),
            "query": test.get("query"),
            "expected_status": expected_status,
            "actual_status": None,
            "expected_domains": expected_domains,
            "actual_domains": [],
            "http_status_code": None,
            "latency_ms": latency,
            "passed": False,
            "timed_out": True,
            "error": f"Request timed out after {timeout}s: {exc}",
        }
    except requests.exceptions.RequestException as exc:
        end = time.time()
        latency = round((end - start) * 1000)
        return {
            "id": test.get("id"),
            "category": test.get("category"),
            "query": test.get("query"),
            "expected_status": expected_status,
            "actual_status": None,
            "expected_domains": expected_domains,
            "actual_domains": [],
            "http_status_code": None,
            "latency_ms": latency,
            "passed": False,
            "timed_out": False,
            "error": f"HTTP request exception: {exc}",
        }
    except Exception as exc:
        return {
            "id": test.get("id"),
            "category": test.get("category"),
            "query": test.get("query"),
            "expected_status": expected_status,
            "actual_status": None,
            "expected_domains": expected_domains,
            "actual_domains": [],
            "http_status_code": None,
            "latency_ms": None,
            "passed": False,
            "timed_out": False,
            "error": f"Unexpected execution error: {exc}",
        }

    http_status_code = res.status_code
    if http_status_code != 200:
        return {
            "id": test.get("id"),
            "category": test.get("category"),
            "query": test.get("query"),
            "expected_status": expected_status,
            "actual_status": None,
            "expected_domains": expected_domains,
            "actual_domains": [],
            "http_status_code": http_status_code,
            "latency_ms": latency,
            "passed": False,
            "timed_out": False,
            "error": f"HTTP {http_status_code}: {res.text[:200]}",
        }

    try:
        data = res.json()
    except Exception as exc:
        return {
            "id": test.get("id"),
            "category": test.get("category"),
            "query": test.get("query"),
            "expected_status": expected_status,
            "actual_status": None,
            "expected_domains": expected_domains,
            "actual_domains": [],
            "http_status_code": http_status_code,
            "latency_ms": latency,
            "passed": False,
            "timed_out": False,
            "error": f"Malformed JSON in response body: {exc}",
        }

    if not isinstance(data, dict):
        return {
            "id": test.get("id"),
            "category": test.get("category"),
            "query": test.get("query"),
            "expected_status": expected_status,
            "actual_status": None,
            "expected_domains": expected_domains,
            "actual_domains": [],
            "http_status_code": http_status_code,
            "latency_ms": latency,
            "passed": False,
            "timed_out": False,
            "error": "Malformed response: root payload is not a JSON object",
        }

    if "status" not in data:
        return {
            "id": test.get("id"),
            "category": test.get("category"),
            "query": test.get("query"),
            "expected_status": expected_status,
            "actual_status": None,
            "expected_domains": expected_domains,
            "actual_domains": [],
            "http_status_code": http_status_code,
            "latency_ms": latency,
            "passed": False,
            "timed_out": False,
            "error": "Malformed response: missing required 'status' field",
        }

    actual_status = data["status"]
    actual_domains_raw = data.get("domains_activated", [])
    if not isinstance(actual_domains_raw, list):
        return {
            "id": test.get("id"),
            "category": test.get("category"),
            "query": test.get("query"),
            "expected_status": expected_status,
            "actual_status": actual_status,
            "expected_domains": expected_domains,
            "actual_domains": [],
            "http_status_code": http_status_code,
            "latency_ms": latency,
            "passed": False,
            "timed_out": False,
            "error": "Malformed response: 'domains_activated' must be a list",
        }

    actual_domains = sorted(actual_domains_raw)

    # ── Strict, non-conflated status verification ─────────────────────────
    # Distinguish blocked, out_of_scope, clarification_needed, and success
    if expected_status == "success":
        passed = (actual_status == "success") and (actual_domains == expected_domains)
    else:
        passed = (actual_status == expected_status)

    error = (
        None
        if passed
        else (
            f"Classification mismatch: expected status '{expected_status}' with domains {expected_domains}, "
            f"got status '{actual_status}' with domains {actual_domains}"
        )
    )

    return {
        "id": test.get("id"),
        "category": test.get("category"),
        "query": test.get("query"),
        "expected_status": expected_status,
        "actual_status": actual_status,
        "expected_domains": expected_domains,
        "actual_domains": actual_domains,
        "http_status_code": http_status_code,
        "latency_ms": latency,
        "passed": passed,
        "timed_out": False,
        "error": error,
    }


def format_console_summary(
    summary_or_result: dict[str, Any],
    per_category: dict[str, dict[str, Any]] | None = None,
) -> str:
    """
    Format evaluation summary into a clean, aligned console report.
    Supports either passing (summary, per_category) or the full eval_result dict.
    """
    if "summary" in summary_or_result and "per_category" in summary_or_result:
        summary = summary_or_result["summary"]
        categories = summary_or_result["per_category"]
    else:
        summary = summary_or_result
        categories = per_category or {}

    total = summary.get("total", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    timed_out = summary.get("timed_out", 0)
    excluded = summary.get("excluded", 0)
    accuracy_pct = summary.get("accuracy_pct", 0.0)

    fastest = summary.get("fastest_latency_ms")
    slowest = summary.get("slowest_latency_ms")
    avg_lat = summary.get("avg_latency_ms")

    fastest_str = f"{fastest}ms" if fastest is not None else "N/A"
    slowest_str = f"{slowest}ms" if slowest is not None else "N/A"
    avg_str = f"{avg_lat}ms" if (avg_lat is not None and fastest is not None) else "N/A"

    col_cat = 24
    col_pass = 15
    col_acc = 12

    divider_char = "─"
    try:
        encoding = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
        divider_char.encode(encoding)
    except Exception:
        divider_char = "-"

    lines = [
        f"Overall accuracy: {passed}/{total} = {accuracy_pct:.1f}%",
        f"Passed: {passed} | Failed: {failed} | Timed Out: {timed_out} | Excluded: {excluded}",
        "",
        "Per-category breakdown:",
        f"  {'Category':<{col_cat}} {'Passed/Total':<{col_pass}} {'Accuracy':<{col_acc}} {'Avg latency'}",
        "  " + divider_char * 65,
    ]

    for cat, counts in categories.items():
        c_pass = counts.get("passed", 0)
        c_tot = counts.get("total", 0)
        c_acc = counts.get("accuracy_pct", 0.0)
        c_lat = counts.get("avg_latency_ms")

        pass_ratio = f"{c_pass}/{c_tot}"
        acc_str = f"{c_acc:.1f}%"
        if c_lat is not None and c_tot > 0:
            lat_str = f"{c_lat}ms"
        else:
            lat_str = "N/A"

        lines.append(f"  {cat:<{col_cat}} {pass_ratio:<{col_pass}} {acc_str:<{col_acc}} {lat_str}")

    lines.extend([
        "",
        "Latency summary:",
        f"  Fastest: {fastest_str}",
        f"  Slowest: {slowest_str}",
        f"  Average: {avg_str}",
    ])

    return "\n".join(lines)


def run_evaluation(
    queries: list[dict] = TEST_QUERIES,
    api_url: str = API,
    base_profile: dict | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    session: requests.Session | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run evaluation across a test query suite.
    Calculates overall and per-category accuracy and latency metrics.
    """
    total = len(queries)
    passed_count = 0
    failed_count = 0
    timed_out_count = 0
    results: list[dict[str, Any]] = []

    if verbose:
        print(f"\nRunning {total} test queries against {api_url}/query...\n")
        print(f"{'ID':<6} {'Category':<15} {'Expected':<22} {'Got':<22} {'ms':<8} {'Status'}")
        print("─" * 90)

    for test in queries:
        r = evaluate_query(
            test=test,
            api_url=api_url,
            base_profile=base_profile,
            session=session,
            timeout=timeout,
        )
        results.append(r)

        if r["passed"]:
            passed_count += 1
            status_str = "PASS"
        elif r["timed_out"]:
            timed_out_count += 1
            status_str = "TIMEOUT"
        else:
            failed_count += 1
            status_str = "FAIL"

        if verbose:
            exp_str = (
                f"{r['expected_status']}:{r['expected_domains']}"
                if r["expected_status"] != "success"
                else str(r["expected_domains"])
            )
            got_str = (
                f"{r['actual_status']}:{r['actual_domains']}"
                if r["actual_status"] != "success"
                else str(r["actual_domains"])
            )
            lat_str = str(r["latency_ms"]) if r["latency_ms"] is not None else "N/A"
            print(f"{test.get('id', 'N/A'):<6} {test.get('category', 'N/A'):<15} {exp_str:<22} {got_str:<22} {lat_str:<8} {status_str}")

    # Latencies strictly tied to responses that completed
    valid_latencies = [
        r["latency_ms"] for r in results if r.get("latency_ms") is not None
    ]

    accuracy_pct = round((passed_count / total) * 100, 1) if total > 0 else 0.0
    avg_latency_ms = (
        round(sum(valid_latencies) / len(valid_latencies))
        if valid_latencies
        else 0
    )

    # Per-category aggregation
    categories: dict[str, dict[str, Any]] = {}
    for r in results:
        cat = r.get("category") or "uncategorized"
        if cat not in categories:
            categories[cat] = {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "timed_out": 0,
                "accuracy_pct": 0.0,
                "avg_latency_ms": 0,
                "latencies": [],
            }
        cat_stats = categories[cat]
        cat_stats["total"] += 1
        if r["passed"]:
            cat_stats["passed"] += 1
        elif r["timed_out"]:
            cat_stats["timed_out"] += 1
        else:
            cat_stats["failed"] += 1

        if r.get("latency_ms") is not None:
            cat_stats["latencies"].append(r["latency_ms"])

    for cat_data in categories.values():
        c_tot = cat_data["total"]
        c_pass = cat_data["passed"]
        c_lats = cat_data.pop("latencies")
        cat_data["accuracy_pct"] = (
            round((c_pass / c_tot) * 100, 1) if c_tot > 0 else 0.0
        )
        cat_data["avg_latency_ms"] = (
            round(sum(c_lats) / len(c_lats)) if c_lats else 0
        )

    summary = {
        "total": total,
        "passed": passed_count,
        "failed": failed_count,
        "timed_out": timed_out_count,
        "excluded": 0,
        "accuracy_pct": accuracy_pct,
        "avg_latency_ms": avg_latency_ms,
        "fastest_latency_ms": min(valid_latencies) if valid_latencies else None,
        "slowest_latency_ms": max(valid_latencies) if valid_latencies else None,
    }

    if verbose:
        sep_char = "═"
        try:
            encoding = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
            sep_char.encode(encoding)
        except Exception:
            sep_char = "="
        print("\n" + sep_char * 90 + "\n")
        print(format_console_summary(summary, categories))

    return {
        "metric_scope": {
            "evaluation_type": "routing_and_safety_classification",
            "target_endpoint": f"{api_url}/query",
            "rag_answer_quality_evaluated": False,
            "tool_execution_accuracy_evaluated": False,
            "notes": (
                "Evaluates intent routing and safety firewall accuracy only. "
                "Does not evaluate factual correctness, groundedness, or text response quality."
            ),
        },
        "summary": summary,
        "per_category": categories,
        "details": results,
    }


def save_evaluation_report(
    eval_payload: dict[str, Any],
    output_dir: str | Path = "benchmark_reports",
    provenance: dict[str, Any] | None = None,
) -> Path:
    """
    Save evaluation results non-destructively to a uniquely named timestamped file.
    Guarantees no historical result is overwritten.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    git_sha = (provenance or {}).get("git_commit", "unknown")
    short_sha = git_sha[:7] if git_sha != "unavailable" else "nogit"

    base_name = f"eval_routing_{timestamp_str}_{short_sha}"
    file_path = out_dir / f"{base_name}.json"

    counter = 1
    while file_path.exists():
        file_path = out_dir / f"{base_name}_{counter}.json"
        counter += 1

    full_report = {
        "provenance": provenance or get_provenance(),
        **eval_payload,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    return file_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate TriDomain Meta-Agent routing and safety accuracy non-destructively."
    )
    parser.add_argument(
        "--api", default=API, help=f"Base API URL (default: {API})"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Timeout in seconds per query (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--dataset",
        choices=["baseline", "additional", "combined"],
        default="baseline",
        help="Dataset to evaluate: 'baseline' (50 queries), 'additional' (200 queries), or 'combined' (250 queries). Default: 'baseline'.",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_reports",
        help="Directory to store timestamped benchmark reports (default: benchmark_reports)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run evaluation without writing report file to disk",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    selected_dataset = args.dataset
    queries = DATASETS[selected_dataset]

    provenance = get_provenance(
        dataset_path="test_queries.py",
        dataset_name=selected_dataset,
        query_count=len(queries),
        endpoint=f"{args.api}/query",
    )

    eval_result = run_evaluation(
        queries=queries,
        api_url=args.api,
        base_profile=BASE_PROFILE,
        timeout=args.timeout,
        verbose=True,
    )

    if not args.no_save:
        saved_file = save_evaluation_report(
            eval_payload=eval_result,
            output_dir=args.output_dir,
            provenance=provenance,
        )
        print(f"\nResults saved non-destructively to: {saved_file}")
        print("Historical file 'evaluation_results.json' was preserved.")


if __name__ == "__main__":
    main()
