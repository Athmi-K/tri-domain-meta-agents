"""
compare_agents.py
-----------------
Comparative latency and fairness benchmark between TriDomain baseline agents
and LangChain implementations across Career, Health, and Finance domains.

Methodology & Fairness Principles:
- High-resolution monotonic timing using time.perf_counter().
- Configurable warm-up runs (separated from reported statistics).
- Multi-trial execution recording mean, median, std_dev, min, max, and P95.
- Explicit comparability classification:
    * Career & Health: Documented architectural disparities (tool omission / ReAct loops).
    * Finance: Identified as pass-through consistency check, NOT a framework speedup.
- Geometric mean aggregation for valid comparable speedup ratios.
- Non-destructive result persistence in benchmark_reports/ (never overwrites historical results).
- Zero secret / credential leakage in benchmark reports.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


class BenchmarkRequest:
    """Mock profile request for standalone domain benchmarking."""

    name: str = "Arjun"
    age: int = 28
    current_skills: list[str] = ["python", "sql"]
    target_role: str = "data scientist"
    location: str = "Bangalore"
    experience_level: str = "mid"
    years_experience: int = 3
    current_level: str = "beginner"
    timeline_months: int = 6
    resume_text: str = ""
    weight_kg: float = 75.0
    height_cm: float = 175.0
    sleep_hours: float = 7.0
    sleep_quality: int = 7
    stress_level: int = 5
    mood_score: int = 7
    anxiety_level: int = 4
    fitness_goal: str = "general fitness"
    available_days: int = 3
    monthly_income: float = 50000.0
    monthly_expenses: float = 35000.0
    risk_tolerance: str = "moderate"
    retirement_age: int = 60
    retirement_savings: float = 0.0
    monthly_contribution: float = 0.0
    query: str = ""

    def __init__(self, query: str = ""):
        self.query = query


DEFAULT_BENCHMARK_QUERIES = {
    "career": "How do I become a data scientist?",
    "health": "I am always tired and stressed",
    "finance": "How should I manage my debt?",
}

COMPARABILITY_METADATA = {
    "career": {
        "comparability": "architectural_disparity",
        "is_framework_speedup": False,
        "notes": (
            "Baseline executes 5 calculator tools + semantic search + JSearch ranking embeddings + RAG + LLM. "
            "LangChain skips RAG (query lacks planning trigger keywords) and executes a ReAct agent with 0-1 tools. "
            "Observed latency differences reflect tool/embedding omission rather than framework efficiency."
        ),
    },
    "health": {
        "comparability": "architectural_disparity",
        "is_framework_speedup": False,
        "notes": (
            "Baseline runs pure Python math in <5ms + RAG + a single Groq LLM completion. "
            "LangChain executes a multi-step ReAct agent requiring 2-3 sequential round-trip calls to Groq API. "
            "Observed slowdown reflects sequential ReAct turn overhead, not framework inefficiency."
        ),
    },
    "finance": {
        "comparability": "pass_through_consistency",
        "is_framework_speedup": False,
        "notes": (
            "LangChain finance agent (finance_lc_agent.py) is a thin wrapper calling agents.finance_agent.run() directly. "
            "Both paths execute the exact same Python code. Measured latency differences reflect Groq API network variance between trials."
        ),
    },
}


def calculate_statistics(latencies: list[float]) -> dict[str, Any]:
    """
    Calculate summary statistics over measured latency trials.
    Returns None for statistical fields if latencies list is empty.
    """
    if not latencies:
        return {
            "count": 0,
            "mean_ms": None,
            "median_ms": None,
            "std_dev_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
        }

    sorted_lats = sorted(latencies)
    count = len(sorted_lats)
    mean_val = sum(sorted_lats) / count

    # Median
    if count % 2 == 1:
        median_val = sorted_lats[count // 2]
    else:
        median_val = (sorted_lats[count // 2 - 1] + sorted_lats[count // 2]) / 2.0

    # Sample standard deviation (N - 1 degrees of freedom for N >= 2)
    if count >= 2:
        variance = sum((x - mean_val) ** 2 for x in sorted_lats) / (count - 1)
        std_dev_val = math.sqrt(variance)
    else:
        std_dev_val = 0.0

    # 95th Percentile (Nearest rank / rank ceiling)
    p95_idx = max(0, min(count - 1, math.ceil(0.95 * count) - 1))
    p95_val = sorted_lats[p95_idx]

    return {
        "count": count,
        "mean_ms": round(mean_val, 2),
        "median_ms": round(median_val, 2),
        "std_dev_ms": round(std_dev_val, 2),
        "p95_ms": round(p95_val, 2),
        "min_ms": round(sorted_lats[0], 2),
        "max_ms": round(sorted_lats[-1], 2),
    }


def calculate_speedup(
    baseline_mean: float | None,
    langchain_mean: float | None,
) -> tuple[float | None, float | None]:
    """
    Calculate speedup ratio and percentage latency reduction.
    Speedup = baseline_latency / langchain_latency.
    Latency Reduction % = ((baseline - langchain) / baseline) * 100.
    Handles missing, zero, or negative inputs safely by returning (None, None).
    """
    if (
        baseline_mean is None
        or langchain_mean is None
        or baseline_mean <= 0
        or langchain_mean <= 0
    ):
        return None, None

    speedup_ratio = round(baseline_mean / langchain_mean, 2)
    latency_reduction_pct = round(
        ((baseline_mean - langchain_mean) / baseline_mean) * 100, 1
    )
    return speedup_ratio, latency_reduction_pct


def calculate_geometric_mean(ratios: list[float | None]) -> float | None:
    """
    Calculate the geometric mean of ratio metrics.
    Excludes None, non-positive, or invalid ratios.
    """
    valid = [r for r in ratios if r is not None and r > 0]
    if not valid:
        return None
    # Compute via log-sum to prevent numeric overflow
    log_sum = sum(math.log(r) for r in valid)
    return round(math.exp(log_sum / len(valid)), 2)


def get_provenance(
    benchmark_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture code revision and environment metadata without inventing values."""
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
        "python_version": sys.version.split()[0],
        "system_platform": platform.platform(),
        "model_configured": model_configured,
        "timer": "time.perf_counter()",
        "benchmark_config": benchmark_config or {},
    }


def benchmark_agent_pair(
    domain_name: str,
    baseline_fn: Callable[[Any], Any],
    lc_fn: Callable[[Any], Any],
    query: str,
    warmup_trials: int = 1,
    measured_trials: int = 3,
    comparability: str = "comparable",
    is_framework_speedup: bool = False,
    notes: str = "",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Execute high-resolution benchmark trials for a single domain agent pair.
    Warm-up runs are recorded separately and excluded from measured statistics.
    """
    if verbose:
        print(f"\n{'=' * 65}")
        print(f"DOMAIN: {domain_name.upper()} | Comparability: {comparability}")
        print(f"Query: \"{query}\"")
        print(f"Plan: {warmup_trials} warm-up run(s), {measured_trials} measured trial(s)")
        print(f"{'=' * 65}")

    # ── 1. Baseline Trials ─────────────────────────────────────────
    baseline_warmup_ms: list[float] = []
    baseline_measured_ms: list[float] = []
    baseline_errors: list[str] = []

    for w in range(warmup_trials):
        req = BenchmarkRequest(query=query)
        try:
            t0 = time.perf_counter()
            baseline_fn(req)
            t1 = time.perf_counter()
            baseline_warmup_ms.append(round(max((t1 - t0) * 1000, 0.01), 2))
        except Exception as exc:
            baseline_errors.append(f"Baseline warmup {w+1} error: {exc}")

    for t in range(measured_trials):
        req = BenchmarkRequest(query=query)
        try:
            t0 = time.perf_counter()
            baseline_fn(req)
            t1 = time.perf_counter()
            elapsed = round(max((t1 - t0) * 1000, 0.01), 2)
            baseline_measured_ms.append(elapsed)
            if verbose:
                print(f"  [Baseline] Trial {t+1}/{measured_trials}: {elapsed} ms")
        except Exception as exc:
            baseline_errors.append(f"Baseline trial {t+1} error: {exc}")
            if verbose:
                print(f"  [Baseline] Trial {t+1}/{measured_trials}: FAILED ({exc})")

    # ── 2. LangChain Trials ───────────────────────────────────────
    lc_warmup_ms: list[float] = []
    lc_measured_ms: list[float] = []
    lc_errors: list[str] = []

    for w in range(warmup_trials):
        req = BenchmarkRequest(query=query)
        try:
            t0 = time.perf_counter()
            lc_fn(req)
            t1 = time.perf_counter()
            lc_warmup_ms.append(round(max((t1 - t0) * 1000, 0.01), 2))
        except Exception as exc:
            lc_errors.append(f"LangChain warmup {w+1} error: {exc}")

    for t in range(measured_trials):
        req = BenchmarkRequest(query=query)
        try:
            t0 = time.perf_counter()
            lc_fn(req)
            t1 = time.perf_counter()
            elapsed = round(max((t1 - t0) * 1000, 0.01), 2)
            lc_measured_ms.append(elapsed)
            if verbose:
                print(f"  [LangChain] Trial {t+1}/{measured_trials}: {elapsed} ms")
        except Exception as exc:
            lc_errors.append(f"LangChain trial {t+1} error: {exc}")
            if verbose:
                print(f"  [LangChain] Trial {t+1}/{measured_trials}: FAILED ({exc})")

    # ── 3. Summary Calculations ───────────────────────────────────
    baseline_stats = calculate_statistics(baseline_measured_ms)
    lc_stats = calculate_statistics(lc_measured_ms)

    # Speedup and Latency Reduction
    raw_ratio, raw_reduction = calculate_speedup(
        baseline_stats["mean_ms"], lc_stats["mean_ms"]
    )

    # Classify metrics based on fairness and equivalence
    if comparability == "comparable" and is_framework_speedup:
        speedup_ratio = raw_ratio
        latency_reduction_pct = raw_reduction
        pipeline_consistency_ratio = None
    elif comparability == "pass_through_consistency":
        speedup_ratio = None
        latency_reduction_pct = None
        pipeline_consistency_ratio = raw_ratio
    else:
        # Architectural disparity: record workload ratio but flag non-framework attribution
        speedup_ratio = raw_ratio
        latency_reduction_pct = raw_reduction
        pipeline_consistency_ratio = None

    if verbose:
        print(f"\n  SUMMARY ({domain_name.upper()}):")
        print(f"    Baseline Mean : {baseline_stats['mean_ms']} ms (median: {baseline_stats['median_ms']} ms, std_dev: {baseline_stats['std_dev_ms']} ms)")
        print(f"    LangChain Mean: {lc_stats['mean_ms']} ms (median: {lc_stats['median_ms']} ms, std_dev: {lc_stats['std_dev_ms']} ms)")
        if comparability == "pass_through_consistency":
            print(f"    Consistency Ratio: {pipeline_consistency_ratio}x (Identical pipeline; network variance only)")
        else:
            print(f"    Workload Speedup: {speedup_ratio}x ({latency_reduction_pct}% latency reduction)")
            if not is_framework_speedup:
                print(f"    [NOTE] {notes}")

    return {
        "domain": domain_name,
        "query": query,
        "comparability": comparability,
        "is_framework_speedup": is_framework_speedup,
        "notes": notes,
        "baseline": {
            "warmup_trials_ms": baseline_warmup_ms,
            "measured_trials_ms": baseline_measured_ms,
            "statistics": baseline_stats,
            "errors": baseline_errors,
        },
        "langchain": {
            "warmup_trials_ms": lc_warmup_ms,
            "measured_trials_ms": lc_measured_ms,
            "statistics": lc_stats,
            "errors": lc_errors,
        },
        "comparison": {
            "speedup_ratio": speedup_ratio,
            "latency_reduction_pct": latency_reduction_pct,
            "pipeline_consistency_ratio": pipeline_consistency_ratio,
        },
    }


def get_default_agent_pairs() -> dict[str, tuple[Callable[[Any], Any], Callable[[Any], Any]]]:
    """Lazy loader for default agent functions to prevent top-level import overhead."""
    import agents.career_agent as career_original
    import agents.finance_agent as finance_original
    import agents.health_agent as health_original
    from langchain_agents.career_lc_agent import run as career_lc_run
    from langchain_agents.finance_lc_agent import run as finance_lc_run
    from langchain_agents.health_lc_agent import run as health_lc_run

    return {
        "career": (career_original.run, career_lc_run),
        "health": (health_original.run, health_lc_run),
        "finance": (finance_original.run, finance_lc_run),
    }


def run_benchmark(
    warmup_trials: int = 1,
    measured_trials: int = 3,
    queries: dict[str, str] | None = None,
    agent_pairs: dict[str, tuple[Callable[[Any], Any], Callable[[Any], Any]]] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Execute latency benchmark across Career, Health, and Finance agent pairs.
    """
    benchmark_queries = queries or DEFAULT_BENCHMARK_QUERIES
    pairs = agent_pairs or get_default_agent_pairs()

    domain_results: list[dict[str, Any]] = []

    for domain_name in ["career", "health", "finance"]:
        if domain_name not in pairs or domain_name not in benchmark_queries:
            continue

        baseline_fn, lc_fn = pairs[domain_name]
        meta = COMPARABILITY_METADATA.get(domain_name, {})
        query = benchmark_queries[domain_name]

        res = benchmark_agent_pair(
            domain_name=domain_name,
            baseline_fn=baseline_fn,
            lc_fn=lc_fn,
            query=query,
            warmup_trials=warmup_trials,
            measured_trials=measured_trials,
            comparability=meta.get("comparability", "comparable"),
            is_framework_speedup=meta.get("is_framework_speedup", False),
            notes=meta.get("notes", ""),
            verbose=verbose,
        )
        domain_results.append(res)

    # ── Summary & Aggregate Calculations ──────────────────────────
    # Only calculate geometric mean for comparisons classified as genuine framework speedup
    framework_speedups = [
        r["comparison"]["speedup_ratio"]
        for r in domain_results
        if r.get("is_framework_speedup") and r["comparison"].get("speedup_ratio") is not None
    ]

    geometric_mean_speedup = calculate_geometric_mean(framework_speedups)

    summary = {
        "domains_benchmarked": len(domain_results),
        "warmup_trials_per_domain": warmup_trials,
        "measured_trials_per_domain": measured_trials,
        "valid_framework_speedup_comparisons_count": len(framework_speedups),
        "aggregate_framework_speedup_geometric_mean": geometric_mean_speedup,
        "aggregate_explanation": (
            "No domain currently features genuinely equivalent workloads between Baseline and LangChain without confounding factors "
            "(Career omits heavy tool/embedding pipelines; Health executes multi-turn ReAct loops vs. single-turn LLM; "
            "Finance delegates directly to baseline). An aggregate framework speedup is therefore withheld to prevent misrepresentation."
            if not framework_speedups
            else f"Geometric mean across {len(framework_speedups)} genuinely comparable domains."
        ),
    }

    if verbose:
        print(f"\n{'=' * 65}")
        print("OVERALL BENCHMARK SUMMARY")
        print(f"{'=' * 65}")
        print(f"Domains evaluated: {summary['domains_benchmarked']}")
        print(f"Trials per domain: {measured_trials} (plus {warmup_trials} warm-up)")
        print(f"Valid framework speedup comparisons: {summary['valid_framework_speedup_comparisons_count']}")
        print(f"Aggregate explanation: {summary['aggregate_explanation']}")

    return {
        "summary": summary,
        "domains": domain_results,
    }


def save_benchmark_report(
    benchmark_payload: dict[str, Any],
    output_dir: str | Path = "benchmark_reports",
    provenance: dict[str, Any] | None = None,
) -> Path:
    """
    Save benchmark results non-destructively to a uniquely named timestamped file.
    Guarantees comparison_results.json and evaluation_results.json are never overwritten.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    git_sha = (provenance or {}).get("git_commit", "unknown")
    short_sha = git_sha[:7] if git_sha != "unavailable" else "nogit"

    base_name = f"benchmark_latency_{timestamp_str}_{short_sha}"
    file_path = out_dir / f"{base_name}.json"

    counter = 1
    while file_path.exists():
        file_path = out_dir / f"{base_name}_{counter}.json"
        counter += 1

    full_report = {
        "provenance": provenance or get_provenance(),
        **benchmark_payload,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    return file_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run rigorous latency and fairness benchmark between baseline and LangChain agents."
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warm-up runs per domain (default: 1)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of measured trials per domain (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_reports",
        help="Directory to save timestamped benchmark report (default: benchmark_reports)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run benchmark without saving JSON report to disk",
    )
    args = parser.parse_args()

    benchmark_config = {
        "warmup_trials": args.warmup,
        "measured_trials": args.trials,
        "domains": ["career", "health", "finance"],
    }

    provenance = get_provenance(benchmark_config=benchmark_config)

    print("\nStarting TriDomain Agent Latency Benchmark...")
    report = run_benchmark(
        warmup_trials=args.warmup,
        measured_trials=args.trials,
        verbose=True,
    )

    if not args.no_save:
        saved_path = save_benchmark_report(
            benchmark_payload=report,
            output_dir=args.output_dir,
            provenance=provenance,
        )
        print(f"\nReport saved non-destructively to: {saved_path}")
        print("Historical result files (comparison_results.json, evaluation_results.json) were preserved.")


if __name__ == "__main__":
    main()
