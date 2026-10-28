"""
tests/test_benchmark.py
-----------------------
Unit tests for compare_agents.py:
- Speedup and latency-reduction formulas (including edge cases and zero/negative guards)
- Geometric mean aggregation and invalid ratio exclusion
- Summary statistics (mean, median, std_dev, p95, min, max) and empty/single-trial handling
- Warm-up trial isolation and exclusion from reported trial statistics
- Failure handling and unequal sample counts across trials
- Comparability classifications (Finance pass-through consistency, Career/Health architectural disparity)
- Exclusion of invalid or non-equivalent comparisons from framework speedups
- Non-destructive report saving, collision handling, and preservation of historical results
- Provenance recording without credential/secret leakage
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from compare_agents import (
    BenchmarkRequest,
    COMPARABILITY_METADATA,
    DEFAULT_BENCHMARK_QUERIES,
    benchmark_agent_pair,
    calculate_geometric_mean,
    calculate_speedup,
    calculate_statistics,
    get_provenance,
    run_benchmark,
    save_benchmark_report,
)


# ── 1. Speedup & Latency-Reduction Formulas ───────────────────────────────────


def test_calculate_speedup_basic():
    # Baseline: 2000 ms, LangChain: 1000 ms -> 2.0x speedup, 50.0% reduction
    speedup, reduction = calculate_speedup(2000.0, 1000.0)
    assert speedup == 2.0
    assert reduction == 50.0


def test_calculate_speedup_slowdown():
    # Baseline: 1000 ms, LangChain: 2000 ms -> 0.5x speedup, -100.0% reduction
    speedup, reduction = calculate_speedup(1000.0, 2000.0)
    assert speedup == 0.5
    assert reduction == -100.0


def test_calculate_speedup_identical():
    # Baseline: 1500 ms, LangChain: 1500 ms -> 1.0x speedup, 0.0% reduction
    speedup, reduction = calculate_speedup(1500.0, 1500.0)
    assert speedup == 1.0
    assert reduction == 0.0


@pytest.mark.parametrize(
    "baseline,langchain",
    [
        (None, 1000.0),
        (2000.0, None),
        (None, None),
        (0.0, 1000.0),
        (2000.0, 0.0),
        (-500.0, 1000.0),
        (2000.0, -100.0),
        (0.0, 0.0),
    ],
)
def test_calculate_speedup_invalid_inputs(baseline, langchain):
    speedup, reduction = calculate_speedup(baseline, langchain)
    assert speedup is None
    assert reduction is None


# ── 2. Geometric Mean Aggregation ─────────────────────────────────────────────


def test_calculate_geometric_mean_basic():
    # sqrt(2.0 * 8.0) = 4.0
    geo_mean = calculate_geometric_mean([2.0, 8.0])
    assert geo_mean == 4.0


def test_calculate_geometric_mean_single():
    geo_mean = calculate_geometric_mean([3.5])
    assert geo_mean == 3.5


def test_calculate_geometric_mean_filters_invalid_and_nones():
    # Valid: 2.0 and 8.0; None and negative numbers are excluded
    geo_mean = calculate_geometric_mean([2.0, None, -5.0, 0.0, 8.0])
    assert geo_mean == 4.0


def test_calculate_geometric_mean_empty():
    assert calculate_geometric_mean([]) is None
    assert calculate_geometric_mean([None, None]) is None
    assert calculate_geometric_mean([0.0, -1.0]) is None


# ── 3. Summary Statistics ─────────────────────────────────────────────────────


def test_calculate_statistics_empty():
    stats = calculate_statistics([])
    assert stats["count"] == 0
    assert stats["mean_ms"] is None
    assert stats["median_ms"] is None
    assert stats["std_dev_ms"] is None
    assert stats["p95_ms"] is None
    assert stats["min_ms"] is None
    assert stats["max_ms"] is None


def test_calculate_statistics_single_trial():
    stats = calculate_statistics([1500.0])
    assert stats["count"] == 1
    assert stats["mean_ms"] == 1500.0
    assert stats["median_ms"] == 1500.0
    assert stats["std_dev_ms"] == 0.0
    assert stats["p95_ms"] == 1500.0
    assert stats["min_ms"] == 1500.0
    assert stats["max_ms"] == 1500.0


def test_calculate_statistics_odd_count():
    latencies = [100.0, 200.0, 300.0]
    stats = calculate_statistics(latencies)
    assert stats["count"] == 3
    assert stats["mean_ms"] == 200.0
    assert stats["median_ms"] == 200.0
    assert stats["min_ms"] == 100.0
    assert stats["max_ms"] == 300.0
    assert stats["std_dev_ms"] == 100.0  # Sample std dev for [100, 200, 300] is 100.0


def test_calculate_statistics_even_count():
    latencies = [100.0, 200.0, 300.0, 400.0]
    stats = calculate_statistics(latencies)
    assert stats["count"] == 4
    assert stats["mean_ms"] == 250.0
    assert stats["median_ms"] == 250.0
    assert stats["min_ms"] == 100.0
    assert stats["max_ms"] == 400.0


def test_calculate_statistics_p95():
    # 20 samples from 10 to 200
    latencies = [float(i * 10) for i in range(1, 21)]
    stats = calculate_statistics(latencies)
    assert stats["count"] == 20
    # ceil(0.95 * 20) - 1 = 19 - 1 = 18 -> index 18 (19th element: 190.0)
    assert stats["p95_ms"] == 190.0


# ── 4. Warm-Up Isolation and Exclusion ────────────────────────────────────────


def test_benchmark_agent_pair_warmup_excluded():
    baseline_calls = []
    lc_calls = []

    def mock_baseline(req):
        baseline_calls.append(req.query)
        return {"status": "ok"}

    def mock_lc(req):
        lc_calls.append(req.query)
        return {"status": "ok"}

    res = benchmark_agent_pair(
        domain_name="test_domain",
        baseline_fn=mock_baseline,
        lc_fn=mock_lc,
        query="What is my budget?",
        warmup_trials=2,
        measured_trials=3,
        comparability="comparable",
        is_framework_speedup=True,
        verbose=False,
    )

    # Invocations: 2 warmup + 3 measured = 5 calls each
    assert len(baseline_calls) == 5
    assert len(lc_calls) == 5

    # Check that warm-up latencies are strictly separated from measured
    assert len(res["baseline"]["warmup_trials_ms"]) == 2
    assert len(res["baseline"]["measured_trials_ms"]) == 3
    assert res["baseline"]["statistics"]["count"] == 3

    assert len(res["langchain"]["warmup_trials_ms"]) == 2
    assert len(res["langchain"]["measured_trials_ms"]) == 3
    assert res["langchain"]["statistics"]["count"] == 3


# ── 5. Trial Failure Handling and Unequal Counts ──────────────────────────────


def test_benchmark_agent_pair_handles_partial_failures():
    baseline_attempts = 0

    def mock_baseline_flaky(req):
        nonlocal baseline_attempts
        baseline_attempts += 1
        if baseline_attempts == 2:  # Fails on 2nd invocation (1st measured trial)
            raise RuntimeError("Temporary API timeout")
        return {"status": "ok"}

    def mock_lc_success(req):
        return {"status": "ok"}

    res = benchmark_agent_pair(
        domain_name="test_flaky",
        baseline_fn=mock_baseline_flaky,
        lc_fn=mock_lc_success,
        query="Test query",
        warmup_trials=1,
        measured_trials=3,
        verbose=False,
    )

    # Baseline had 1 warmup (success) and 3 measured (1 fail, 2 success)
    assert len(res["baseline"]["errors"]) == 1
    assert "Temporary API timeout" in res["baseline"]["errors"][0]
    assert len(res["baseline"]["measured_trials_ms"]) == 2
    assert res["baseline"]["statistics"]["count"] == 2

    # Langchain had all 3 measured trials succeed
    assert len(res["langchain"]["errors"]) == 0
    assert len(res["langchain"]["measured_trials_ms"]) == 3
    assert res["langchain"]["statistics"]["count"] == 3

    # Speedup is computed from surviving trials without crashing
    assert res["comparison"]["speedup_ratio"] is not None


def test_benchmark_agent_pair_complete_failure():
    def mock_baseline_always_fails(req):
        raise ConnectionError("Service unavailable")

    def mock_lc_success(req):
        return {"status": "ok"}

    res = benchmark_agent_pair(
        domain_name="test_down",
        baseline_fn=mock_baseline_always_fails,
        lc_fn=mock_lc_success,
        query="Test query",
        warmup_trials=1,
        measured_trials=2,
        verbose=False,
    )

    assert len(res["baseline"]["measured_trials_ms"]) == 0
    assert res["baseline"]["statistics"]["count"] == 0
    assert res["baseline"]["statistics"]["mean_ms"] is None
    # Gracefully sets comparison ratios to None
    assert res["comparison"]["speedup_ratio"] is None
    assert res["comparison"]["latency_reduction_pct"] is None


# ── 6. Fairness Classification and Comparability Guardrails ───────────────────


def test_finance_pass_through_classified_correctly():
    def mock_fn(req):
        return {"status": "ok"}

    res = benchmark_agent_pair(
        domain_name="finance",
        baseline_fn=mock_fn,
        lc_fn=mock_fn,
        query="Manage my debt",
        warmup_trials=0,
        measured_trials=2,
        comparability="pass_through_consistency",
        is_framework_speedup=False,
        notes=COMPARABILITY_METADATA["finance"]["notes"],
        verbose=False,
    )

    # Must NOT report speedup_ratio or latency_reduction_pct for pass-through consistency
    assert res["comparison"]["speedup_ratio"] is None
    assert res["comparison"]["latency_reduction_pct"] is None
    # Must report pipeline_consistency_ratio instead
    assert res["comparison"]["pipeline_consistency_ratio"] is not None
    assert res["is_framework_speedup"] is False
    assert "thin wrapper" in res["notes"]


def test_career_health_architectural_disparity_flagged():
    def mock_fn(req):
        return {"status": "ok"}

    for domain in ["career", "health"]:
        meta = COMPARABILITY_METADATA[domain]
        res = benchmark_agent_pair(
            domain_name=domain,
            baseline_fn=mock_fn,
            lc_fn=mock_fn,
            query="Test query",
            warmup_trials=0,
            measured_trials=2,
            comparability=meta["comparability"],
            is_framework_speedup=meta["is_framework_speedup"],
            notes=meta["notes"],
            verbose=False,
        )

        assert res["comparability"] == "architectural_disparity"
        assert res["is_framework_speedup"] is False
        assert len(res["notes"]) > 0


def test_run_benchmark_aggregate_withheld_when_no_framework_speedups():
    def mock_fn(req):
        return {"status": "ok"}

    mock_pairs = {
        "career": (mock_fn, mock_fn),
        "health": (mock_fn, mock_fn),
        "finance": (mock_fn, mock_fn),
    }

    report = run_benchmark(
        warmup_trials=0,
        measured_trials=1,
        agent_pairs=mock_pairs,
        verbose=False,
    )

    summary = report["summary"]
    assert summary["domains_benchmarked"] == 3
    assert summary["valid_framework_speedup_comparisons_count"] == 0
    assert summary["aggregate_framework_speedup_geometric_mean"] is None
    assert "withheld" in summary["aggregate_explanation"]


def test_run_benchmark_aggregate_included_only_when_valid():
    def mock_fn(req):
        return {"status": "ok"}

    mock_pairs = {
        "career": (mock_fn, mock_fn),
    }

    # Temporarily patch COMPARABILITY_METADATA to simulate a genuinely comparable domain
    custom_metadata = {
        "career": {
            "comparability": "comparable",
            "is_framework_speedup": True,
            "notes": "Verified equivalent workload",
        }
    }

    with patch.dict("compare_agents.COMPARABILITY_METADATA", custom_metadata):
        report = run_benchmark(
            warmup_trials=0,
            measured_trials=2,
            agent_pairs=mock_pairs,
            verbose=False,
        )

        summary = report["summary"]
        assert summary["valid_framework_speedup_comparisons_count"] == 1
        assert summary["aggregate_framework_speedup_geometric_mean"] is not None


# ── 7. Non-Destructive Report Saving & Provenance ──────────────────────────────


def test_save_benchmark_report_creates_unique_file(tmp_path: Path):
    dummy_payload = {
        "summary": {"domains_benchmarked": 3},
        "domains": [],
    }
    dummy_provenance = {
        "timestamp_utc": "2026-10-19T18:00:00Z",
        "git_commit": "abc1234567890",
        "git_branch": "feature/benchmark",
        "python_version": "3.11.0",
        "system_platform": "Windows-10",
        "timer": "time.perf_counter()",
    }

    path1 = save_benchmark_report(
        benchmark_payload=dummy_payload,
        output_dir=tmp_path,
        provenance=dummy_provenance,
    )
    assert path1.exists()
    assert path1.suffix == ".json"

    # Save a second time with exact same timestamp and sha -> anti-collision suffix
    path2 = save_benchmark_report(
        benchmark_payload=dummy_payload,
        output_dir=tmp_path,
        provenance=dummy_provenance,
    )
    assert path2.exists()
    assert path1 != path2
    assert path2.name.endswith("_1.json")

    # Verify content was saved accurately
    with open(path1, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["summary"]["domains_benchmarked"] == 3
    assert data["provenance"]["git_commit"] == "abc1234567890"


def test_historical_results_never_overwritten(tmp_path: Path):
    # Place dummy historical files in repo-like structure
    eval_results_file = tmp_path / "evaluation_results.json"
    comp_results_file = tmp_path / "comparison_results.json"
    eval_results_file.write_text('{"original": "evaluation"}', encoding="utf-8")
    comp_results_file.write_text('{"original": "comparison"}', encoding="utf-8")

    dummy_payload = {"summary": {}, "domains": []}
    # save_benchmark_report writes to a subdirectory (like benchmark_reports)
    reports_dir = tmp_path / "benchmark_reports"
    saved_file = save_benchmark_report(
        benchmark_payload=dummy_payload,
        output_dir=reports_dir,
    )

    # Ensure historical files were untouched
    assert eval_results_file.read_text(encoding="utf-8") == '{"original": "evaluation"}'
    assert comp_results_file.read_text(encoding="utf-8") == '{"original": "comparison"}'
    assert saved_file.parent == reports_dir


def test_provenance_contains_no_secrets():
    prov = get_provenance()
    assert "timestamp_utc" in prov
    assert "python_version" in prov
    assert "timer" in prov

    # Serialize to JSON and check for typical secret keys
    prov_str = json.dumps(prov).lower()
    for secret_keyword in ["api_key", "secret", "password", "token", "gsk_"]:
        assert secret_keyword not in prov_str
