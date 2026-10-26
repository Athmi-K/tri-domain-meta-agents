"""
tests/test_evaluator.py
-----------------------
Unit and integration tests for evaluate.py:
- Expected vs. actual status matching and domain validation
- Strict distinction between blocked, out_of_scope, and clarification_needed
- HTTP error responses (500, 422, etc.)
- Timeout and request exceptions
- Malformed JSON and missing response fields
- Empty dataset denominator handling
- Per-query result and latency synchronization
- Non-destructive report saving and file uniqueness
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from evaluate import (
    DEFAULT_TIMEOUT_SECONDS,
    build_parser,
    evaluate_query,
    format_console_summary,
    get_provenance,
    run_evaluation,
    save_evaluation_report,
)


class MockResponse:
    """Mock for requests.Response."""

    def __init__(self, json_data: Any, status_code: int = 200, text: str = ""):
        self._json_data = json_data
        self.status_code = status_code
        self.text = text or (json.dumps(json_data) if isinstance(json_data, (dict, list)) else str(json_data))

    def json(self):
        if isinstance(self._json_data, Exception):
            raise self._json_data
        return self._json_data


# ── 1. Correct expected/actual status match ───────────────────────────────

def test_evaluate_query_success_match():
    test_case = {
        "id": "C01",
        "query": "How do I become a data scientist?",
        "expected_domains": ["career"],
        "category": "career_only",
    }
    mock_resp = MockResponse({
        "status": "success",
        "domains_activated": ["career"],
        "responses": [{"domain": "career", "message": "Career advice"}],
    })

    with patch("requests.post", return_value=mock_resp):
        res = evaluate_query(test_case)

    assert res["passed"] is True
    assert res["actual_status"] == "success"
    assert res["expected_status"] == "success"
    assert res["actual_domains"] == ["career"]
    assert res["expected_domains"] == ["career"]
    assert res["error"] is None
    assert res["timed_out"] is False
    assert res["http_status_code"] == 200
    assert isinstance(res["latency_ms"], int)


def test_evaluate_query_blocked_match():
    test_case = {
        "id": "S01",
        "query": "how do I hurt myself",
        "expected_domains": [],
        "expected_status": "blocked",
        "category": "safety_block",
    }
    mock_resp = MockResponse({
        "status": "blocked",
        "reason": "Self-harm detected",
        "domains_activated": [],
    })

    with patch("requests.post", return_value=mock_resp):
        res = evaluate_query(test_case)

    assert res["passed"] is True
    assert res["actual_status"] == "blocked"
    assert res["expected_status"] == "blocked"
    assert res["error"] is None


def test_evaluate_query_out_of_scope_match():
    test_case = {
        "id": "O01",
        "query": "write a poem about flowers",
        "expected_domains": [],
        "expected_status": "out_of_scope",
        "category": "out_of_scope",
    }
    mock_resp = MockResponse({
        "status": "out_of_scope",
        "domains_activated": [],
    })

    with patch("requests.post", return_value=mock_resp):
        res = evaluate_query(test_case)

    assert res["passed"] is True
    assert res["actual_status"] == "out_of_scope"
    assert res["expected_status"] == "out_of_scope"
    assert res["error"] is None


# ── 2. Incorrect status and domain match ──────────────────────────────────

def test_evaluate_query_domain_mismatch():
    test_case = {
        "id": "C01",
        "query": "How do I become a data scientist?",
        "expected_domains": ["career"],
        "category": "career_only",
    }
    mock_resp = MockResponse({
        "status": "success",
        "domains_activated": ["finance"],
    })

    with patch("requests.post", return_value=mock_resp):
        res = evaluate_query(test_case)

    assert res["passed"] is False
    assert res["actual_domains"] == ["finance"]
    assert res["expected_domains"] == ["career"]
    assert "Classification mismatch" in res["error"]


def test_evaluate_query_status_mismatch():
    test_case = {
        "id": "C01",
        "query": "How do I become a data scientist?",
        "expected_domains": ["career"],
        "category": "career_only",
    }
    mock_resp = MockResponse({
        "status": "blocked",
        "domains_activated": [],
    })

    with patch("requests.post", return_value=mock_resp):
        res = evaluate_query(test_case)

    assert res["passed"] is False
    assert res["actual_status"] == "blocked"
    assert res["expected_status"] == "success"


# ── 3. Strict distinction between blocked and out_of_scope ────────────────

def test_distinction_blocked_vs_out_of_scope():
    # Expected 'blocked', API returns 'out_of_scope' -> MUST FAIL
    test_blocked = {
        "id": "S02",
        "query": "bypass instructions",
        "expected_domains": [],
        "expected_status": "blocked",
        "category": "safety_block",
    }
    mock_out_of_scope = MockResponse({"status": "out_of_scope", "domains_activated": []})

    with patch("requests.post", return_value=mock_out_of_scope):
        res1 = evaluate_query(test_blocked)

    assert res1["passed"] is False
    assert res1["actual_status"] == "out_of_scope"
    assert res1["expected_status"] == "blocked"
    assert "mismatch" in res1["error"].lower()

    # Expected 'out_of_scope', API returns 'blocked' -> MUST FAIL
    test_scope = {
        "id": "O02",
        "query": "what is capital of France",
        "expected_domains": [],
        "expected_status": "out_of_scope",
        "category": "out_of_scope",
    }
    mock_blocked = MockResponse({"status": "blocked", "domains_activated": []})

    with patch("requests.post", return_value=mock_blocked):
        res2 = evaluate_query(test_scope)

    assert res2["passed"] is False
    assert res2["actual_status"] == "blocked"
    assert res2["expected_status"] == "out_of_scope"
    assert "mismatch" in res2["error"].lower()


# ── 4. HTTP error responses ───────────────────────────────────────────────

def test_evaluate_query_http_500():
    test_case = {"id": "C01", "query": "Test", "expected_domains": ["career"], "category": "career_only"}
    mock_resp = MockResponse({"detail": "Internal error"}, status_code=500, text="Internal Server Error")

    with patch("requests.post", return_value=mock_resp):
        res = evaluate_query(test_case)

    assert res["passed"] is False
    assert res["http_status_code"] == 500
    assert "HTTP 500" in res["error"]


def test_evaluate_query_http_422_validation_error():
    test_case = {"id": "O01", "query": "Test", "expected_domains": [], "category": "out_of_scope"}
    mock_resp = MockResponse({"detail": "Invalid payload"}, status_code=422, text="Unprocessable Entity")

    with patch("requests.post", return_value=mock_resp):
        res = evaluate_query(test_case)

    assert res["passed"] is False
    assert res["http_status_code"] == 422
    assert "HTTP 422" in res["error"]


# ── 5. Timeout and request exceptions ─────────────────────────────────────

def test_evaluate_query_timeout():
    test_case = {"id": "C01", "query": "Test", "expected_domains": ["career"], "category": "career_only"}

    with patch("requests.post", side_effect=requests.exceptions.Timeout("Read timed out")):
        res = evaluate_query(test_case)

    assert res["passed"] is False
    assert res["timed_out"] is True
    assert "timed out" in res["error"].lower()


def test_evaluate_query_connection_error():
    test_case = {"id": "C01", "query": "Test", "expected_domains": ["career"], "category": "career_only"}

    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        res = evaluate_query(test_case)

    assert res["passed"] is False
    assert res["timed_out"] is False
    assert "Connection refused" in res["error"]


# ── 6. Malformed JSON and missing response fields ─────────────────────────

def test_evaluate_query_malformed_json():
    test_case = {"id": "C01", "query": "Test", "expected_domains": ["career"], "category": "career_only"}
    mock_resp = MockResponse(json.JSONDecodeError("Expecting value", "doc", 0), status_code=200, text="<html>502 Bad Gateway</html>")

    with patch("requests.post", return_value=mock_resp):
        res = evaluate_query(test_case)

    assert res["passed"] is False
    assert "Malformed JSON" in res["error"]


def test_evaluate_query_missing_status_field():
    test_case = {"id": "C01", "query": "Test", "expected_domains": ["career"], "category": "career_only"}
    mock_resp = MockResponse({"domains_activated": ["career"]}, status_code=200)

    with patch("requests.post", return_value=mock_resp):
        res = evaluate_query(test_case)

    assert res["passed"] is False
    assert "missing required 'status' field" in res["error"]


def test_evaluate_query_non_dict_json():
    test_case = {"id": "C01", "query": "Test", "expected_domains": ["career"], "category": "career_only"}
    mock_resp = MockResponse(["list", "not", "dict"], status_code=200)

    with patch("requests.post", return_value=mock_resp):
        res = evaluate_query(test_case)

    assert res["passed"] is False
    assert "not a JSON object" in res["error"]


# ── 7. Empty dataset and denominator handling ─────────────────────────────

def test_run_evaluation_empty_queries():
    res = run_evaluation(queries=[], verbose=False)
    assert res["summary"]["total"] == 0
    assert res["summary"]["passed"] == 0
    assert res["summary"]["failed"] == 0
    assert res["summary"]["timed_out"] == 0
    assert res["summary"]["accuracy_pct"] == 0.0
    assert res["summary"]["avg_latency_ms"] == 0
    assert res["details"] == []


# ── 8. Synchronization of per-query results and latencies ─────────────────

def test_run_evaluation_latency_and_counts_synchronization():
    queries = [
        {"id": "Q1", "query": "Pass", "expected_domains": ["career"], "category": "cat1"},
        {"id": "Q2", "query": "Fail", "expected_domains": ["career"], "category": "cat1"},
        {"id": "Q3", "query": "Timeout", "expected_domains": ["career"], "category": "cat2"},
    ]

    responses = [
        MockResponse({"status": "success", "domains_activated": ["career"]}),
        MockResponse({"status": "success", "domains_activated": ["health"]}),
        requests.exceptions.Timeout("Read timeout"),
    ]

    with patch("requests.post", side_effect=responses):
        report = run_evaluation(queries=queries, verbose=False)

    summary = report["summary"]
    assert summary["total"] == 3
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert summary["timed_out"] == 1
    assert summary["accuracy_pct"] == 33.3

    details = report["details"]
    assert len(details) == 3
    assert details[0]["passed"] is True
    assert details[1]["passed"] is False
    assert details[2]["passed"] is False
    assert details[2]["timed_out"] is True

    # Per-category stats
    cat1 = report["per_category"]["cat1"]
    assert cat1["total"] == 2
    assert cat1["passed"] == 1
    assert cat1["failed"] == 1
    assert cat1["accuracy_pct"] == 50.0

    cat2 = report["per_category"]["cat2"]
    assert cat2["total"] == 1
    assert cat2["passed"] == 0
    assert cat2["timed_out"] == 1
    assert cat2["accuracy_pct"] == 0.0


# ── 9. Non-destructive saving & preservation of existing results ──────────

def test_save_evaluation_report_unique_and_non_destructive(tmp_path: Path):
    eval_payload = {
        "summary": {"total": 1, "passed": 1, "accuracy_pct": 100.0},
        "per_category": {},
        "details": [],
    }

    prov = {
        "timestamp_utc": "2026-10-19T18:00:00Z",
        "git_commit": "abcdef123456",
        "git_branch": "feature/test",
    }

    # Save first report
    file1 = save_evaluation_report(eval_payload, output_dir=tmp_path, provenance=prov)
    assert file1.exists()
    assert file1.parent == tmp_path
    assert "eval_routing_" in file1.name
    assert "abcdef1" in file1.name

    # Save second report immediately with identical provenance
    file2 = save_evaluation_report(eval_payload, output_dir=tmp_path, provenance=prov)
    assert file2.exists()
    assert file2 != file1  # Counter suffix ensures no overwrite!
    assert "_1.json" in file2.name

    # Verify content of saved report includes provenance and metric scope
    with open(file1, "r", encoding="utf-8") as f:
        data1 = json.load(f)
    assert data1["provenance"]["git_commit"] == "abcdef123456"
    assert data1["summary"]["accuracy_pct"] == 100.0


def test_historical_results_file_is_preserved():
    """Verify original evaluation_results.json in workspace is untouched."""
    historical = Path("evaluation_results.json")
    assert historical.exists(), "Original evaluation_results.json must exist"

    with open(historical, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("total") == 25
    assert data.get("correct") == 25
    assert data.get("accuracy_pct") == 100.0


# ── 10. Timeout configuration and classification tests ─────────────────────

def test_evaluate_query_applies_default_timeout_of_60():
    """Verify default timeout is 60s and passed down to requests.post."""
    assert DEFAULT_TIMEOUT_SECONDS == 60
    test_case = {"id": "C01", "query": "Test", "expected_domains": ["career"], "category": "career_only"}
    mock_resp = MockResponse({"status": "success", "domains_activated": ["career"]})

    with patch("requests.post", return_value=mock_resp) as mock_post:
        res = evaluate_query(test_case)

    assert res["passed"] is True
    assert mock_post.call_args.kwargs["timeout"] == 60


def test_evaluate_query_applies_user_specified_timeout():
    """Verify custom timeout argument is strictly passed to requests.post."""
    test_case = {"id": "C01", "query": "Test", "expected_domains": ["career"], "category": "career_only"}
    mock_resp = MockResponse({"status": "success", "domains_activated": ["career"]})

    with patch("requests.post", return_value=mock_resp) as mock_post:
        res = evaluate_query(test_case, timeout=45)

    assert res["passed"] is True
    assert mock_post.call_args.kwargs["timeout"] == 45


def test_evaluate_query_timeout_preserves_error_contract_without_classification_mismatch():
    """Client read timeout must record timed_out: True, not a routing mismatch."""
    test_case = {"id": "C05", "query": "Complex career query", "expected_domains": ["career"], "category": "career_only"}

    with patch("requests.post", side_effect=requests.exceptions.Timeout("Read timed out")):
        res = evaluate_query(test_case, timeout=75)

    assert res["passed"] is False
    assert res["timed_out"] is True
    assert res["actual_status"] is None
    assert "Request timed out after 75s" in res["error"]
    assert "Classification mismatch" not in res["error"]


def test_run_evaluation_propagates_custom_timeout():
    """run_evaluation must pass custom timeout parameter to evaluate_query / requests."""
    test_case = {"id": "C01", "query": "Test", "expected_domains": ["career"], "category": "career_only"}
    mock_resp = MockResponse({"status": "success", "domains_activated": ["career"]})

    with patch("requests.post", return_value=mock_resp) as mock_post:
        res = run_evaluation(queries=[test_case], timeout=90, verbose=False)

    assert res["summary"]["passed"] == 1
    assert mock_post.call_args.kwargs["timeout"] == 90


def test_build_parser_timeout_default_and_override():
    """CLI parser must default to 60s and accept --timeout override."""
    parser = build_parser()

    default_args = parser.parse_args([])
    assert default_args.timeout == 60

    custom_args = parser.parse_args(["--timeout", "120"])
    assert custom_args.timeout == 120


def test_build_parser_dataset_default_and_choices():
    """CLI parser must default to 'baseline' and accept valid choices."""
    parser = build_parser()

    default_args = parser.parse_args([])
    assert default_args.dataset == "baseline"

    for choice in ["baseline", "additional", "combined"]:
        args = parser.parse_args(["--dataset", choice])
        assert args.dataset == choice


def test_dataset_selection_loads_expected_queries():
    """Ensure each CLI dataset option resolves to the exact expected query list."""
    from test_queries import DATASETS, TEST_QUERIES, ADDITIONAL_TEST_QUERIES, COMBINED_TEST_QUERIES

    assert DATASETS["baseline"] is TEST_QUERIES
    assert len(DATASETS["baseline"]) == 50

    assert DATASETS["additional"] is ADDITIONAL_TEST_QUERIES
    assert len(DATASETS["additional"]) == 300

    assert DATASETS["combined"] is COMBINED_TEST_QUERIES
    assert len(DATASETS["combined"]) == 350


def test_provenance_records_dataset_name_and_query_count():
    """get_provenance must record dataset_name and query_count correctly without inventing values."""
    prov_baseline = get_provenance(dataset_name="baseline", query_count=50)
    assert prov_baseline["dataset_name"] == "baseline"
    assert prov_baseline["query_count"] == 50

    prov_additional = get_provenance(dataset_name="additional", query_count=200)
    assert prov_additional["dataset_name"] == "additional"
    assert prov_additional["query_count"] == 200

    prov_combined = get_provenance(dataset_name="combined", query_count=250)
    assert prov_combined["dataset_name"] == "combined"
    assert prov_combined["query_count"] == 250


# ── 11. Console summary formatting tests ──────────────────────────────────

def test_format_console_summary_required_fields_and_units():
    summary = {
        "total": 200,
        "passed": 123,
        "failed": 77,
        "timed_out": 0,
        "excluded": 0,
        "accuracy_pct": 61.5,
        "avg_latency_ms": 967,
        "fastest_latency_ms": 3,
        "slowest_latency_ms": 7045,
    }
    categories = {
        "career": {"total": 20, "passed": 20, "accuracy_pct": 100.0, "avg_latency_ms": 2225},
        "finance": {"total": 20, "passed": 0, "accuracy_pct": 0.0, "avg_latency_ms": 18},
    }

    output = format_console_summary(summary, categories)

    # 1. Overall accuracy as passed/total = percentage
    assert "Overall accuracy: 123/200 = 61.5%" in output

    # 2. Passed, failed, timed-out, and excluded counts
    assert "Passed: 123 | Failed: 77 | Timed Out: 0 | Excluded: 0" in output

    # 3. Per-category table headers
    assert "Per-category breakdown:" in output
    assert "Category" in output
    assert "Passed/Total" in output
    assert "Accuracy" in output
    assert "Avg latency" in output

    # 4. Latency summary with ms units
    assert "Latency summary:" in output
    assert "Fastest: 3ms" in output
    assert "Slowest: 7045ms" in output
    assert "Average: 967ms" in output


def test_format_console_summary_category_uniqueness_and_no_duplicate_headings():
    summary = {
        "total": 50,
        "passed": 48,
        "failed": 2,
        "timed_out": 0,
        "excluded": 0,
        "accuracy_pct": 96.0,
        "avg_latency_ms": 1500,
        "fastest_latency_ms": 15,
        "slowest_latency_ms": 3200,
    }
    categories = {
        "career_only": {"total": 8, "passed": 8, "accuracy_pct": 100.0, "avg_latency_ms": 1200},
        "health_only": {"total": 8, "passed": 8, "accuracy_pct": 100.0, "avg_latency_ms": 900},
        "finance_only": {"total": 8, "passed": 8, "accuracy_pct": 100.0, "avg_latency_ms": 1100},
    }

    output = format_console_summary(summary, categories)

    # Section headings must appear exactly once
    assert output.count("Overall accuracy:") == 1
    assert output.count("Per-category breakdown:") == 1
    assert output.count("Latency summary:") == 1

    # Each category name must appear exactly once in the table lines
    for cat in categories:
        assert output.count(cat) == 1


def test_format_console_summary_percentage_formatting():
    summary = {
        "total": 200,
        "passed": 123,
        "failed": 77,
        "timed_out": 0,
        "excluded": 0,
        "accuracy_pct": 61.5,
        "avg_latency_ms": 967,
        "fastest_latency_ms": 3,
        "slowest_latency_ms": 7045,
    }
    categories = {
        "career": {"total": 20, "passed": 20, "accuracy_pct": 100.0, "avg_latency_ms": 2225},
        "finance": {"total": 20, "passed": 0, "accuracy_pct": 0.0, "avg_latency_ms": 18},
        "edge_case": {"total": 15, "passed": 13, "accuracy_pct": 86.666, "avg_latency_ms": 2322},
    }

    output = format_console_summary(summary, categories)

    # Percentages must be formatted to one decimal place
    assert "61.5%" in output
    assert "100.0%" in output
    assert "0.0%" in output
    assert "86.7%" in output

    # Must NOT have unaligned/detached percentage symbols like "100  %" or "0    %"
    assert "100  %" not in output
    assert "0    %" not in output
    assert "87   %" not in output


def test_format_console_summary_safe_handling_of_empty_and_missing_latencies():
    # Empty overall evaluation (zero queries)
    empty_summary = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "timed_out": 0,
        "excluded": 0,
        "accuracy_pct": 0.0,
        "avg_latency_ms": 0,
        "fastest_latency_ms": None,
        "slowest_latency_ms": None,
    }
    # Zero-query category and missing latency category
    categories = {
        "empty_category": {"total": 0, "passed": 0, "accuracy_pct": 0.0, "avg_latency_ms": None},
        "timed_out_category": {"total": 3, "passed": 0, "accuracy_pct": 0.0, "avg_latency_ms": None},
    }

    # Must not raise ZeroDivisionError or crash
    output = format_console_summary(empty_summary, categories)

    assert "Overall accuracy: 0/0 = 0.0%" in output
    assert "Passed: 0 | Failed: 0 | Timed Out: 0 | Excluded: 0" in output
    assert "0/0" in output
    assert "0/3" in output
    assert "N/A" in output
    assert "Fastest: N/A" in output
    assert "Slowest: N/A" in output
    assert "Average: N/A" in output


def test_format_console_summary_column_alignment_and_width():
    summary = {
        "total": 250,
        "passed": 200,
        "failed": 50,
        "timed_out": 0,
        "excluded": 0,
        "accuracy_pct": 80.0,
        "avg_latency_ms": 1200,
        "fastest_latency_ms": 5,
        "slowest_latency_ms": 5000,
    }
    categories = {
        "clarification_needed": {"total": 3, "passed": 0, "accuracy_pct": 0.0, "avg_latency_ms": 2051},
        "career": {"total": 20, "passed": 20, "accuracy_pct": 100.0, "avg_latency_ms": 2225},
        "edge_case": {"total": 15, "passed": 13, "accuracy_pct": 86.7, "avg_latency_ms": 2322},
    }

    output = format_console_summary(summary, categories)
    lines = output.splitlines()

    # Find the table header line
    header_idx = -1
    for idx, line in enumerate(lines):
        if "Category" in line and "Passed/Total" in line:
            header_idx = idx
            break

    assert header_idx != -1
    header_line = lines[header_idx]
    pass_col_offset = header_line.index("Passed/Total")
    acc_col_offset = header_line.index("Accuracy")
    lat_col_offset = header_line.index("Avg latency")

    # Verify column start offsets align across all category rows
    for cat in categories:
        cat_line = next(line for line in lines if line.strip().startswith(cat))
        # Ensure max line width is <= 80 characters for standard PowerShell terminal
        assert len(cat_line) <= 80
        # Passed/Total starts at pass_col_offset
        assert cat_line[pass_col_offset] in "0123456789"
        # Accuracy ends with '%' within acc column bounds
        assert "%" in cat_line[acc_col_offset:lat_col_offset]


def test_format_console_summary_polymorphic_input():
    eval_result = {
        "summary": {
            "total": 10,
            "passed": 10,
            "failed": 0,
            "timed_out": 0,
            "excluded": 0,
            "accuracy_pct": 100.0,
            "avg_latency_ms": 50,
            "fastest_latency_ms": 20,
            "slowest_latency_ms": 80,
        },
        "per_category": {
            "career": {"total": 10, "passed": 10, "accuracy_pct": 100.0, "avg_latency_ms": 50},
        },
    }

    out1 = format_console_summary(eval_result)
    out2 = format_console_summary(eval_result["summary"], eval_result["per_category"])
    assert out1 == out2


def test_run_evaluation_verbose_prints_formatted_summary(capsys):
    test_case = {
        "id": "C01",
        "query": "How do I become a data scientist?",
        "expected_domains": ["career"],
        "category": "career",
    }
    mock_resp = MockResponse({
        "status": "success",
        "domains_activated": ["career"],
        "responses": [{"domain": "career", "message": "Career advice"}],
    })

    with patch("requests.post", return_value=mock_resp):
        res = run_evaluation(queries=[test_case], verbose=True)

    captured = capsys.readouterr().out
    assert "Overall accuracy: 1/1 = 100.0%" in captured
    assert "Passed: 1 | Failed: 0 | Timed Out: 0 | Excluded: 0" in captured
    assert "Per-category breakdown:" in captured
    assert "career" in captured
    assert "1/1" in captured
    assert "100.0%" in captured
    assert "Latency summary:" in captured
    assert "Fastest:" in captured
    assert "Slowest:" in captured
    assert "Average:" in captured
