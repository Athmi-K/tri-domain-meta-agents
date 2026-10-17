from utils.intent_detector import resolve_domain
from unittest.mock import patch

from langchain_agents.meta_lc_agent import run_chat_agents


def test_auto_routes_career_query():
    routing = resolve_domain("auto", "How do I prepare for a software engineering interview?")

    assert routing["selected_domain"] == "auto"
    assert routing["detected_domain"] == "career"
    assert routing["routed_domain"] == "career"


def test_auto_routes_health_query():
    routing = resolve_domain("auto", "How can I improve my mental health?")

    assert routing["detected_domain"] == "health"
    assert routing["routed_domain"] == "health"


def test_auto_routes_finance_query():
    routing = resolve_domain("auto", "How should I build an emergency fund?")

    assert routing["detected_domain"] == "finance"
    assert routing["routed_domain"] == "finance"


def test_manual_domain_is_preserved_for_ambiguous_query():
    routing = resolve_domain("career", "What should I focus on next?")

    assert routing["detected_domain"] is None
    assert routing["route_status"] == "ambiguous"
    assert routing["routed_domain"] is None


def test_clear_health_intent_overrides_career_selector():
    routing = resolve_domain("career", "I feel anxious and overwhelmed. How can I improve my mental health?")

    assert routing["detected_domain"] == "health"
    assert routing["routed_domain"] == "health"


def test_clear_finance_intent_overrides_health_selector():
    routing = resolve_domain("health", "How should I manage my debt and monthly budget?")

    assert routing["detected_domain"] == "finance"
    assert routing["routed_domain"] == "finance"


def test_unsupported_query_is_not_forced_into_career():
    routing = resolve_domain("career", "Write a poem about the ocean.")

    assert routing["route_status"] == "unsupported"
    assert routing["routed_domain"] is None


def test_unsupported_query_is_not_forced_into_health():
    routing = resolve_domain("health", "Who won yesterday's cricket match?")

    assert routing["route_status"] == "unsupported"
    assert routing["routed_domain"] is None


def test_unsupported_query_is_not_forced_into_finance():
    routing = resolve_domain("finance", "Write a poem about the ocean.")

    assert routing["route_status"] == "unsupported"
    assert routing["routed_domain"] is None


def test_unsupported_query_is_not_routed_in_auto_mode():
    routing = resolve_domain("auto", "Who won yesterday's cricket match?")

    assert routing["route_status"] == "unsupported"
    assert routing["routed_domain"] is None


def test_mixed_domain_query_requires_clarification():
    routing = resolve_domain(
        "auto",
        "How should I manage work stress while planning my monthly budget?",
    )

    assert routing["route_status"] == "mixed"
    assert routing["routed_domain"] is None
    assert routing["routed_domains"] == ["health", "finance"]


def test_mixed_domain_query_routes_only_detected_agents():
    routing = resolve_domain(
        "career",
        "How can I balance career growth and my monthly budget?",
    )

    assert routing["route_status"] == "mixed"
    assert routing["routed_domains"] == ["career", "finance"]


def test_follow_up_uses_single_recent_context_domain():
    routing = resolve_domain(
        "auto",
        "What about certifications?",
        conversation_context="I want to prepare for software engineering interviews.",
    )

    assert routing["route_status"] == "routed"
    assert routing["context_domain"] == "career"
    assert routing["routed_domains"] == ["career"]


def test_multi_domain_execution_uses_only_official_langchain_runner():
    results = {
        "career": {"recommendation": "Career result", "confidence": 0.8},
        "finance": {"recommendation": "Finance result", "confidence": 0.9},
    }

    with patch(
        "langchain_agents.meta_lc_agent.run_chat_agent",
        side_effect=lambda db, user_id, domain, query: results[domain],
    ) as runner:
        combined = run_chat_agents(
            db=object(),
            user_id="user-1",
            domains=["career", "finance"],
            query="How can I balance career growth and my budget?",
        )

    assert runner.call_count == 2
    assert [call.kwargs["domain"] for call in runner.call_args_list] == [
        "career",
        "finance",
    ]
    assert "Career perspective" in combined["message"]
    assert "Finance perspective" in combined["message"]
