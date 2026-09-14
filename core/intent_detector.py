from core.llm_client import call_llm
from core.safety_layer import normalize_query, DOMAIN_KEYWORDS, has_keyword


def detect_intent(query: str) -> dict:
    """
    Detects which domain(s) a query belongs to.
    Uses DOMAIN_KEYWORDS from safety_layer — single source of truth.
    No duplicate keyword lists.
    """
    query_lower = normalize_query(query)

    if not query_lower:
        return {
            "domains":    ["general"],
            "confidence": 0.0,
            "reasoning":  "Empty query"
        }

    # ── Use shared DOMAIN_KEYWORDS — no duplicate list ────────
    matched = []
    match_scores = {}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if has_keyword(query_lower, kw))
        if score > 0:
            matched.append(domain)
            match_scores[domain] = score

    # ── No match fallback ─────────────────────────────────────
    if not matched:
        return {
            "domains":    ["general"],
            "confidence": 0.5,
            "reasoning":  "No clear domain detected"
        }

    # ── Sort by match strength ────────────────────────────────
    matched = sorted(matched, key=lambda d: match_scores[d], reverse=True)

    # ── Confidence based on number of domains ─────────────────
    if len(matched) == 1:
        confidence = 0.85
    elif len(matched) == 2:
        confidence = 0.90
    else:
        confidence = 0.95

    reasoning = f"Detected domains: {', '.join(matched)} based on keyword matches"

    return {
        "domains":    matched,
        "confidence": confidence,
        "reasoning":  reasoning
    }