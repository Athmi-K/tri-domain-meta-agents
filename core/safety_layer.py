import re
from core.domain_keywords import (
    CAREER_INTENT_PHRASES,
    CAREER_ROLE_KEYWORDS,
    DOMAIN_KEYWORDS,
)

BLOCKED_KEYWORDS = [
    # Prompt injection and secret-extraction attempts
    "ignore all previous instructions", "ignore your instructions",
    "bypass domain routing", "reveal your system prompt",
    "reveal your prompt", "api keys", "system prompt",
    # Violence
    "hurt", "kill", "harm", "attack", "weapon", "bomb", "poison",
    # Illegal financial
    "money laundering", "tax evasion", "insider trading",
    "fake invoice", "bribe", "black money",
    # Fraud
    "fake resume", "fake certificate", "fake degree",
    "fake my resume", "fake the resume", "falsify resume",
    "cheat", "plagiarize", "forge", "fabricate",
    "lie on resume", "lie in resume", "lie about experience",
    # Self harm
    "suicide", "self harm", "end my life", "kill myself",
    "how to die", "ways to die", "want to die",
    # Hate
    "hate", "racist", "discriminate"
]

SELF_HARM_KEYWORDS = [
    "suicide", "self harm", "end my life", "kill myself",
    "how to die", "ways to die", "want to die",
]

SENSITIVE_TOPICS = [
    "depression", "anxiety", "mental health",
    "burnout", "overwhelmed", "hopeless"
]

QUERY_NORMALIZATIONS = {
    "anlyst": "analyst",
    "anlysit": "analyst",
    "analit": "analyst",
    "datascientist": "data scientist",
    "dataanalyst": "data analyst",
    "uiux": "ui ux",
}


def normalize_query(query: str) -> str:
    normalized = query.lower().strip()
    for source, target in QUERY_NORMALIZATIONS.items():
        normalized = normalized.replace(source, target)
    return normalized

def has_keyword(query: str, keyword: str) -> bool:
    """
    Match a keyword safely:
    - multi-word phrase: substring match
    - single word: whole-word regex match
      (prevents false positives like 'kill' in 'skills', or 'eat' in 'create')
    """
    if " " in keyword:
        return keyword in query
    pattern = rf"\b{re.escape(keyword)}\b"
    return re.search(pattern, query) is not None


def has_blocked_keyword(query: str, keyword: str) -> bool:
    return has_keyword(query, keyword)


def is_uncertainty_request(query: str) -> bool:
    """Detect requests that ask for guarantees or exact future predictions."""
    normalized = normalize_query(query)
    return (
        "guaranteed" in normalized
        or "guarantee" in normalized
        or "exact salary" in normalized
        or "exact investment return" in normalized
        or "exact return" in normalized
    )


def check_safety(query: str) -> dict:
    """
    Checks query against blocked keywords and sensitive topics.
    Returns:
        is_safe      → True if query can proceed
        is_sensitive → True if needs careful handling
        reason       → why it was blocked
        message      → message to show user
    """
    query_lower = normalize_query(query)

    # Self-harm requests need a supportive response rather than a generic
    # domain-routing message.
    for keyword in SELF_HARM_KEYWORDS:
        if has_blocked_keyword(query_lower, keyword):
            return {
                "is_safe": False,
                "is_sensitive": True,
                "reason": f"Self-harm content detected: '{keyword}'",
                "message": (
                    "I’m sorry you’re feeling this way. I can’t help with ways "
                    "to die, but I can stay with you and help you get through "
                    "the next few minutes. If you might act on these thoughts "
                    "or are in immediate danger, call your local emergency "
                    "number now or go to the nearest emergency department. "
                    "Please also contact someone you trust and ask them to "
                    "stay with you. If you can, tell me: are you in immediate "
                    "danger right now?"
                ),
            }

    # Hard block — never proceed
    for keyword in BLOCKED_KEYWORDS:
        if has_blocked_keyword(query_lower, keyword):
            return {
                "is_safe":      False,
                "is_sensitive": False,
                "reason":       f"Query contains restricted content: '{keyword}'",
                "message":      "I can only help with Career, Health, and Finance queries. Please rephrase your question."
            }

    # Sensitive — proceed but flag it
    for topic in SENSITIVE_TOPICS:
        if topic in query_lower:
            return {
                "is_safe":      True,
                "is_sensitive": True,
                "reason":       f"Sensitive topic detected: '{topic}'",
                "message":      "I noticed your query touches on a sensitive topic. I will do my best to help, but please consult a professional for personal support."
            }

    # All clear
    return {
        "is_safe":      True,
        "is_sensitive": False,
        "reason":       None,
        "message":      None
    }


NEGATION_PATTERNS = [
    re.compile(r"\b(?:don't|do not|doesn't|does not|didn't|did not)\s+(?:need|want|require)\b"),
    re.compile(r"\b(?:not\s+looking\s+for|not\s+searching\s+for|not\s+seeking|not\s+interested\s+in|not\s+wanting)\b"),
    re.compile(r"\b(?:no\s+need\s+for|no\s+interest\s+in)\b"),
    re.compile(r"\b(?:skip|exclude|avoid)\b"),
    re.compile(r"\bno\s+(?:career|health|finance|financial|budgeting|medical|doctor)\b"),
    re.compile(r"(?:^|[,;]\s*)\bnot\s+(?:for\s+|about\s+|in\s+)?(?:career|health|finance|financial|budgeting|changing\s+jobs?|switching\s+careers?|job\s+hunting|medical|doctor)\b"),
    re.compile(r"\bwithout\s+(?:discussing|mentioning|including|touching\s+on)?\s*(?:career|health|finance|financial|budgeting|medical|doctor|advice|tips)\b"),
]

CONTRAST_CONNECTORS = re.compile(r"\b(?:instead\s+of|rather\s+than)\b")
WITHOUT_CONNECTOR = re.compile(r"\bwithout\b")


def is_clause_negated(clause: str) -> bool:
    """Returns True if the clause contains an explicit exclusion/negation pattern."""
    for pattern in NEGATION_PATTERNS:
        if pattern.search(clause):
            return True
    return False


def _extract_clause_domains(clause: str) -> tuple[set[str], dict[str, int]]:
    """Helper to detect domain keyword matches within a sub-clause."""
    matched_domains = set()
    scores = {d: 0 for d in DOMAIN_KEYWORDS}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        matches = sum(1 for kw in keywords if has_keyword(clause, kw))
        if matches > 0:
            matched_domains.add(domain)
            scores[domain] += matches

    # Extra career intent check for indirect phrasing
    has_career_intent = any(phrase in clause for phrase in CAREER_INTENT_PHRASES)
    has_career_role = any(role in clause for role in CAREER_ROLE_KEYWORDS)
    if has_career_intent and has_career_role:
        matched_domains.add("career")
        scores["career"] += 1

    return matched_domains, scores


def extract_domain_signals(query: str) -> tuple[list[str], dict[str, int], set[str]]:
    """
    Splits query into clauses and classifies domain mentions into:
    - active_domains: domains explicitly requested with positive intent
    - positive_scores: keyword match counts from positive clauses
    - negated_domains: domains explicitly negated/excluded
    """
    query_lower = normalize_query(query)
    # Split by major punctuation and contrastive conjunctions
    clauses = re.split(r'[;!?.]|\s*,\s*|\s+\b(?:but|however|although|whereas|while)\b\s+', query_lower)

    positive_domains = set()
    positive_scores = {d: 0 for d in DOMAIN_KEYWORDS}
    negated_domains = set()

    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue

        # Check for contrastive connectors: "instead of", "rather than"
        contrast_match = CONTRAST_CONNECTORS.search(clause)
        if contrast_match:
            pos_part = clause[:contrast_match.start()].strip()
            neg_part = clause[contrast_match.end():].strip()

            if pos_part:
                pos_doms, pos_sc = _extract_clause_domains(pos_part)
                if not is_clause_negated(pos_part):
                    positive_domains.update(pos_doms)
                    for d, sc in pos_sc.items():
                        positive_scores[d] += sc
                else:
                    negated_domains.update(pos_doms)

            if neg_part:
                neg_doms, _ = _extract_clause_domains(neg_part)
                negated_domains.update(neg_doms)
            continue

        # Check for "without" connector
        without_match = WITHOUT_CONNECTOR.search(clause)
        if without_match:
            pos_part = clause[:without_match.start()].strip()
            neg_part = clause[without_match.end():].strip()
            pos_doms, pos_sc = _extract_clause_domains(pos_part)

            # If there is a positive domain request before "without":
            # e.g. "Help me find a job without spending money" -> career positive, finance excluded
            if pos_doms and not is_clause_negated(pos_part):
                positive_domains.update(pos_doms)
                for d, sc in pos_sc.items():
                    positive_scores[d] += sc

                if neg_part:
                    neg_doms, _ = _extract_clause_domains(neg_part)
                    negated_domains.update(neg_doms)
                continue

            # If "without" is at the start (or pos_part has no domains), check if it's an explicit exclusion:
            # e.g. "without medical tips", "without finance advice", "without discussing health"
            if is_clause_negated(clause):
                doms, _ = _extract_clause_domains(clause)
                negated_domains.update(doms)
                continue

            # Otherwise (e.g. "I am without a job" or "without money"), treat hardship as positive domain
            doms, scs = _extract_clause_domains(clause)
            positive_domains.update(doms)
            for d, sc in scs.items():
                positive_scores[d] += sc
            continue

        # Standard clause processing
        clause_is_negated = is_clause_negated(clause)
        doms, scs = _extract_clause_domains(clause)
        if doms:
            if clause_is_negated:
                negated_domains.update(doms)
            else:
                positive_domains.update(doms)
                for d, sc in scs.items():
                    positive_scores[d] += sc

    # Preserve consistent domain order: career, health, finance
    domain_order = ["career", "health", "finance"]
    active_domains = [d for d in domain_order if d in positive_domains]

    return active_domains, positive_scores, negated_domains


def check_relevance(query: str) -> dict:
    """
    Checks if query is related to Career, Health, or Finance.
    Blocks completely off-topic queries and queries that only contain negated domain terms.
    Uses DOMAIN_KEYWORDS — single source of truth.
    """
    active_domains, _, _ = extract_domain_signals(query)

    if not active_domains:
        return {
            "is_relevant":     False,
            "matched_domains": [],
            "message":         "Your query does not seem related to Career, Health, or Finance. Please ask something in one of these areas."
        }

    return {
        "is_relevant":     True,
        "matched_domains": active_domains,
        "message":         None
    }