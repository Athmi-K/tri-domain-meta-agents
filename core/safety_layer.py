import re

BLOCKED_KEYWORDS = [
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
    # Hate
    "hate", "racist", "discriminate"
]

SENSITIVE_TOPICS = [
    "depression", "anxiety", "mental health",
    "burnout", "overwhelmed", "hopeless"
]

# ── Single source of truth for domain keywords ────────────────
# Used by BOTH check_relevance() and detect_intent()
DOMAIN_KEYWORDS = {
    "career": [
        "job", "work", "career", "skill", "resume", "cv",
        "interview", "promotion", "switch", "role", "hire",
        "scientist", "developer", "engineer", "analyst",
        "learn", "course", "salary", "linkedin", "data science",
        "placement", "internship", "fresher", "experience",
        "want to become", "want to be", "become a", "become an",
        "looking for a job", "switch careers", "change career",
        "career advice", "career growth", "job advice"
    ],
    "health": [
        "health", "weight", "bmi", "fitness", "exercise",
        "diet", "sick", "doctor", "sleep", "tired", "gym",
        "calories", "overweight", "fat", "muscle", "mental", "stress",
        "eat", "food", "nutrition", "lose weight", "gain weight",
        "workout", "anxiety", "mood", "wellness"
    ],
    "finance": [
        "money", "saving", "invest", "debt", "loan", "budget",
        "expense", "income", "tax", "emi", "rent", "insurance",
        "sip", "ppf", "stock", "mutual fund", "earning",
        "spend", "finance", "bank", "credit", "debit",
        "financial", "stability", "wealth", "financial stability",
        "retirement", "retire", "pension", "corpus",
        "save for", "how much should i save", "savings goal",
        "financial goal", "emergency fund", "net worth"
    ]
}

CAREER_INTENT_PHRASES = [
    "want to become",
    "want to be",
    "become a",
    "become an",
    "looking for a job",
    "switch careers",
    "change career",
    "career advice",
    "career growth",
    "job advice",
]

CAREER_ROLE_KEYWORDS = [
    "analyst",
    "data analyst",
    "data scientist",
    "scientist",
    "developer",
    "engineer",
    "manager",
    "product manager",
    "designer",
    "tester",
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


def check_relevance(query: str) -> dict:
    """
    Checks if query is related to Career, Health, or Finance.
    Blocks completely off-topic queries.
    Uses DOMAIN_KEYWORDS — single source of truth.
    """
    query_lower = normalize_query(query)
    matched = []

    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(has_keyword(query_lower, kw) for kw in keywords):
            matched.append(domain)

    # Extra career intent check for indirect phrasing
    has_career_intent = any(
        phrase in query_lower for phrase in CAREER_INTENT_PHRASES
    )
    has_career_role = any(
        role in query_lower for role in CAREER_ROLE_KEYWORDS
    )
    if not matched and has_career_intent and has_career_role:
        matched.append("career")

    if not matched:
        return {
            "is_relevant":     False,
            "matched_domains": [],
            "message":         "Your query does not seem related to Career, Health, or Finance. Please ask something in one of these areas."
        }

    return {
        "is_relevant":     True,
        "matched_domains": matched,
        "message":         None
    }