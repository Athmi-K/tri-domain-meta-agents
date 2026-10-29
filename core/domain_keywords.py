"""Shared vocabulary used for domain classification and relevance checks."""

DOMAIN_KEYWORDS = {
    "career": [
        "job", "jobs", "work", "career", "careers", "skill", "skills",
        "resume", "cv", "interview", "interviews", "promotion", "switch",
        "role", "roles", "hire", "hiring", "scientist", "developer",
        "engineer", "engineering", "analyst", "coding", "programming",
        "software development", "software engineering", "software engineer",
        "learn", "course", "courses", "salary", "salaries", "linkedin",
        "data science", "placement", "placements", "internship",
        "internships", "fresher", "freshers", "experience", "project",
        "projects", "portfolio", "technical skills", "technical interview",
        "leetcode", "coding practice", "certification", "certifications",
        "upskill", "reskill", "job search", "job offer", "offer letter",
        "want to become", "want to be", "become a", "become an",
        "looking for a job", "switch careers", "change career",
        "career advice", "career growth", "job advice",
    ],
    "health": [
        "health", "weight", "bmi", "fitness", "exercise", "diet", "sick",
        "doctor", "sleep", "tired", "fatigue", "gym", "calories",
        "overweight", "fat", "muscle", "mental", "stress", "eat", "food",
        "nutrition", "lose weight", "gain weight", "workout", "workouts",
        "anxiety", "mood", "wellness", "wellbeing", "symptom", "symptoms",
        "pain", "medication", "blood pressure", "heart rate",
        "running", "walking", "strength training", "cardio", "hydration",
        "sleep quality", "insomnia", "burnout", "sleeping", "healthy",
    ],
    "finance": [
        "money", "saving", "savings", "invest", "investment", "investments",
        "debt", "loan", "loans", "budget", "budgeting", "expense", "expenses",
        "college expenses", "tuition", "income", "tax", "taxes", "emi",
        "rent", "insurance", "sip", "ppf", "stock", "stocks", "mutual fund",
        "mutual funds", "earning", "earnings", "spend", "spending", "finance",
        "financial", "finances", "bank", "banking", "credit", "debit", "stability",
        "wealth", "financial stability", "retirement", "retire", "pension",
        "corpus", "cash flow", "net worth", "emergency fund", "interest",
        "return", "returns", "portfolio", "risk tolerance", "risk appetite",
        "financial goal", "financial goals", "savings goal", "save for",
        "how much should i save", "car loan", "home loan", "credit card",
    ],
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
