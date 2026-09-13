
from pathlib import Path

from tools.career_tools import _extract_skills_from_career_knowledge


career_knowledge = Path("rag/knowledge_base/career_kb.txt").read_text(
    encoding="utf-8"
)

skills = _extract_skills_from_career_knowledge(
    knowledge_text=career_knowledge,
    target_role="Data Analyst",
)

print("Extracted skills:", skills)