import json
import traceback

from tools.career_tools import skill_gap_analyzer


def test_skill_gap_analyzer():
    current_skills = [
        "JavaScript",
        "HTML",
        "CSS",
       
        "Vue.js",
        "Angular",
        
        "Git",
    ]

    target_role = "Frontend Developer"

    print("=" * 60)
    print("Testing skill_gap_analyzer()")
    print("=" * 60)

    try:
        result = skill_gap_analyzer(
            current_skills=current_skills,
            target_role=target_role,
            similarity_threshold=0.65,
        )

        assert result is not None, "Function returned None"

        print("Test status: PASSED")
        print(f"Target role: {target_role}")
        print("\nFunction output:")

        print(json.dumps(
            result,
            indent=4,
            ensure_ascii=False,
        ))

    except Exception as error:
        print("Test status: FAILED")
        print(f"Error type: {type(error).__name__}")
        print(f"Error message: {error}")
        traceback.print_exc()

        raise


if __name__ == "__main__":
    test_skill_gap_analyzer()