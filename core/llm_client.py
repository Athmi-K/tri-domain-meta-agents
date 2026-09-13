import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

# Force reload .env every time
load_dotenv(override=True)

# Groq replacement for the retired llama-3.3-70b-versatile
MODEL = "openai/gpt-oss-120b"


def get_client():
    """Creates a fresh Groq client every time."""
    load_dotenv(override=True)

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file")

    return Groq(api_key=api_key)


def extract_json(text: str):
    if not text:
        return None

    text = text.strip()

    # Remove Markdown code fences if the model adds them.
    text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    # Try to parse the complete response.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract the JSON object from surrounding text.
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


def call_llm(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.4,
    max_tokens: int = 1200,
) -> dict:
    """
    Shared LLM caller used by all agents.

    Preserves the caller-provided prompt schema so specific tasks
    such as skill extraction can request custom JSON fields.
    """

    try:
        client = get_client()

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or ""

        print("DEBUG - RAW LLM RESPONSE:")
        print(repr(raw))

        parsed = extract_json(raw)

        if isinstance(parsed, dict):
            return parsed

        return {
            "summary": raw.strip(),
            "error": "LLM returned unstructured text",
            "confidence": 0.5,
        }

    except Exception as e:
        import traceback

        traceback.print_exc()

        return {
            "summary": "",
            "error": str(e),
            "confidence": 0.0,
        }