import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv(override=True)

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    max_tokens=300,
    api_key=os.getenv("GROQ_API_KEY"),
)

response = llm.invoke(
    "What are the top 3 skills for data science?"
)

print(response.content)