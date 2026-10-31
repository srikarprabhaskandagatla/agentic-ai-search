import os
from dotenv import load_dotenv

load_dotenv("backend/.env")

LLM_PROVIDER = "groq"
LLM_BASE_URL = "https://api.groq.com/openai/v1"
LLM_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
LLM_MAX_RETRIES = 5
LLM_REASONING = "low"

PLANNER_MODEL = "openai/gpt-oss-20b"
EXTRACTOR_MODEL = "openai/gpt-oss-20b"
GAP_MODEL = "openai/gpt-oss-20b"
FILLER_MODEL = "openai/gpt-oss-20b"
RANKER_MODEL = "openai/gpt-oss-20b"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
