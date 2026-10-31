import os
from dotenv import load_dotenv

load_dotenv("backend/.env")

LLM_PROVIDER = "mistral"
LLM_BASE_URL = "https://api.mistral.ai/v1"
LLM_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip()
LLM_MAX_RETRIES = 5

PLANNER_MODEL = "mistral-small-latest"
EXTRACTOR_MODEL = "mistral-small-latest"
RESOLVER_MODEL = "mistral-small-latest"
GAP_MODEL = "mistral-small-latest"
FILLER_MODEL = "mistral-small-latest"
RANKER_MODEL = "mistral-small-latest"
