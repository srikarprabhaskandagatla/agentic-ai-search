import asyncio, logging, math, os, time
from collections import deque

from dotenv import load_dotenv

load_dotenv("backend/.env")

LLM_PROVIDER = "groq"
LLM_BASE_URL = "https://api.groq.com/openai/v1"
LLM_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
LLM_MAX_RETRIES = 5
LLM_REASONING = "low"

PLANNER_MODEL = "openai/gpt-oss-20b"
EXTRACTOR_MODEL = "openai/gpt-oss-20b"
FILLER_MODEL = "openai/gpt-oss-20b"
RANKER_MODEL = "openai/gpt-oss-20b"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

TOKENS_PER_MINUTE = 7900
EXTRACT_INPUT_TOKENS = 3000
FILL_INPUT_TOKENS = 4000
MIN_FILL_TOKENS = 500

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 60.0
CHARS_PER_TOKEN = 3.5


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def fit_text(text: str, max_tokens: int) -> str:
    max_chars = int(max(max_tokens, 0) * CHARS_PER_TOKEN)
    return text if len(text) <= max_chars else text[:max_chars]


class TokenPacer:
    def __init__(self, limit: int, window: float = WINDOW_SECONDS, clock=time.monotonic, sleep=asyncio.sleep):
        self.limit = limit
        self.window = window
        self.clock = clock
        self.sleep = sleep
        self.events: deque[list] = deque()
        self.lock = asyncio.Lock()

    def _used(self) -> int:
        now = self.clock()
        while self.events and self.events[0][0] + self.window <= now:
            self.events.popleft()
        return sum(event[1] for event in self.events)

    def available(self) -> int:
        return max(self.limit - self._used(), 0)

    async def reserve(self, tokens: int) -> list:
        tokens = min(max(tokens, 0), self.limit)
        async with self.lock:
            while self._used() + tokens > self.limit:
                wait = self.events[0][0] + self.window - self.clock()
                logger.info("Pacer: waiting %.1fs for %d tokens", max(wait, 0), tokens)
                await self.sleep(max(wait, 0.05))
            event = [self.clock(), tokens]
            self.events.append(event)
            return event

    def settle(self, event: list, usage, stage: str) -> None:
        if not isinstance(getattr(usage, "total_tokens", None), int):
            return
        event[1] = usage.total_tokens
        logger.info("Tokens %s: %d in, %d out, %d left this minute",
                    stage, usage.prompt_tokens, usage.completion_tokens, self.available())


pacer = TokenPacer(TOKENS_PER_MINUTE)
