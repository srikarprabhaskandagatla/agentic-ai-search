# Stage 3 - Scraper

# Takes a list of URLs, fetches them concurrently (bounded by a semaphore),
# strips boilerplate HTML, and returns clean text content.

# Features:
#   - In-memory URL cache - never re-fetches within a session
#   - Smart content extraction: prefers <article>/<main> over full <body>
#   - Hard-caps content at 8 000 chars to keep extraction prompts cheap
#   - Gracefully skips binary files, non-HTML content, and errors


from __future__ import annotations
import asyncio
import logging
import re
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from ..models import ScrapedPage

logger = logging.getLogger(__name__)


# Configuration
MAX_CONCURRENT = 6          # simultaneous HTTP connections
REQUEST_TIMEOUT = 12.0      # seconds per request
MAX_CONTENT_CHARS = 8_000   # truncation limit (keeps LLM costs down)

SKIP_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tar", ".gz", ".mp4", ".mp3", ".wav", ".png", ".jpg",
    ".jpeg", ".gif", ".svg", ".webp",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Module-level cache - shared across all requests in a process lifetime
_url_cache: dict[str, ScrapedPage] = {}


# HTML cleaning
_JUNK_TAGS = [
    "script", "style", "nav", "footer", "header", "aside",
    "iframe", "noscript", "form", "button", "svg", "figure",
    "advertisement", "ads", "cookie-banner",
]


def _clean_html(html: str, url: str) -> tuple[str, str]:
    """Return (title, clean_text) from raw HTML."""
    soup = BeautifulSoup(html, "lxml")

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Remove boilerplate elements
    for tag in soup.find_all(_JUNK_TAGS):
        tag.decompose()

    # Prefer semantic content containers
    main = (
        soup.find("article")
        or soup.find("main")
        or soup.find(id=re.compile(r"content|main|article", re.I))
        or soup.find(class_=re.compile(r"content|main|article|post|entry", re.I))
        or soup.body
    )

    if main:
        raw_text = main.get_text(separator="\n", strip=True)
    else:
        raw_text = soup.get_text(separator="\n", strip=True)

    # Collapse whitespace
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    # Drop very short noise lines (navigation crumbs etc.)
    lines = [l for l in lines if len(l) > 20 or l.endswith(":")]
    text = "\n".join(lines)

    return title, text[:MAX_CONTENT_CHARS]



# Single-URL fetcher
def _should_skip(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in SKIP_EXTENSIONS)


async def _fetch_one(client: httpx.AsyncClient, url: str) -> ScrapedPage:
    """Fetch and clean a single URL.  Returns ScrapedPage (possibly with error)."""

    if url in _url_cache:
        logger.debug("Cache hit: %s", url)
        return _url_cache[url]

    if _should_skip(url):
        return ScrapedPage(url=url, content="", error="skipped: binary file")

    try:
        resp = await client.get(url, timeout=REQUEST_TIMEOUT, follow_redirects=True)

        if resp.status_code != 200:
            return ScrapedPage(url=url, content="", error=f"HTTP {resp.status_code}")

        ctype = resp.headers.get("content-type", "")
        if "text/html" not in ctype and "text/plain" not in ctype:
            return ScrapedPage(url=url, content="", error=f"non-HTML: {ctype[:40]}")

        title, text = _clean_html(resp.text, url)
        page = ScrapedPage(url=url, title=title, content=text)
        _url_cache[url] = page
        logger.debug("Scraped %d chars from %s", len(text), url)
        return page

    except httpx.TimeoutException:
        return ScrapedPage(url=url, content="", error="timeout")
    except Exception as exc:
        return ScrapedPage(url=url, content="", error=str(exc)[:120])



# Public API
async def scrape_urls(urls: list[str]) -> list[ScrapedPage]:
    """Scrape a list of URLs concurrently.  Returns only successful pages."""
    if not urls:
        return []

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def bounded_fetch(client: httpx.AsyncClient, url: str) -> ScrapedPage:
        async with semaphore:
            return await _fetch_one(client, url)

    async with httpx.AsyncClient(headers=HEADERS) as client:
        tasks = [bounded_fetch(client, url) for url in urls]
        pages = await asyncio.gather(*tasks)

    successful = [p for p in pages if p.content and not p.error]
    logger.info(
        "Scraped %d/%d pages successfully", len(successful), len(urls)
    )
    return successful


def clear_cache() -> None:
    """Clear the in-memory scrape cache (useful between test runs)."""
    _url_cache.clear()
