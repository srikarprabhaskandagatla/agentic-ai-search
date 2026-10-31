from __future__ import annotations
import asyncio, logging, os
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from tavily import AsyncTavilyClient
from ..models import ScrapedPage
from ..utils import clean_markdown

logger = logging.getLogger(__name__)

MAX_RESULTS = 4
MAX_CONTENT_CHARS = 8_000
MIN_CONTENT_CHARS = 80
PRIMARY_QUERIES = 2
MIN_PAGES = 4
EXCLUDED_DOMAINS = [
    "youtube.com", "pinterest.com", "instagram.com", 
    "tiktok.com", "facebook.com"
]
TRACKING_PARAMS = {"fbclid", "gclid", "msclkid", "mc_cid", "mc_eid"}


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    host = parts.netloc.lower().removeprefix("www.")
    path = parts.path.rstrip("/") or "/"
    params = [
        (key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in TRACKING_PARAMS
    ]
    return urlunsplit(("https", host, path, urlencode(params), ""))


def _client() -> AsyncTavilyClient:
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set")
    return AsyncTavilyClient(api_key=api_key)


def _to_page(item: dict) -> ScrapedPage | None:
    url = (item.get("url") or "").strip()
    text = clean_markdown(item.get("raw_content") or item.get("content") or "")
    if not url or len(text) < MIN_CONTENT_CHARS:
        return None
    return ScrapedPage(url=url, title=item.get("title") or "", content=text[:MAX_CONTENT_CHARS])


async def _search(client: AsyncTavilyClient, query: str) -> list[ScrapedPage]:
    response = await client.search(
        query=query,
        max_results=MAX_RESULTS,
        search_depth="basic",
        include_raw_content=True,
        exclude_domains=EXCLUDED_DOMAINS,
    )
    pages = [page for page in map(_to_page, response.get("results", [])) if page]
    logger.info("Tavily returned %d usable pages for: %s", len(pages), query)
    return pages


async def retrieve(queries: list[str], seen_urls: set[str]) -> list[ScrapedPage]:
    client = _client()
    results = await asyncio.gather(*(_search(client, q) for q in queries), return_exceptions=True)

    errors = [r for r in results if isinstance(r, Exception)]
    for error in errors:
        logger.warning("Search failed: %s", error)
    if errors and len(errors) == len(results):
        raise errors[0]

    pages: list[ScrapedPage] = []
    taken = {normalize_url(url) for url in seen_urls}
    for result in results:
        if isinstance(result, Exception):
            continue
        for page in result:
            key = normalize_url(page.url)
            if key not in taken:
                taken.add(key)
                pages.append(page)
    return pages


async def retrieve_with_backup(queries: list[str], seen_urls: set[str]) -> tuple[list[ScrapedPage], list[str]]:
    primary, backup = queries[:PRIMARY_QUERIES], queries[PRIMARY_QUERIES:]
    pages = await retrieve(primary, seen_urls)
    if len(pages) >= MIN_PAGES or not backup:
        return pages, primary
    logger.info("Only %d pages from primary queries, running %d backup queries", len(pages), len(backup))
    more = await retrieve(backup, seen_urls | {page.url for page in pages})
    return pages + more, queries
