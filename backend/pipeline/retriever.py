from __future__ import annotations
import asyncio, logging, os

from tavily import AsyncTavilyClient
from ..models import ScrapedPage

logger = logging.getLogger(__name__)

MAX_RESULTS = 4
MAX_CONTENT_CHARS = 8_000
MIN_CONTENT_CHARS = 80


def _client() -> AsyncTavilyClient:
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set")
    return AsyncTavilyClient(api_key=api_key)


def _to_page(item: dict) -> ScrapedPage | None:
    url = (item.get("url") or "").strip()
    text = item.get("raw_content") or item.get("content") or ""
    if not url or len(text) < MIN_CONTENT_CHARS:
        return None
    return ScrapedPage(url=url, title=item.get("title") or "", content=text[:MAX_CONTENT_CHARS])


async def _search(client: AsyncTavilyClient, query: str) -> list[ScrapedPage]:
    response = await client.search(
        query=query,
        max_results=MAX_RESULTS,
        search_depth="basic",
        include_raw_content=True,
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
    taken = set(seen_urls)
    for result in results:
        if isinstance(result, Exception):
            continue
        for page in result:
            if page.url not in taken:
                taken.add(page.url)
                pages.append(page)
    return pages
