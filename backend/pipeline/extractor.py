# Stage 4 - Extractor 

from __future__ import annotations
import asyncio, uuid, logging
from openai import AsyncOpenAI
from ..config import LLM_REASONING, EXTRACTOR_MODEL
from ..models import Entity, CellValue, ScrapedPage, SourceRef
from .utils import extract_json_obj

logger = logging.getLogger(__name__)

PAGES_PER_CALL = 3
PAGE_CHARS = 3_000
MAX_CONCURRENT_CALLS = 2

EXTRACTOR_SYSTEM_PROMPT = """You are a precise information extraction engine.

You get several numbered web pages and a list of columns. Extract ALL entities of the target entity type from every page.

STRICT RULES:
1. Only extract values EXPLICITLY present in the content. Never infer or hallucinate.
2. For every value, copy a short exact phrase (max 15 words) from that page into "snippet".
3. Set confidence: 1.0 = stated directly, 0.8 = light interpretation, 0.6 = implied, 0.4 = uncertain.
4. Leave out any field that is not found on the page.
5. "name" must always be present - if you can't find a clear name, skip the entity.
6. Set "page" to the number of the page the entity came from. If an entity is on two pages, list it once per page.

Respond with ONLY a raw JSON object - no markdown, no preamble:
{
  "entities": [
    {
      "page": 1,
      "name":     {"value": "...", "confidence": 0.99, "snippet": "exact phrase"},
      "column_b": {"value": "...", "confidence": 0.85, "snippet": "exact phrase"}
    }
  ]
}"""

USER_TEMPLATE = """Extract all {entity_type} entities from these pages.
Columns: {columns}

{pages}
"""

PAGE_TEMPLATE = """[PAGE {number}]
URL: {url}
Title: {title}
CONTENT:
{content}
"""


def _batches(pages: list[ScrapedPage]) -> list[list[ScrapedPage]]:
    usable = [page for page in pages if page.content and len(page.content) >= 80]
    return [usable[i:i + PAGES_PER_CALL] for i in range(0, len(usable), PAGES_PER_CALL)]


def _to_entity(raw: dict, page: ScrapedPage, columns: list[str]) -> Entity | None:
    cells: dict[str, CellValue] = {}
    for col in columns:
        field = raw.get(col, {})
        if not isinstance(field, dict):
            continue
        val = field.get("value")
        if val is None:
            continue
        cells[col] = CellValue(
            value=val,
            confidence=float(field.get("confidence", 0.8)),
            sources=[SourceRef(
                url=page.url,
                title=page.title,
                snippet=str(field.get("snippet", ""))[:400],
            )],
        )
    if "name" not in cells:
        return None
    name_val = str(cells["name"].value)
    if len(name_val) > 60 or len(name_val.split()) > 6:
        return None
    return Entity(id=str(uuid.uuid4()), cells=cells)


async def _extract_from_batch(
    client: AsyncOpenAI,
    batch: list[ScrapedPage],
    columns: list[str],
    entity_type: str,
    semaphore: asyncio.Semaphore,
) -> list[Entity]:
    pages_text = "\n".join(
        PAGE_TEMPLATE.format(number=i, url=page.url, title=page.title, content=page.content[:PAGE_CHARS])
        for i, page in enumerate(batch, start=1)
    )
    urls = [page.url for page in batch]

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=EXTRACTOR_MODEL,
                messages=[
                    {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(
                        entity_type=entity_type,
                        columns=", ".join(columns),
                        pages=pages_text,
                    )},
                ],
                max_tokens=4096,
                reasoning_effort=LLM_REASONING,
                temperature=0.1,
            )
        except Exception as exc:
            logger.error("Extraction failed for %s: %s", urls, exc)
            return []

    text = (response.choices[0].message.content or "").strip()
    data = extract_json_obj(text)
    if data is None:
        logger.warning("Extractor: could not parse JSON from response for %s", urls)
        return []

    entities: list[Entity] = []
    for raw in data.get("entities", []):
        if not isinstance(raw, dict):
            continue
        try:
            number = int(raw.get("page", 0))
        except (TypeError, ValueError):
            continue
        if not 1 <= number <= len(batch):
            continue
        entity = _to_entity(raw, batch[number - 1], columns)
        if entity:
            entities.append(entity)

    logger.info("Extracted %d entities from %d pages", len(entities), len(batch))
    return entities


async def extract_from_pages(
    client: AsyncOpenAI,
    pages: list[ScrapedPage],
    columns: list[str],
    entity_type: str,
) -> list[Entity]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)
    tasks = [
        _extract_from_batch(client, batch, columns, entity_type, semaphore)
        for batch in _batches(pages)
    ]
    results = await asyncio.gather(*tasks)
    return [e for batch_entities in results for e in batch_entities]
