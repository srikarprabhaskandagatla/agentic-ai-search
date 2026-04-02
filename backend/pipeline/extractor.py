# Stage 4 - Extractor  (uses Cerebras / Llama 3.3 70B)

from __future__ import annotations
import asyncio
import uuid
import logging

from cerebras.cloud.sdk import AsyncCerebras
from ..models import Entity, CellValue, ScrapedPage, SourceRef
from .utils import extract_json_obj

logger = logging.getLogger(__name__)

_MAX_CONCURRENT_EXTRACTIONS = 8  # Cerebras handles high concurrency well

_SYSTEM = """You are a precise information extraction engine.

Given a web page and a list of columns, extract ALL entities that belong to the target entity type.

STRICT RULES:
1. Only extract values EXPLICITLY present in the content. Never infer or hallucinate.
2. For every non-null value, copy the exact phrase from the content into "snippet".
3. Set confidence: 1.0 = stated directly, 0.8 = light interpretation, 0.6 = implied, 0.4 = uncertain.
4. If a field is not found, set value to null.
5. "name" must always be present - if you can't find a clear name, skip the entity.

Respond with ONLY a raw JSON object - no markdown, no preamble:
{
  "entities": [
    {
      "name":     {"value": "...", "confidence": 0.99, "snippet": "exact phrase"},
      "column_b": {"value": "...", "confidence": 0.85, "snippet": "exact phrase"},
      "column_c": {"value": null,  "confidence": 0.0,  "snippet": ""}
    }
  ]
}"""

_USER_TEMPLATE = """\
Extract all {entity_type} entities from this page.
Columns: {columns}

URL: {url}
Title: {title}

CONTENT:
{content}
"""


async def _extract_from_page(
    client: AsyncCerebras,
    page: ScrapedPage,
    columns: list[str],
    entity_type: str,
    semaphore: asyncio.Semaphore,
) -> list[Entity]:
    if not page.content or len(page.content) < 80:
        return []

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="qwen-3-235b-a22b-instruct-2507",
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": _USER_TEMPLATE.format(
                        entity_type=entity_type,
                        columns=", ".join(columns),
                        url=page.url,
                        title=page.title,
                        content=page.content[:6000],
                    )},
                ],
                max_tokens=3000,
                temperature=0.1,
            )
        except Exception as exc:
            logger.error("Cerebras extraction failed for %s: %s", page.url, exc)
            return []

    text = response.choices[0].message.content.strip()
    data = extract_json_obj(text)
    if data is None:
        logger.warning("Extractor: could not parse JSON from response for %s", page.url)
        return []

    entities: list[Entity] = []
    for raw in data.get("entities", []):
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
            continue
        name_val = str(cells["name"].value)
        if len(name_val) > 60 or len(name_val.split()) > 6:
            continue
        entities.append(Entity(id=str(uuid.uuid4()), cells=cells))

    logger.info("Cerebras extracted %d entities from %s", len(entities), page.url)
    return entities


async def extract_from_pages(
    client: AsyncCerebras,
    pages: list[ScrapedPage],
    columns: list[str],
    entity_type: str,
) -> list[Entity]:
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_EXTRACTIONS)
    tasks = [
        _extract_from_page(client, page, columns, entity_type, semaphore)
        for page in pages
    ]
    results = await asyncio.gather(*tasks)
    return [e for page_entities in results for e in page_entities]
