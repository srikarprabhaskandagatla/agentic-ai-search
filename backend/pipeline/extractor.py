from __future__ import annotations
import uuid, logging
from openai import AsyncOpenAI
from ..config import LLM_REASONING, EXTRACTOR_MODEL, EXTRACT_INPUT_TOKENS, estimate_tokens, fit_text, pacer
from ..models import Entity, CellValue, SourceRef
from ..embedder import Chunk
from ..utils import extract_json_obj, is_missing

logger = logging.getLogger(__name__)

OUTPUT_TOKENS = 2500
EXTRACT_CONFIDENCE = 0.9

EXTRACTOR_SYSTEM_PROMPT = """You are a precise information extraction engine.

You get numbered text chunks from web pages and a list of columns. Extract ALL entities of the target entity type from every chunk.

STRICT RULES:
1. Only extract values EXPLICITLY present in the chunk. Never infer or hallucinate.
2. For every value, copy a short exact phrase (max 15 words) from that chunk into "snippet".
3. Leave out any field that is not found. Never write placeholders like "unknown", "not specified" or "N/A".
4. "name" must always be present - if you can't find a clear name, skip the entity.
5. ALWAYS set "chunk" to the number of the chunk the entity came from. If an entity is in two chunks, list it once per chunk.

Respond with ONLY a raw JSON object - no markdown, no preamble:
{
  "entities": [
    {
      "chunk": 1,
      "name":     {"value": "...", "snippet": "exact phrase"},
      "column_b": {"value": "...", "snippet": "exact phrase"}
    }
  ]
}"""

USER_TEMPLATE = """Extract all {entity_type} entities from these chunks.
Columns: {columns}

{chunks}
"""

CHUNK_TEMPLATE = """[CHUNK {number}] {title}
{text}
"""


def _to_entity(raw: dict, chunk: Chunk, columns: list[str]) -> Entity | None:
    cells: dict[str, CellValue] = {}
    for col in columns:
        field = raw.get(col, {})
        if not isinstance(field, dict):
            continue
        val = field.get("value")
        if is_missing(val):
            continue
        cells[col] = CellValue(
            value=val,
            confidence=EXTRACT_CONFIDENCE,
            sources=[SourceRef(
                url=chunk.url,
                title=chunk.title,
                snippet=str(field.get("snippet", ""))[:400],
            )],
        )
    if "name" not in cells:
        return None
    name_val = str(cells["name"].value)
    if len(name_val) > 60 or len(name_val.split()) > 6:
        return None
    return Entity(id=str(uuid.uuid4()), cells=cells)


def _chunk_for(raw: dict, chunks: list[Chunk]) -> Chunk | None:
    try:
        number = int(raw.get("chunk"))
        if 1 <= number <= len(chunks):
            return chunks[number - 1]
    except (TypeError, ValueError):
        pass
    name = raw.get("name", {})
    name = str(name.get("value", "") if isinstance(name, dict) else name).strip().lower()
    if not name:
        return None
    return next((chunk for chunk in chunks if name in chunk.text.lower()), None)


def _chunks_text(chunks: list[Chunk]) -> tuple[str, list[Chunk]]:
    parts: list[str] = []
    kept: list[Chunk] = []
    used = 0
    for chunk in chunks:
        part = CHUNK_TEMPLATE.format(number=len(kept) + 1, title=chunk.title, text=chunk.text)
        cost = estimate_tokens(part)
        if used + cost > EXTRACT_INPUT_TOKENS:
            if not kept:
                parts.append(fit_text(part, EXTRACT_INPUT_TOKENS))
                kept.append(chunk)
            break
        parts.append(part)
        kept.append(chunk)
        used += cost
    return "\n".join(parts), kept


async def extract_from_chunks(
    client: AsyncOpenAI,
    chunks: list[Chunk],
    columns: list[str],
    entity_type: str,
) -> list[Entity]:
    if not chunks:
        return []

    chunks_text, kept = _chunks_text(chunks)
    user = USER_TEMPLATE.format(entity_type=entity_type, columns=", ".join(columns), chunks=chunks_text)
    event = await pacer.reserve(estimate_tokens(EXTRACTOR_SYSTEM_PROMPT + user) + OUTPUT_TOKENS)
    try:
        response = await client.chat.completions.create(
            model=EXTRACTOR_MODEL,
            messages=[
                {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            max_tokens=OUTPUT_TOKENS,
            reasoning_effort=LLM_REASONING,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        logger.error("Extraction failed: %s", exc)
        return []
    pacer.settle(event, getattr(response, "usage", None), "extract")

    data = extract_json_obj((response.choices[0].message.content or "").strip())
    if data is None:
        logger.warning("Extractor: could not parse JSON")
        return []

    entities: list[Entity] = []
    for raw in data.get("entities", []):
        if not isinstance(raw, dict):
            continue
        chunk = _chunk_for(raw, kept)
        if chunk is None:
            continue
        entity = _to_entity(raw, chunk, columns)
        if entity:
            entities.append(entity)

    logger.info("Extracted %d entities from %d chunks", len(entities), len(kept))
    return entities
