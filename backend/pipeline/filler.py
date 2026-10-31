from __future__ import annotations
import asyncio, json, logging, re, unicodedata
from dataclasses import dataclass, field

from openai import AsyncOpenAI
from ..config import LLM_REASONING, FILLER_MODEL
from ..models import Entity, CellValue, SourceRef
from .embedder import Chunk, ChunkIndex
from .resolver import _normalise_name
from .utils import extract_json_obj

logger = logging.getLogger(__name__)

MAX_FILL_ENTITIES = 15
SEARCH_K = 5
CHUNKS_PER_CELL = 2
MIN_SCORE = 0.6
MAX_CHUNKS = 30
FILL_CONFIDENCE = 0.8

FILLER_SYSTEM_PROMPT = """You fill empty table cells using ONLY the text chunks given to you.

STRICT RULES:
1. Each cell lists the chunk ids you may use. Use only those chunks for that cell.
2. Fill a value only if the chunk clearly states it about that exact entity. A chunk can mention several entities, read carefully.
3. Copy a short exact quote (max 20 words) from the chunk that contains the value.
4. Keep values short (e.g. "2019", "Palo Alto, California", "Series B").
5. If no chunk states the value, set value to null. Never guess.

Respond with ONLY a raw JSON object:
{
  "cells": [
    {"id": 0, "value": "...", "chunk": "c1", "quote": "exact words from the chunk"},
    {"id": 1, "value": null, "chunk": null, "quote": null}
  ]
}"""


@dataclass
class Gap:
    entity: Entity
    column: str
    chunk_ids: list[str] = field(default_factory=list)


def _plain(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.findall(r"\w+", text.lower()))


def find_gaps(entities: list[Entity], columns: list[str]) -> list[Gap]:
    return [
        Gap(entity=entity, column=col)
        for entity in entities[:MAX_FILL_ENTITIES]
        for col in columns
        if col != "name" and (col not in entity.cells or entity.cells[col].value is None)
    ]


def _mentions(chunk: Chunk, entity: Entity) -> bool:
    name = _normalise_name(entity.get_name())
    return bool(name) and f" {name} " in f" {_plain(chunk.text)} "


def _query(gap: Gap) -> str:
    return f"{gap.entity.get_name()} {gap.column.replace('_', ' ')}"


def _attach_chunks(gaps: list[Gap], hits: list[list[tuple[Chunk, float]]]) -> dict[str, Chunk]:
    chunk_ids: dict[int, str] = {}
    chosen: dict[str, Chunk] = {}
    for gap, gap_hits in zip(gaps, hits):
        usable = [c for c, score in gap_hits if score >= MIN_SCORE and _mentions(c, gap.entity)]
        for chunk in usable[:CHUNKS_PER_CELL]:
            key = id(chunk)
            if key not in chunk_ids:
                if len(chosen) >= MAX_CHUNKS:
                    continue
                chunk_ids[key] = f"c{len(chosen) + 1}"
                chosen[chunk_ids[key]] = chunk
            gap.chunk_ids.append(chunk_ids[key])
    return chosen


def _quote_ok(value, quote, chunk: Chunk) -> bool:
    if value is None or not quote:
        return False
    plain_quote = _plain(str(quote))
    if not plain_quote or plain_quote not in _plain(chunk.text):
        return False
    quote_words = set(plain_quote.split())
    return all(word in quote_words for word in _plain(str(value)).split())


def _build_prompt(entity_type: str, cells: list[Gap], chunks: dict[str, Chunk]) -> str:
    cell_rows = [
        {"id": i, "entity": gap.entity.get_name(), "column": gap.column, "chunks": gap.chunk_ids}
        for i, gap in enumerate(cells)
    ]
    chunk_rows = [{"id": cid, "text": chunk.text} for cid, chunk in chunks.items()]
    return (
        f"Entity type: {entity_type}\n\n"
        f"CHUNKS:\n{json.dumps(chunk_rows, indent=1)}\n\n"
        f"CELLS:\n{json.dumps(cell_rows, indent=1)}"
    )


async def fill_from_chunks(
    client: AsyncOpenAI,
    entities: list[Entity],
    columns: list[str],
    entity_type: str,
    index: ChunkIndex | None,
) -> tuple[list[Entity], int]:
    gaps = find_gaps(entities, columns)
    if not gaps or index is None or not len(index):
        return entities, 0

    hits = await asyncio.to_thread(index.search, [_query(g) for g in gaps], SEARCH_K)
    chunks = _attach_chunks(gaps, hits)
    cells = [gap for gap in gaps if gap.chunk_ids]
    if not cells:
        logger.info("Fill: no chunk mentions any entity with gaps")
        return entities, 0

    try:
        response = await client.chat.completions.create(
            model=FILLER_MODEL,
            messages=[
                {"role": "system", "content": FILLER_SYSTEM_PROMPT},
                {"role": "user", "content": _build_prompt(entity_type, cells, chunks)},
            ],
            max_tokens=4096,
            reasoning_effort=LLM_REASONING,
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = extract_json_obj((response.choices[0].message.content or "").strip())
    except Exception as exc:
        logger.warning("Fill call failed: %s", exc)
        return entities, 0
    if data is None:
        logger.warning("Fill: could not parse JSON")
        return entities, 0

    filled = 0
    for row in data.get("cells", []):
        if not isinstance(row, dict) or not isinstance(row.get("id"), int):
            continue
        if not 0 <= row["id"] < len(cells):
            continue
        gap = cells[row["id"]]
        chunk_id = row.get("chunk")
        if chunk_id not in gap.chunk_ids:
            continue
        chunk = chunks[chunk_id]
        value, quote = row.get("value"), row.get("quote")
        if gap.column in gap.entity.cells and gap.entity.cells[gap.column].value is not None:
            continue
        if not _quote_ok(value, quote, chunk):
            logger.info("Fill: dropped %s / %s, quote not found in chunk", gap.entity.get_name(), gap.column)
            continue
        gap.entity.cells[gap.column] = CellValue(
            value=value,
            confidence=FILL_CONFIDENCE,
            sources=[SourceRef(url=chunk.url, title=chunk.title, snippet=str(quote)[:400])],
        )
        filled += 1

    logger.info("Fill: %d of %d gaps filled from %d chunks", filled, len(gaps), len(chunks))
    return entities, filled
