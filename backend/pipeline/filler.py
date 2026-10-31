from __future__ import annotations
import asyncio, json, logging, re, unicodedata
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from openai import AsyncOpenAI
from ..config import (
    LLM_REASONING, FILLER_MODEL, FILL_INPUT_TOKENS, MIN_FILL_TOKENS, TOKENS_PER_MINUTE,
    estimate_tokens, pacer,
)
from ..models import Entity, CellValue, SourceRef
from ..embedder import Chunk, ChunkIndex, embed_pages
from .resolver import _normalise_name
from .retriever import retrieve
from ..utils import extract_json_obj, is_missing

logger = logging.getLogger(__name__)

MAX_FILL_ENTITIES = 15
SEARCH_K = 8
CHUNKS_PER_CELL = 2
MIN_SCORE = 0.6
OUTPUT_TOKENS = 1200
PROMPT_TOKENS = 400
CELL_TOKENS = 25
FILL_CONFIDENCE = 0.8
MAX_TARGETED = 8

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
        if col != "name" and (col not in entity.cells or is_missing(entity.cells[col].value))
    ]


def _mentions(chunk: Chunk, entity: Entity) -> bool:
    name = _normalise_name(entity.get_name())
    return bool(name) and f" {name} " in f" {_plain(chunk.text)} "


def _query(gap: Gap) -> str:
    return f"{gap.entity.get_name()} {gap.column.replace('_', ' ')}"


def _trim(chunk: Chunk, entity: Entity) -> Chunk:
    sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
    name = _normalise_name(entity.get_name())
    hits = [i for i, s in enumerate(sentences) if f" {name} " in f" {_plain(s)} "]
    if not hits:
        return chunk
    keep = sorted({j for i in hits for j in (i - 1, i, i + 1) if 0 <= j < len(sentences)})
    return Chunk(url=chunk.url, title=chunk.title, text=" ".join(sentences[j] for j in keep))


def _cell_key(gap: Gap) -> tuple[str, str]:
    return gap.entity.id, gap.column


def _attach_chunks(
    gaps: list[Gap],
    hits: list[list[tuple[Chunk, float]]],
    token_budget: int,
    sent: dict[tuple[str, str], set[int]],
) -> dict[str, Chunk]:
    chunk_ids: dict[tuple[int, str], str] = {}
    chosen: dict[str, Chunk] = {}
    used = 0
    for gap, gap_hits in zip(gaps, hits):
        already = sent.setdefault(_cell_key(gap), set())
        usable = [
            c for c, score in gap_hits
            if score >= MIN_SCORE and id(c) not in already and _mentions(c, gap.entity)
        ]
        for chunk in usable[:CHUNKS_PER_CELL]:
            key = (id(chunk), gap.entity.id)
            if key not in chunk_ids:
                trimmed = _trim(chunk, gap.entity)
                cost = estimate_tokens(trimmed.text) + CELL_TOKENS
                if used + cost > token_budget:
                    continue
                used += cost
                chunk_ids[key] = f"c{len(chosen) + 1}"
                chosen[chunk_ids[key]] = trimmed
            gap.chunk_ids.append(chunk_ids[key])
            already.add(id(chunk))
    return chosen


def _quote_ok(value, quote, chunk: Chunk) -> bool:
    if is_missing(value) or not quote:
        return False
    plain_quote = _plain(str(quote))
    if not plain_quote or plain_quote not in _plain(chunk.text):
        return False
    quote_words = set(plain_quote.split())
    return all(_word_in_quote(word, quote_words) for word in _plain(str(value)).split())


def _word_in_quote(word: str, quote_words: set[str]) -> bool:
    if word in quote_words:
        return True
    if not word.isdigit():
        return False
    return any(re.match(r"\d+", q) and re.match(r"\d+", q).group() == word for q in quote_words)


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
    token_budget: int = FILL_INPUT_TOKENS,
    sent: dict[tuple[str, str], set[int]] | None = None,
) -> tuple[list[Entity], int]:
    gaps = find_gaps(entities, columns)
    if not gaps or index is None or not len(index):
        return entities, 0
    sent = {} if sent is None else sent

    hits = await asyncio.to_thread(index.search, [_query(g) for g in gaps], SEARCH_K)
    chunks = _attach_chunks(gaps, hits, token_budget, sent)
    cells = [gap for gap in gaps if gap.chunk_ids]
    if not cells:
        logger.info("Fill: no new chunk mentions any entity with gaps")
        return entities, 0

    user = _build_prompt(entity_type, cells, chunks)
    while len(cells) > 1 and estimate_tokens(FILLER_SYSTEM_PROMPT + user) + OUTPUT_TOKENS > TOKENS_PER_MINUTE:
        cells = cells[:-1]
        user = _build_prompt(entity_type, cells, chunks)

    try:
        event = await pacer.reserve(estimate_tokens(FILLER_SYSTEM_PROMPT + user) + OUTPUT_TOKENS)
        response = await client.chat.completions.create(
            model=FILLER_MODEL,
            messages=[
                {"role": "system", "content": FILLER_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            max_tokens=OUTPUT_TOKENS,
            reasoning_effort=LLM_REASONING,
            temperature=0,
            response_format={"type": "json_object"},
        )
        pacer.settle(event, getattr(response, "usage", None), "fill")
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
        if gap.column in gap.entity.cells and not is_missing(gap.entity.cells[gap.column].value):
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


@dataclass
class FillResult:
    entities: list[Entity]
    filled: int = 0
    rounds: int = 0
    queries: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


def targeted_queries(gaps: list[Gap], entity_type: str, round_no: int) -> list[str]:
    missing: dict[str, list[str]] = {}
    for gap in gaps:
        missing.setdefault(gap.entity.get_name(), []).append(gap.column.replace("_", " "))
    queries = []
    for name, cols in list(missing.items())[:MAX_TARGETED]:
        context = f" {entity_type}" if round_no > 2 else ""
        queries.append(f"{name}{context} {' '.join(cols)}")
    return queries


def _fill_budget(fast: bool) -> int:
    if not fast:
        return FILL_INPUT_TOKENS
    left = pacer.available() - OUTPUT_TOKENS - PROMPT_TOKENS
    return max(MIN_FILL_TOKENS, min(FILL_INPUT_TOKENS, left))


async def fill_loop(
    client: AsyncOpenAI,
    entities: list[Entity],
    columns: list[str],
    entity_type: str,
    index: ChunkIndex | None,
    seen_urls: set[str],
    max_fills: int,
    progress: Callable[[str], Awaitable[None]] | None = None,
) -> FillResult:
    result = FillResult(entities=entities)
    if index is None:
        return result
    sent: dict[tuple[str, str], set[int]] = {}

    for round_no in range(1, max_fills + 1):
        if round_no > 1:
            gaps = find_gaps(result.entities, columns)
            if not gaps:
                break
            queries = targeted_queries(gaps, entity_type, round_no)
            if progress:
                await progress(f"Round {round_no}: searching for {len(queries)} entities with empty cells")
            try:
                pages = await retrieve(queries, seen_urls)
            except Exception as exc:
                logger.warning("Targeted search failed: %s", exc)
                break
            result.queries.extend(queries)
            seen_urls.update(page.url for page in pages)
            result.sources.extend(page.url for page in pages)
            if not pages:
                break
            chunks, vectors = await asyncio.to_thread(embed_pages, pages)
            index.add(chunks, vectors)

        budget = _fill_budget(max_fills == 1)
        result.entities, filled = await fill_from_chunks(
            client, result.entities, columns, entity_type, index, budget, sent,
        )
        result.filled += filled
        result.rounds = round_no
        if progress:
            await progress(f"Round {round_no}: filled {filled} empty cells")
        if round_no > 1 and filled == 0:
            break

    return result
