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
    chunks: list[Chunk] = field(default_factory=list)


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


def _attach_chunks(
    gaps: list[Gap],
    hits: list[list[tuple[Chunk, float]]],
    token_budget: int,
    tried: dict[tuple[str, str], set[str]],
) -> list[Gap]:
    used = 0
    for gap, gap_hits in zip(gaps, hits):
        seen = tried.setdefault((gap.entity.id, gap.column), set())
        for chunk, score in gap_hits:
            if len(gap.chunks) == CHUNKS_PER_CELL:
                break
            if score < MIN_SCORE or chunk.text in seen or not _mentions(chunk, gap.entity):
                continue
            piece = _trim(chunk, gap.entity)
            if piece in gap.chunks:
                continue
            if not any(piece in g.chunks for g in gaps):
                cost = estimate_tokens(piece.text) + CELL_TOKENS
                if used + cost > token_budget:
                    continue
                used += cost
            gap.chunks.append(piece)
            seen.add(chunk.text)
    return [gap for gap in gaps if gap.chunks]


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


def _build_prompt(entity_type: str, gaps: list[Gap]) -> tuple[str, dict[str, Chunk]]:
    chunks: dict[str, Chunk] = {}
    cell_rows = []
    for i, gap in enumerate(gaps):
        refs = []
        for chunk in gap.chunks:
            cid = next((k for k, c in chunks.items() if c == chunk), None)
            if cid is None:
                cid = f"c{len(chunks) + 1}"
                chunks[cid] = chunk
            refs.append(cid)
        cell_rows.append({"id": i, "entity": gap.entity.get_name(), "column": gap.column, "chunks": refs})
    chunk_rows = [{"id": cid, "text": chunk.text} for cid, chunk in chunks.items()]
    prompt = (
        f"Entity type: {entity_type}\n\n"
        f"CHUNKS:\n{json.dumps(chunk_rows, ensure_ascii=False)}\n\n"
        f"CELLS:\n{json.dumps(cell_rows, ensure_ascii=False)}"
    )
    return prompt, chunks


def _call_tokens(user: str) -> int:
    return estimate_tokens(FILLER_SYSTEM_PROMPT + user) + OUTPUT_TOKENS


async def fill_from_chunks(
    client: AsyncOpenAI,
    entities: list[Entity],
    columns: list[str],
    entity_type: str,
    index: ChunkIndex | None,
    token_budget: int = FILL_INPUT_TOKENS,
    tried: dict[tuple[str, str], set[str]] | None = None,
    token_limit: int = TOKENS_PER_MINUTE,
) -> tuple[list[Entity], int]:
    gaps = find_gaps(entities, columns)
    if not gaps or index is None or not len(index):
        return entities, 0
    tried = {} if tried is None else tried
    total = len(gaps)

    hits = await asyncio.to_thread(index.search, [_query(g) for g in gaps], SEARCH_K)
    gaps = _attach_chunks(gaps, hits, token_budget, tried)
    if not gaps:
        logger.info("Fill: no new chunk mentions any entity with gaps")
        return entities, 0

    limit = min(token_limit, TOKENS_PER_MINUTE)
    user_prompt, chunks = _build_prompt(entity_type, gaps)
    while len(gaps) > 1 and _call_tokens(user_prompt) > limit:
        gaps = gaps[:-1]
        user_prompt, chunks = _build_prompt(entity_type, gaps)

    try:
        event = await pacer.reserve(_call_tokens(user_prompt))
        response = await client.chat.completions.create(
            model=FILLER_MODEL,
            messages=[
                {"role": "system", "content": FILLER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
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
        if not 0 <= row["id"] < len(gaps):
            continue
        gap = gaps[row["id"]]
        chunk = chunks.get(row.get("chunk"))
        if chunk is None or chunk not in gap.chunks:
            continue
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

    logger.info("Fill: %d of %d gaps filled from %d chunks", filled, total, len(chunks))
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
    tried: dict[tuple[str, str], set[str]] = {}

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

        fast = max_fills == 1
        limit = pacer.available() if fast else TOKENS_PER_MINUTE
        result.entities, filled = await fill_from_chunks(
            client, result.entities, columns, entity_type, index, _fill_budget(fast), tried, limit,
        )
        result.filled += filled
        result.rounds = round_no
        if progress:
            await progress(f"Round {round_no}: filled {filled} empty cells")
        if round_no > 1 and filled == 0:
            break

    return result
