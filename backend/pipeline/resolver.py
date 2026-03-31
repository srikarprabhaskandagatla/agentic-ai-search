# Stage 5 — Resolver

# Deduplicate and merge entities from across all pages.

# The challenge: the same entity (e.g. "Tempus AI") may have been found
# on 5 different pages, each providing different attribute values.
# We need to:
#   1. Identify which entities refer to the same real-world thing
#   2. Merge their cells, preferring higher-confidence values
#   3. Accumulate all source references so every merged cell is fully traceable

# Approach:
#   - Fast path: exact name normalisation deduplciation (lowercase, strip punctuation)
#   - Slow path (when fast path leaves ambiguity): Claude Haiku groups remaining near-duplicates
#   - Merge: for each column, pick the highest-confidence value, union all sources

from __future__ import annotations
import json
import re
import logging
import unicodedata

from anthropic import AsyncAnthropic
from ..models import Entity, CellValue, SourceRef

logger = logging.getLogger(__name__)

_MAX_ENTITIES_OUT = 35   # Hard cap on final table size



# Name normalisation


def _normalise_name(name: str) -> str:
    """Lowercase, strip punctuation and accents for fuzzy matching."""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    # Remove common suffixes that cause false mismatches
    for suffix in (" inc", " llc", " ltd", " corp", " co"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name



# Cell merging


def _merge_cells(cells_list: list[CellValue]) -> CellValue:
    """Merge multiple CellValue objects: best value + all sources."""
    best = max(cells_list, key=lambda c: c.confidence)
    all_sources: list[SourceRef] = []
    seen_urls: set[str] = set()
    for cell in cells_list:
        for src in cell.sources:
            if src.url not in seen_urls:
                seen_urls.add(src.url)
                all_sources.append(src)
    return CellValue(value=best.value, confidence=best.confidence, sources=all_sources)


def _merge_entity_group(entities: list[Entity]) -> Entity:
    """Merge a group of duplicate entities into one, combining all columns."""
    if len(entities) == 1:
        return entities[0]

    all_columns: set[str] = set()
    for e in entities:
        all_columns.update(e.cells.keys())

    merged_cells: dict[str, CellValue] = {}
    for col in all_columns:
        candidates = [e.cells[col] for e in entities if col in e.cells]
        if candidates:
            merged_cells[col] = _merge_cells(candidates)

    return Entity(id=entities[0].id, cells=merged_cells)



# Fast dedup by normalised name


def _fast_dedup(entities: list[Entity]) -> list[list[Entity]]:
    """Group entities by normalised name (O(n))."""
    groups: dict[str, list[Entity]] = {}
    for entity in entities:
        name_cell = entity.cells.get("name")
        if not name_cell or name_cell.value is None:
            continue
        key = _normalise_name(str(name_cell.value))
        if key not in groups:
            groups[key] = []
        groups[key].append(entity)
    return list(groups.values())



# LLM-assisted near-duplicate detection


_DEDUP_SYSTEM = """You are deduplicating a list of entity names.
Some entries may be near-duplicates (same company, different formatting).
Group them.

Respond with ONLY a JSON object — no other text:
{
  "merge_groups": [
    [0, 3],   // indices of entities that are the same real-world thing
    [1],
    [2, 5, 7],
    ...
  ]
}
Each index must appear in exactly one group."""


async def _llm_dedup(
    client: AsyncAnthropic,
    name_index: list[dict],
) -> list[list[int]]:
    """Use Claude Haiku to detect near-duplicates by name."""
    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=_DEDUP_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Deduplicate:\n{json.dumps(name_index, indent=2)}",
        }],
    )
    text = response.content[0].text.strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return [[i] for i in range(len(name_index))]  # fallback: no merges
    try:
        data = json.loads(match.group())
        return data.get("merge_groups", [[i] for i in range(len(name_index))])
    except json.JSONDecodeError:
        return [[i] for i in range(len(name_index))]



# Public API


async def resolve_entities(
    client: AsyncAnthropic,
    raw_entities: list[Entity],
) -> list[Entity]:
    """Deduplicate and merge raw entities, returning a clean list."""

    if not raw_entities:
        return []

    logger.info("Resolving %d raw entities", len(raw_entities))

    # Step 1: Fast dedup by normalised name
    fast_groups = _fast_dedup(raw_entities)
    partially_merged = [_merge_entity_group(g) for g in fast_groups]

    # Step 2: LLM near-dedup if there are many remaining (catch "OpenAI" vs "Open AI" etc.)
    if len(partially_merged) > 5:
        name_index = [
            {
                "idx": i,
                "name": str(e.cells["name"].value) if "name" in e.cells else f"entity_{i}",
            }
            for i, e in enumerate(partially_merged)
        ]
        try:
            groups = await _llm_dedup(client, name_index)
            final_groups: list[list[Entity]] = []
            for group_indices in groups:
                valid = [partially_merged[i] for i in group_indices if 0 <= i < len(partially_merged)]
                if valid:
                    final_groups.append(valid)
            resolved = [_merge_entity_group(g) for g in final_groups]
        except Exception as exc:
            logger.warning("LLM dedup failed (%s), using fast dedup result", exc)
            resolved = partially_merged
    else:
        resolved = partially_merged

    # Sort by name, cap output
    resolved.sort(key=lambda e: str(e.cells.get("name", CellValue(value="zzz", sources=[])).value).lower())
    logger.info("Resolved to %d unique entities", len(resolved))
    return resolved[:_MAX_ENTITIES_OUT]
