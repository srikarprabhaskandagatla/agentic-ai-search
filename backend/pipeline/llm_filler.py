"""
Stage 7 — LLM Knowledge Filler

Last-resort pass that asks the LLM to fill remaining null cells from its
parametric knowledge. Filled values are marked llm_filled=True so the
frontend can star them and show a disclaimer.
"""

from __future__ import annotations
import json
import logging

from cerebras.cloud.sdk import AsyncCerebras
from ..models import Entity, CellValue
from .utils import extract_json_arr

logger = logging.getLogger(__name__)

_MODEL = "qwen-3-235b-a22b-instruct-2507"


async def llm_fill_gaps(
    client: AsyncCerebras,
    entities: list[Entity],
    columns: list[str],
    entity_type: str,
) -> list[Entity]:
    """
    For each entity that has null cells, ask the LLM to fill them from
    parametric knowledge. Returns the same entity list with gaps filled where
    the LLM had knowledge (confidence 0.5, llm_filled=True, no sources).
    """
    # Build the work list: only entities that have at least one missing column
    non_name_cols = [c for c in columns if c != "name"]
    work = []
    for entity in entities:
        missing = [
            c for c in non_name_cols
            if c not in entity.cells or entity.cells[c].value is None
        ]
        if missing:
            work.append((entity, missing))

    if not work:
        return entities

    # Build a compact payload for the LLM
    payload = []
    for entity, missing_cols in work:
        known = {
            c: entity.cells[c].value
            for c in non_name_cols
            if c in entity.cells and entity.cells[c].value is not None
        }
        payload.append({
            "name": entity.get_name(),
            "known": known,
            "fill": missing_cols,
        })

    prompt = (
        f"You are filling missing fields for {entity_type} entities "
        "from your training knowledge.\n\n"
        "Rules:\n"
        "- Fill ONLY the fields listed in 'fill'.\n"
        "- Use concise values (e.g. '2019', 'San Francisco, CA', 'Series B').\n"
        "- If you genuinely don't know, use null.\n"
        "- Do NOT invent plausible-sounding values you are not confident about.\n\n"
        f"Entities:\n{json.dumps(payload, indent=2)}\n\n"
        "Respond with ONLY a JSON array — one object per entity, same order:\n"
        '[{"name": "...", "<field>": "<value or null>", ...}, ...]'
    )

    try:
        resp = await client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        filled_list = extract_json_arr(raw)
        if filled_list is None:
            raise ValueError("No JSON array found in LLM filler response")
    except Exception as exc:
        logger.warning("LLM filler failed: %s", exc)
        return entities

    # Map by name for O(1) lookup
    fill_map: dict[str, dict] = {
        item["name"]: item
        for item in filled_list
        if isinstance(item, dict) and "name" in item
    }

    result: list[Entity] = []
    for entity in entities:
        name = entity.get_name()
        fills = fill_map.get(name, {})
        if not fills:
            result.append(entity)
            continue

        new_cells = dict(entity.cells)
        for col in non_name_cols:
            if col not in new_cells or new_cells[col].value is None:
                raw_val = fills.get(col)
                if raw_val is not None and raw_val != "null" and raw_val != "":
                    new_cells[col] = CellValue(
                        value=str(raw_val),
                        confidence=0.5,
                        sources=[],
                        llm_filled=True,
                    )
        result.append(entity.model_copy(update={"cells": new_cells}))

    return result
