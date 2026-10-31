from __future__ import annotations
import logging, re
from typing import Any

import numpy as np

from ..models import Entity, RankingInfo
from ..utils import is_missing
from .resolver import _source_count

logger = logging.getLogger(__name__)

TOP_N = 10
MIN_KEEP_COLUMNS = 3
FALLBACK_FILL = 0.8
RELEVANCE_STEP = 0.1

UNIT_MULT = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
NUMBER_PATTERN = re.compile(r"(?P<val>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>[KkMmBb](?:illion)?)?\b")


def _text(entity: Entity, column: str) -> str:
    cell = entity.cells.get(column)
    if cell is None or is_missing(cell.value):
        return ""
    return str(cell.value)


def _has_phrase(text: str, phrase: str) -> bool:
    phrase = phrase.strip().lower()
    return bool(phrase) and re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text.lower()) is not None


def _to_number(value: Any) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\$+", text):
        return float(len(text))
    numbers = []
    for match in NUMBER_PATTERN.finditer(text):
        number = float(match.group("val").replace(",", ""))
        unit = match.group("unit")
        if unit:
            number *= UNIT_MULT[unit[0].lower()]
        numbers.append(number)
    return max(numbers) if numbers else None


def _passes(actual: float, op: str, limit: float) -> bool:
    return {">": actual > limit, ">=": actual >= limit, "<": actual < limit, "<=": actual <= limit}.get(op, False)


def _location_met(entity: Entity, rule: dict) -> bool:
    text = _text(entity, rule["column"])
    return any(_has_phrase(text, name) for name in rule["names"])


def _numeric_met(entity: Entity, rule: dict) -> bool:
    actual = _to_number(_text(entity, rule["column"]))
    return actual is not None and _passes(actual, rule["op"], rule["value"])


def _categorical_met(entity: Entity, rule: dict) -> bool:
    row = " ".join(_text(entity, col) for col in entity.cells)
    return any(_has_phrase(row, kw) for kw in rule["keywords"])


def _checks(rules: dict) -> list[tuple[str, Any, dict]]:
    checks = []
    if rules.get("location"):
        rule = rules["location"]
        checks.append((f"In {rule['names'][0]}", _location_met, rule))
    for rule in rules.get("numeric", []):
        value = int(rule["value"]) if float(rule["value"]).is_integer() else rule["value"]
        checks.append((f"{rule['column'].replace('_', ' ')} {rule['op']} {value}", _numeric_met, rule))
    for rule in rules.get("categorical", []):
        checks.append((f"{rule['column'].replace('_', ' ')}: {rule['keywords'][0]}", _categorical_met, rule))
    return checks


def _rule_columns(rules: dict) -> set[str]:
    columns = {rule["column"] for rule in rules.get("numeric", []) + rules.get("categorical", [])}
    if rules.get("location"):
        columns.add(rules["location"]["column"])
    return columns


def _relevance(entities: list[Entity], columns: list[str], query: str, model) -> list[float]:
    if model is None or not entities:
        return [0.0] * len(entities)
    try:
        rows = [" | ".join(_text(e, col) for col in columns if _text(e, col)) for e in entities]
        row_vecs = np.asarray(list(model.embed(rows)), dtype="float32")
        query_vec = np.asarray(list(model.query_embed([query])), dtype="float32")[0]
        row_vecs /= np.linalg.norm(row_vecs, axis=1, keepdims=True) + 1e-9
        query_vec /= np.linalg.norm(query_vec) + 1e-9
        return [float(s) for s in row_vecs @ query_vec]
    except Exception as exc:
        logger.warning("Relevance scoring failed: %s", exc)
        return [0.0] * len(entities)


def _fullness(entity: Entity, columns: list[str]) -> int:
    return sum(1 for col in columns if _text(entity, col))


def pick_columns(entities: list[Entity], columns: list[str], keep: set[str]) -> list[str]:
    if not entities:
        return columns
    fill = {col: sum(1 for e in entities if _text(e, col)) / len(entities) for col in columns}
    chosen = {col for col in columns if col == "name" or col in keep or fill[col] == 1.0}
    if len(chosen - {"name"}) < MIN_KEEP_COLUMNS:
        chosen |= {col for col in columns if fill[col] >= FALLBACK_FILL}
    return [col for col in columns if col in chosen]


def rank(
    entities: list[Entity],
    rules: dict,
    columns: list[str],
    query: str,
    model=None,
) -> tuple[list[Entity], list[str], RankingInfo]:
    rules = rules or {}
    checks = _checks(rules)
    labels = [label for label, _, _ in checks]
    if not entities:
        return [], columns, RankingInfo(total=len(checks), labels=labels, scores={})

    met = [sum(1 for _, check, rule in checks if check(e, rule)) for e in entities]
    relevance = _relevance(entities, columns, query, model)
    order = sorted(
        range(len(entities)),
        key=lambda i: (
            met[i],
            round(relevance[i] / RELEVANCE_STEP),
            _source_count(entities[i]),
            _fullness(entities[i], columns),
        ),
        reverse=True,
    )[:TOP_N]
    top = [entities[i] for i in order]
    shown = pick_columns(top, columns, _rule_columns(rules))

    for i in order:
        logger.info("[rank] %-35s met=%d/%d relevance=%.2f sources=%d",
                    entities[i].get_name(), met[i], len(checks), relevance[i], _source_count(entities[i]))
    logger.info("Columns shown: %s, dropped: %s", shown, [c for c in columns if c not in shown])
    return top, shown, RankingInfo(total=len(checks), labels=labels, scores={entities[i].id: met[i] for i in order})
