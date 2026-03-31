# Stage 6 — Gap Analyzer  (the "agentic" component)

# After Round 1, we inspect the partially-filled table:
#   • Which columns have low coverage? (< 50% of entities have a value)
#   • Which entities have many null cells?
#   • Are there too few entities overall?

# If there are meaningful gaps, we generate targeted follow-up queries for Round 2.
# This is what transforms a simple pipeline into a genuine agentic system —
# the output of one pass feeds back into the search plan for the next pass.

from __future__ import annotations
import json
import re
import logging

from cerebras.cloud.sdk import AsyncCerebras
from ..models import Entity, CellValue

logger = logging.getLogger(__name__)


_SYSTEM = """You are a data completeness analyst.

You will receive a partially-filled entity table and must:
1. Identify columns with low coverage (< 50% filled)
2. Note if there are too few entities (< 8 is usually thin)
3. Generate 4–6 ENTITY-SPECIFIC search queries to fill the gaps

CRITICAL: Do NOT generate broad topic queries like "AI healthcare startup headquarters".
Instead generate per-entity queries for the specific entities missing data, e.g.:
  - "Hippocratic AI founded year headquarters funding"
  - "Bayesian Health crunchbase funding stage location"
  - "Freenome series funding headquarters"

Pick the 4-6 entities with the most missing columns and generate one query per entity.
Each query should target 2-3 missing attributes for that specific entity.

Respond with ONLY a JSON object — no other text:
{
  "coverage": {"column_name": 0.75, ...},
  "gap_summary": "one sentence describing the main gaps",
  "gap_queries": ["EntityName missing_attr1 missing_attr2", ...],
  "should_continue": true
}

Set should_continue to false if coverage is already good (> 70% across all columns)
and entity count is adequate (>= 8 entities).
"""


def _compute_coverage(entities: list[Entity], columns: list[str]) -> dict[str, float]:
    if not entities:
        return {col: 0.0 for col in columns}
    return {
        col: sum(1 for e in entities if col in e.cells and e.cells[col].value is not None)
             / len(entities)
        for col in columns
    }


async def analyze_gaps(
    client: AsyncCerebras,
    entities: list[Entity],
    columns: list[str],
    original_query: str,
    entity_type: str,
) -> dict:
    if not entities:
        return {
            "gap_queries": [f"{original_query} list", f"top {original_query}"],
            "should_continue": True,
            "gap_summary": "No entities found yet — broadening search.",
        }

    coverage = _compute_coverage(entities, columns)
    avg_coverage = sum(coverage.values()) / len(coverage) if coverage else 0.0

    if avg_coverage >= 0.70 and len(entities) >= 8:
        logger.info("Coverage %.0f%% — skipping gap analysis", avg_coverage * 100)
        return {
            "gap_queries": [],
            "should_continue": False,
            "gap_summary": f"Coverage {avg_coverage:.0%} — sufficient.",
        }

    low_cols = [col for col, cov in coverage.items() if cov < 0.5]

    entity_gaps = []
    for e in entities:
        if "name" not in e.cells:
            continue
        name = str(e.cells["name"].value)
        missing = [col for col in low_cols if col not in e.cells or e.cells[col].value is None]
        if missing:
            entity_gaps.append({"name": name, "missing": missing})

    entity_gaps.sort(key=lambda x: -len(x["missing"]))
    entity_gaps = entity_gaps[:8]

    prompt = f"""Analyse gaps in this entity table:

Query: "{original_query}"  |  Entity type: {entity_type}
Entities found so far: {len(entities)}

Column coverage:
{json.dumps(coverage, indent=2)}

Entities with missing data (generate one search query per entity to fill their gaps):
{json.dumps(entity_gaps, indent=2)}

Generate per-entity queries to fill the missing attributes."""

    try:
        response = await client.chat.completions.create(
            model="qwen-3-235b-a22b-instruct-2507",
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
        )
        text = response.choices[0].message.content.strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            result = json.loads(match.group())
            logger.info("Gap analysis: %s", result.get("gap_summary", ""))
            return result
    except Exception as exc:
        logger.warning("Gap analyzer LLM call failed: %s", exc)

    fallback_queries = []
    for eg in entity_gaps[:3]:
        fallback_queries.append(f"{eg['name']} {' '.join(eg['missing'][:2])}")
    if not fallback_queries:
        fallback_queries.append(f"{original_query} {' '.join(low_cols[:2])} information")

    return {
        "gap_queries": fallback_queries[:3],
        "should_continue": avg_coverage < 0.70 or len(entities) < 6,
        "gap_summary": f"Low coverage on: {', '.join(low_cols[:4])}",
    }
