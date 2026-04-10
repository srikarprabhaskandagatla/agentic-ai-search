# Stage 5.5 – Constraint-aware entity ranker
#
# Parses the user query into three generic constraint types, then re-orders
# entities so those satisfying MORE constraints always appear first.
#
# Constraint types:
#   location    – where the entity must be based (country, city, region)
#   numeric     – any threshold on a number field (funding, year, rating …)
#   categorical – required keywords / tags in any descriptive field
#                 (open source, no-code, pizza, AI, healthcare …)
#
# Ranking key (all descending):
#   1. n_satisfied   – entities satisfying ALL constraints → mandatory top
#   2. coverage      – within the same tier, more-complete entities rank higher
#
# If no constraints are found the original order is preserved.

from __future__ import annotations

import re
import logging
from typing import Any

from cerebras.cloud.sdk import AsyncCerebras

from ..models import Entity
from .utils import extract_json_obj

logger = logging.getLogger(__name__)

MODEL = "qwen-3-235b-a22b-instruct-2507"


# ── Geography lookups ─────────────────────────────────────────────────────────

_US_STATES_FULL = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york", "north carolina",
    "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
    "rhode island", "south carolina", "south dakota", "tennessee", "texas",
    "utah", "vermont", "virginia", "washington", "west virginia",
    "wisconsin", "wyoming",
}

_US_STATES_ABBR = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id",
    "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms",
    "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok",
    "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv",
    "wi", "wy", "dc",
}

# Canonical token sets for country/region queries.
# Keys are everything the LLM might return for that country.
_LOCATION_CANON: dict[str, set[str]] = {
    "us":             {"us", "usa", "u.s.", "u.s.a.", "united states",
                       "united states of america", "america"},
    "usa":            {"us", "usa", "u.s.", "u.s.a.", "united states",
                       "united states of america", "america"},
    "united states":  {"us", "usa", "u.s.", "u.s.a.", "united states",
                       "united states of america", "america"},
    "america":        {"us", "usa", "u.s.", "u.s.a.", "united states",
                       "united states of america", "america"},
    "uk":             {"uk", "u.k.", "united kingdom", "britain",
                       "england", "scotland", "wales", "northern ireland"},
    "united kingdom": {"uk", "u.k.", "united kingdom", "britain", "england"},
    "eu":             {"eu", "europe", "european union"},
    "europe":         {"eu", "europe", "european union"},
    "canada":         {"canada", "canadian"},
    "australia":      {"australia", "australian"},
    "india":          {"india", "indian"},
    "china":          {"china", "chinese"},
    "germany":        {"germany", "german"},
    "france":         {"france", "french"},
    "singapore":      {"singapore"},
    "israel":         {"israel", "israeli"},
    "brazil":         {"brazil", "brazilian"},
    "japan":          {"japan", "japanese"},
    "south korea":    {"south korea", "korea", "korean"},
    "netherlands":    {"netherlands", "dutch", "holland"},
    "sweden":         {"sweden", "swedish"},
}

# Column name fragments that indicate a geographic field
_GEO_COLUMN_KW = {
    "country", "location", "headquarter", "hq", "based", "region",
    "city", "state", "office", "address", "origin", "geography",
}

# Column name fragments that likely hold a numeric financial / size value
_NUMERIC_COLUMN_KW = {
    "funding", "raised", "investment", "capital", "revenue", "arr",
    "mrr", "valuation", "amount", "total", "employees", "headcount",
    "price", "cost", "rating", "score", "year", "founded",
}

# Column name fragments that hold descriptive / categorical text
_TEXT_COLUMN_KW = {
    "description", "type", "category", "tag", "label", "domain",
    "industry", "sector", "model", "cuisine", "style", "focus",
    "specialt", "product", "service", "feature", "about",
}

# Unit multipliers
_UNIT_MULT: dict[str, float] = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}

_NUMERIC_RE = re.compile(
    r"\$?\s*(?P<val>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<unit>[KkMmBb](?:illion|llion)?)?\b",
)


# ── Numeric helpers ───────────────────────────────────────────────────────────

def _parse_amount(val_str: str, unit_str: str) -> float:
    val = float(val_str.replace(",", ""))
    if unit_str:
        val *= _UNIT_MULT.get(unit_str[0].lower(), 1.0)
    return val


def _cell_to_number(value: Any) -> float | None:
    """Extract the best numeric value from a cell (handles '$15M', '15 million', etc.)."""
    if value is None:
        return None
    s = str(value)
    results: list[float] = []
    for val_str, unit_str in _NUMERIC_RE.findall(s):
        try:
            results.append(_parse_amount(val_str, unit_str))
        except ValueError:
            pass
    return max(results) if results else None


def _op_passes(actual: float, op: str, threshold: float) -> bool:
    op = op.strip().lower()
    if op in (">", "gt", "above", "over", "more than", "greater than"):
        return actual > threshold
    if op in ("<", "lt", "below", "under", "less than"):
        return actual < threshold
    if op in (">=", "ge", "at least", "min", "minimum", "no less than"):
        return actual >= threshold
    if op in ("<=", "le", "at most", "max", "maximum", "no more than"):
        return actual <= threshold
    if op in ("=", "==", "eq", "equal", "equals", "around", "about"):
        return abs(actual - threshold) <= max(threshold * 0.05, 1)
    return False


# ── Column selection helpers ──────────────────────────────────────────────────

def _cols_for_hint(hint: str, columns: list[str], fallback_kw: set[str]) -> list[str]:
    """
    Return columns most likely to hold data described by `hint`.
    Priority: exact word overlap → fallback keyword set → all columns.
    """
    hint_words = set(re.split(r"[\s_\-]+", hint.lower()))
    # Remove very generic words
    hint_words -= {"the", "a", "an", "of", "in", "for", "and", "or"}

    exact: list[str] = []
    fuzzy: list[str] = []
    for col in columns:
        col_words = set(re.split(r"[\s_\-]+", col.lower()))
        if hint_words & col_words:
            exact.append(col)
        elif any(kw in col.lower() for kw in fallback_kw):
            fuzzy.append(col)

    return exact or fuzzy or columns


# ── LLM constraint extraction ─────────────────────────────────────────────────

_CONSTRAINT_SYSTEM = """\
You extract structured search constraints from a user query.
Constraints are things the results MUST satisfy.

Respond with ONLY a valid JSON object — no markdown, no explanation:
{
  "location": "<place the entity must be based in — city, region or country — or null>",
  "numeric": [
    {
      "field_hint": "<what the number describes, e.g. funding, revenue, employees, year, rating, price>",
      "op": "<one of: >, <, >=, <=, =>",
      "threshold": <number in full, no units, no $>
    }
  ],
  "categorical": [
    {
      "field_hint": "<which aspect — e.g. type, cuisine, industry, domain, technology>",
      "keywords": ["<primary keyword>", "<synonym 1>", "<synonym 2>"]
    }
  ]
}

Rules:
- "location": keep as stated ("Brooklyn", "US", "Europe", "India"). null if not mentioned.
- "numeric": use for ALL numeric comparisons, including years ("2024" → year >= 2024),
  prices, ratings, counts, and funding amounts.
- "categorical": use for required categories, tags, technologies, or descriptors.
  Always include common synonyms / alternate spellings in "keywords".
- Extract ONLY mandatory constraints. Ignore vague words like "top", "best", "good"
  unless they imply a concrete threshold.

Examples:
  "search engine startups in the US with funding > 10M"
  → {"location": "US",
     "numeric": [{"field_hint": "funding", "op": ">", "threshold": 10000000}],
     "categorical": []}

  "pizza places in Brooklyn"
  → {"location": "Brooklyn, NY",
     "numeric": [],
     "categorical": [{"field_hint": "cuisine", "keywords": ["pizza", "pizzeria", "italian"]}]}

  "open source database tools"
  → {"location": null,
     "numeric": [],
     "categorical": [{"field_hint": "type", "keywords": ["open source", "open-source", "oss"]}]}

  "no-code app building tools"
  → {"location": null,
     "numeric": [],
     "categorical": [{"field_hint": "type", "keywords": ["no-code", "no code", "low-code", "nocode"]}]}

  "AI startups in healthcare"
  → {"location": null,
     "numeric": [],
     "categorical": [
       {"field_hint": "industry", "keywords": ["healthcare", "health", "medical", "clinical", "biotech"]},
       {"field_hint": "technology", "keywords": ["AI", "artificial intelligence", "machine learning", "ML"]}
     ]}

  "large language model providers"
  → {"location": null,
     "numeric": [],
     "categorical": [{"field_hint": "domain",
                       "keywords": ["large language model", "LLM", "language model",
                                    "foundation model", "generative AI"]}]}

  "autonomous vehicle companies 2024"
  → {"location": null,
     "numeric": [{"field_hint": "founded_year", "op": ">=", "threshold": 2020}],
     "categorical": [{"field_hint": "domain",
                       "keywords": ["autonomous vehicle", "self-driving", "AV",
                                    "autonomous driving", "robotaxi"]}]}

  "restaurants with rating above 4.5 in Chicago"
  → {"location": "Chicago",
     "numeric": [{"field_hint": "rating", "op": ">", "threshold": 4.5}],
     "categorical": []}

  "B2B SaaS companies in Europe with ARR > 5M"
  → {"location": "Europe",
     "numeric": [{"field_hint": "arr", "op": ">", "threshold": 5000000}],
     "categorical": [
       {"field_hint": "model", "keywords": ["B2B", "business to business"]},
       {"field_hint": "type",  "keywords": ["SaaS", "software as a service"]}
     ]}
"""


async def _extract_constraints_llm(client: AsyncCerebras, query: str) -> dict:
    """Call the LLM to parse the query into structured constraints."""
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _CONSTRAINT_SYSTEM},
                {"role": "user", "content": query},
            ],
            max_tokens=512,
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
        parsed = extract_json_obj(text)
        if parsed and isinstance(parsed, dict):
            return parsed
    except Exception as exc:
        logger.warning("LLM constraint extraction failed: %s", exc)
    return {}


# ── Per-constraint scorers ────────────────────────────────────────────────────

def _location_satisfied(entity: Entity, columns: list[str], loc_key: str) -> bool:
    """
    True if the entity is in the specified location.

    Matching strategy (in order):
      1. Canonical identifier match — handles country aliases and abbreviations
      2. US state detection — full state names and comma-separated abbreviations
      3. Phrase substring match — for any city or region not in the canon table
    """
    loc_lower = loc_key.lower().strip()
    canon = _LOCATION_CANON.get(loc_lower)
    is_us_query = loc_lower in _LOCATION_CANON and "united states" in _LOCATION_CANON.get(loc_lower, set())

    geo_cols = [c for c in columns if any(kw in c.lower() for kw in _GEO_COLUMN_KW)] or columns

    for col in geo_cols:
        cell = entity.cells.get(col)
        if not cell or cell.value is None:
            continue
        text = str(cell.value).lower()

        # 1. Canonical identifier match
        if canon and any(ident in text for ident in canon):
            return True

        # 2. US state names / abbreviations (implicit US indicator)
        if is_us_query:
            if any(state in text for state in _US_STATES_FULL):
                return True
            parts = [p.strip().rstrip(".") for p in re.split(r"[,/]", text)]
            if any(p in _US_STATES_ABBR for p in parts):
                return True

        # 3. Generic phrase match (covers cities, neighbourhoods, sub-regions)
        if loc_lower in text:
            return True

    return False


def _numeric_satisfied(entity: Entity, columns: list[str], nc: dict) -> bool:
    """
    True if any relevant column satisfies the numeric constraint.
    Handles money strings ($10M), plain numbers, and year values.
    """
    field_hint = nc.get("field_hint", "")
    op = nc.get("op", ">")
    threshold = float(nc.get("threshold", 0))

    target_cols = _cols_for_hint(field_hint, columns, _NUMERIC_COLUMN_KW)

    for col in target_cols:
        cell = entity.cells.get(col)
        if not cell or cell.value is None:
            continue
        actual = _cell_to_number(cell.value)
        if actual is not None and _op_passes(actual, op, threshold):
            return True
    return False


def _categorical_satisfied(entity: Entity, columns: list[str], cc: dict) -> bool:
    """
    True if at least one keyword from `cc["keywords"]` appears in any
    text value of the relevant columns (case-insensitive substring match).
    Falls back to searching ALL columns when no specific column is found.
    """
    field_hint = cc.get("field_hint", "")
    keywords: list[str] = [kw.lower() for kw in cc.get("keywords", []) if kw]

    if not keywords:
        return False

    target_cols = _cols_for_hint(field_hint, columns, _TEXT_COLUMN_KW)

    for col in target_cols:
        cell = entity.cells.get(col)
        if not cell or cell.value is None:
            continue
        text = str(cell.value).lower()
        if any(kw in text for kw in keywords):
            return True

    return False


# ── Entity scoring ────────────────────────────────────────────────────────────

def _score_entity(entity: Entity, constraints: dict, columns: list[str]) -> int:
    """
    Count how many constraints this entity satisfies.
    Each satisfied constraint contributes +1 to the sort key.
    """
    n = 0

    location = constraints.get("location")
    if location:
        if _location_satisfied(entity, columns, location):
            n += 1

    for nc in constraints.get("numeric", []):
        if _numeric_satisfied(entity, columns, nc):
            n += 1

    for cc in constraints.get("categorical", []):
        if _categorical_satisfied(entity, columns, cc):
            n += 1

    return n


# ── Public API ────────────────────────────────────────────────────────────────

def rank_entities(
    entities: list[Entity],
    constraints: dict,
    columns: list[str],
) -> list[Entity]:
    """
    Re-rank entities by constraint satisfaction (descending), then by
    data coverage (descending) within the same tier.

    Entities satisfying ALL constraints are guaranteed to appear first.
    """
    has_location = bool(constraints.get("location"))
    has_numeric = bool(constraints.get("numeric"))
    has_categorical = bool(constraints.get("categorical"))

    if not (has_location or has_numeric or has_categorical):
        # Nothing to rank on — preserve current order
        return entities

    scored: list[tuple[Entity, int, float]] = []
    for entity in entities:
        n_sat = _score_entity(entity, constraints, columns)
        cov = entity.coverage(columns)
        scored.append((entity, n_sat, cov))

    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)

    if logger.isEnabledFor(logging.INFO):
        for entity, n_sat, cov in scored:
            logger.info(
                "  [rank] %-40s  satisfied=%d  coverage=%.2f",
                entity.get_name(), n_sat, cov,
            )

    return [x[0] for x in scored]


async def extract_and_rank(
    client: AsyncCerebras,
    entities: list[Entity],
    columns: list[str],
    query: str,
) -> list[Entity]:
    """
    Main entry point called by the pipeline.

    1. LLM parses the query into location / numeric / categorical constraints.
    2. Each entity is scored (how many constraints it satisfies).
    3. Entities are re-ordered: more constraints met → higher position, mandatory.
    """
    if not entities:
        return entities

    constraints = await _extract_constraints_llm(client, query)
    logger.info("Ranking constraints extracted: %s", constraints)

    return rank_entities(entities, constraints, columns)
