from __future__ import annotations
import logging, re
from datetime import date

from openai import AsyncOpenAI
from ..config import LLM_REASONING, PLANNER_MODEL, estimate_tokens, pacer
from ..models import SearchPlan
from ..utils import extract_json_obj

logger = logging.getLogger(__name__)

MAX_COLUMNS = 8
OUTPUT_TOKENS = 1024
NUM_QUERIES = 3
VALID_OPS = {">", ">=", "<", "<="}
LOOKUP_COLUMNS = {
    "phone", "phone_number", "telephone", "address", "street_address", "email",
    "contact", "hours", "opening_hours", "business_hours", "fax",
}

SYSTEM_PLANNER_PROMPT = """You plan a web search that will build a comparison table.

Given a topic query you must:

1. Name the entity type in a few words (e.g. "AI healthcare startups", "pizza restaurants").

2. Choose 7 or 8 columns. "name" is always the FIRST column.
   Pick the attributes that articles, rankings and news about THIS topic use to
   compare these entities. Ask yourself: "what does every 'best X' or 'top X'
   article say about each one?"

   Good columns are:
   - a category or focus column with short values
     (e.g. "pizza_style": "Neapolitan", "focus_area": "Clinical documentation")
   - the topic's key measure or offering
     (e.g. "autonomy_level", "key_product", "funding", "license")
   - where it operates, if that matters for this topic
     (e.g. "neighborhood", "primary_market")
   - ONE short "highlights" column: why it stands out right now
     (awards, recent milestones, what reviewers praise)

   Examples:
     "best pizza places in San Francisco"
       -> name, neighborhood, pizza_style, signature_pie, price_range, highlights
     "autonomous vehicle companies"
       -> name, av_focus, autonomy_level, primary_market, commercial_status, highlights
     "AI startups in healthcare"
       -> name, focus_area, key_product, funding, headquarters, highlights
     "open source database tools"
       -> name, database_type, license, best_for, main_language, highlights

   Avoid:
   - contact or lookup details anyone can find on a map: phone, address,
     opening hours, email
   - generic profile fields unless the query asks for them
     (founded_year only if the query mentions a year or age)
   - facts that are rarely public for this kind of entity
     (revenue, profit, exact employee count, user counts)

   If the query names a filter (a city, a year, a price), include a column
   that holds it. Use short snake_case names.
   The examples show the style only. Choose columns that fit THIS topic.

3. Write EXACTLY 3 different search queries that find list, ranking or review
   pages naming many of these entities. Put the best query first.
   Only add a year to a query if it is the current year given by the user.

4. Turn the real filters in the query into ranking rules. Use your EXACT column names.
   - "location": {"column": "...", "names": ["place", "nickname", ...]} or null
   - "numeric": [{"column": "...", "op": ">", ">=", "<" or "<=", "value": number}]
   - "categorical": [{"column": "...", "keywords": ["word", "synonym", ...]}]
   Ignore words that are not filters, like "best", "top" or "popular".
   Examples:
     "cheap pizza places in Brooklyn"
       -> {"location": {"column": "neighborhood", "names": ["Brooklyn", "BK"]},
           "numeric": [{"column": "price_range", "op": "<=", "value": 2}],
           "categorical": [{"column": "pizza_style", "keywords": ["pizza", "pizzeria"]}]}
     "AI startups in healthcare founded after 2020"
       -> {"location": null,
           "numeric": [{"column": "founded_year", "op": ">", "value": 2020}],
           "categorical": [{"column": "focus_area", "keywords": ["health", "healthcare", "medical", "clinical"]}]}

Respond with ONLY a JSON object:
{
  "entity_type": "short description",
  "columns": ["name", "...", "highlights"],
  "search_queries": ["query1", "query2", "query3"],
  "rules": {"location": null, "numeric": [], "categorical": []},
  "rationale": "one sentence"
}"""


def _snake(column: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_")


def _clean_columns(columns: list) -> list[str]:
    cleaned: list[str] = ["name"]
    for column in columns:
        col = _snake(column)
        if col and col not in cleaned and col not in LOOKUP_COLUMNS:
            cleaned.append(col)
    return cleaned[:MAX_COLUMNS]


def _clean_queries(queries: list, query: str) -> list[str]:
    cleaned: list[str] = []
    for q in list(queries) + [query, f"top {query}", f"best {query}"]:
        q = str(q).strip()
        if q and q.lower() not in {c.lower() for c in cleaned}:
            cleaned.append(q)
    return cleaned[:NUM_QUERIES]


def _words(values) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(v).strip() for v in values if str(v).strip()]


def _clean_rules(rules, columns: list[str]) -> dict:
    clean = {"location": None, "numeric": [], "categorical": []}
    if not isinstance(rules, dict):
        return clean

    location = rules.get("location")
    if isinstance(location, dict) and _snake(location.get("column", "")) in columns and _words(location.get("names")):
        clean["location"] = {"column": _snake(location["column"]), "names": _words(location["names"])}

    for rule in rules.get("numeric") or []:
        if not isinstance(rule, dict) or _snake(rule.get("column", "")) not in columns:
            continue
        if rule.get("op") not in VALID_OPS:
            continue
        try:
            value = float(rule.get("value"))
        except (TypeError, ValueError):
            continue
        clean["numeric"].append({"column": _snake(rule["column"]), "op": rule["op"], "value": value})

    for rule in rules.get("categorical") or []:
        if not isinstance(rule, dict) or _snake(rule.get("column", "")) not in columns:
            continue
        if _words(rule.get("keywords")):
            clean["categorical"].append({"column": _snake(rule["column"]), "keywords": _words(rule["keywords"])})

    return clean


def _fallback_plan(query: str) -> SearchPlan:
    return SearchPlan(
        entity_type=query,
        columns=["name", "category", "key_offering", "location", "highlights"],
        search_queries=_clean_queries([], query),
        rationale="Fallback plan.",
    )


async def plan_search(client: AsyncOpenAI, query: str) -> SearchPlan:
    user = f"Current year: {date.today().year}\nPlan a structured search for: {query}"
    try:
        event = await pacer.reserve(estimate_tokens(SYSTEM_PLANNER_PROMPT + user) + OUTPUT_TOKENS)
        response = await client.chat.completions.create(
            model=PLANNER_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PLANNER_PROMPT},
                {"role": "user", "content": user},
            ],
            max_tokens=OUTPUT_TOKENS,
            reasoning_effort=LLM_REASONING,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        pacer.settle(event, getattr(response, "usage", None), "plan")
        data = extract_json_obj((response.choices[0].message.content or "").strip())
        if data is None:
            raise ValueError("No JSON object found in planner response")
        columns = _clean_columns(data.get("columns", []))
        if len(columns) < 3:
            raise ValueError(f"Too few usable columns: {columns}")
        return SearchPlan(
            entity_type=str(data.get("entity_type") or query),
            columns=columns,
            search_queries=_clean_queries(data.get("search_queries", []), query),
            rules=_clean_rules(data.get("rules"), columns),
            rationale=str(data.get("rationale", "")),
        )
    except Exception as exc:
        logger.warning("Planner failed (%s), using fallback", exc)
        return _fallback_plan(query)
