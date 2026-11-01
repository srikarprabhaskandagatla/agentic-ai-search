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
MAX_PLACE_NAMES = 10
VALID_OPS = {">", ">=", "<", "<="}
LOOKUP_COLUMNS = {
    "phone", "phone_number", "telephone", "address", "street_address", "email",
    "contact", "hours", "opening_hours", "business_hours", "fax",
}

SYSTEM_PLANNER_PROMPT = """You plan a web search that builds a comparison table.

1. entity_type, a few words, like "pizza restaurants".

2. columns, 6 to 8 short snake_case names with "name" first and "highlights" last.
   Pick facts that "best X" lists and reviews state for almost every entry, with short values.
   Use a category or style column, the key measure or offering, and a location column if place matters.
   If the query has a filter (a place, a price, a year), add a column that holds it.
   Never use phone, address, hours, email, revenue, profit or employee count.

3. search_queries, exactly 3 different queries that find list, ranking or review pages naming
   many entities. Best first. Add a year only if it is the current year.

4. rules, ONLY for filters written in the query. "best", "top" and "popular" are not filters,
   and the entity type itself is not a filter. Never invent a style, type or price.
   location, null or {"column", "names"}. names starts with the place exactly as written in
   the query, then up to 8 well known neighborhoods or cities inside it.
   numeric, a limit like "cheap" or "after 2020" as {"column", "op", "value"}, op is >, >=, < or <=,
   value is a number. Price ranges use 1 to 4 for $ to $$$$.
   categorical, a kind the query asks for, as {"column", "keywords"}. keywords hold that word
   from the query plus synonyms.

Example query "cheap vegan pizza in Brooklyn"
{"entity_type": "pizza restaurants",
 "columns": ["name", "neighborhood", "pizza_style", "signature_pie", "price_range", "vegan_options", "highlights"],
 "search_queries": ["best vegan pizza Brooklyn", "Brooklyn vegan pizza places ranked", "cheap vegan pizza Brooklyn list"],
 "rules": {"location": {"column": "neighborhood", "names": ["Brooklyn", "Williamsburg", "Bushwick", "Park Slope", "DUMBO"]},
           "numeric": [{"column": "price_range", "op": "<=", "value": 2}],
           "categorical": [{"column": "vegan_options", "keywords": ["vegan", "plant based", "dairy free"]}]},
 "rationale": "one sentence"}

Respond with ONLY a JSON object with these keys."""


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


def _in_query(word: str, query: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(word.lower())}(?!\w)", query.lower()) is not None


def _clean_rules(rules, columns: list[str], query: str) -> dict:
    clean = {"location": None, "numeric": [], "categorical": []}
    if not isinstance(rules, dict):
        return clean

    location = rules.get("location")
    if isinstance(location, dict) and _snake(location.get("column", "")) in columns:
        names = _words(location.get("names"))
        named = [n for n in names if _in_query(n, query)]
        if named:
            ordered = named + [n for n in names if n not in named]
            clean["location"] = {"column": _snake(location["column"]), "names": ordered[:MAX_PLACE_NAMES]}

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
        keywords = _words(rule.get("keywords"))
        if any(_in_query(k, query) for k in keywords):
            clean["categorical"].append({"column": _snake(rule["column"]), "keywords": keywords})

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
            rules=_clean_rules(data.get("rules"), columns, query),
            rationale=str(data.get("rationale", "")),
        )
    except Exception as exc:
        logger.warning("Planner failed (%s), using fallback", exc)
        return _fallback_plan(query)
