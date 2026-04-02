# Stage 1 - Planner
# Plans a structured search based on the initial query, defining entity type, columns, and search queries.

from __future__ import annotations
import logging

from cerebras.cloud.sdk import AsyncCerebras
from ..models import SearchPlan
from .utils import extract_json_obj

logger = logging.getLogger(__name__)

SYSTEM_PLANNER_PROMPT = """You are an expert at planning structured web research.

Given a topic query you must:
1. Identify the entity type (e.g. "AI healthcare startups", "pizza restaurants")
2. Choose EXACTLY 5 columns that best describe these entities (no more, no fewer).
   Rules:
   - "name" must always be the FIRST column
   - Include a short "description" column second
   - Add 3 domain-specific attributes (for companies pick 3 from: founded_year, headquarters, funding_stage, website; for restaurants pick 3 from: cuisine, price_range, rating, address)
3. Generate EXACTLY 3 DIVERSE search queries covering different angles

Respond with ONLY a JSON object - no markdown, no extra text as shown in this example:

Example: {
  "entity_type": "short description",
  "columns": ["name", "description", ...],
  "search_queries": ["query1", "query2", ...],
  "rationale": "one sentence"
}"""


async def plan_search(client: AsyncCerebras, query: str) -> SearchPlan:
    response = await client.chat.completions.create(
        model="qwen-3-235b-a22b-instruct-2507",
        messages=[
            {"role": "system", "content": SYSTEM_PLANNER_PROMPT},
            {"role": "user", "content": f"Plan a structured search for: {query}"},
        ],
        max_tokens=1024,
        temperature=0.3,
    )

    text = response.choices[0].message.content.strip()

    try:
        data = extract_json_obj(text)
        if data is None:
            raise ValueError("No JSON object found in planner response")
        plan = SearchPlan(**data)
        if "name" not in plan.columns:
            plan.columns.insert(0, "name")
        elif plan.columns[0] != "name":
            plan.columns.remove("name")
            plan.columns.insert(0, "name")
        return plan
    except Exception as exc:
        logger.warning("Planner parse failed (%s), using fallback", exc)
        return SearchPlan(
            entity_type=query,
            columns=["name", "description", "website", "details"],
            search_queries=[query, f"top {query}", f"best {query} 2024"],
            rationale="Fallback plan.",
        )
