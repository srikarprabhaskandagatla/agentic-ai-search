# Stage 1 - Planner  (uses Cerebras / Llama 3.3 70B)

from __future__ import annotations
import json, re, logging

from cerebras.cloud.sdk import AsyncCerebras
from ..models import SearchPlan

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert at planning structured web research.

Given a topic query you must:
1. Identify the entity type (e.g. "AI healthcare startups", "pizza restaurants")
2. Choose 5 - 8 columns that best describe these entities.
   Rules:
   - "name" must always be the FIRST column
   - Include a short "description" column second
   - Add domain-specific attributes (for companies: founded_year, headquarters, funding_stage; for restaurants: cuisine, price_range, rating, address)
3. Generate 4 - 6 DIVERSE search queries covering different angles

Respond with ONLY a JSON object - no markdown, no extra text:
{
  "entity_type": "short description",
  "columns": ["name", "description", ...],
  "search_queries": ["query1", "query2", ...],
  "rationale": "one sentence"
}"""


async def plan_search(client: AsyncCerebras, query: str) -> SearchPlan:
    response = await client.chat.completions.create(
        model="qwen-3-235b-a22b-instruct-2507",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Plan a structured search for: {query}"},
        ],
        max_tokens=1024,
        temperature=0.3,
    )

    text = response.choices[0].message.content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
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
