# Srikar Prabhas Kandagatla

# Import Libraries and Modules
from __future__ import annotations
import asyncio, json, logging, os
from contextlib import asynccontextmanager

from openai import AsyncOpenAI

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.config import LLM_API_KEY, LLM_BASE_URL, LLM_MAX_RETRIES, LLM_PROVIDER, PLANNER_MODEL
from backend.models import EntityTable
from backend.pipeline import resolver, gap_analyzer
from backend.pipeline.planner import plan_search
from backend.pipeline.retriever import retrieve, retrieve_with_backup
from backend.pipeline.extractor import extract_from_pages
from backend.pipeline.llm_filler import llm_fill_gaps
from backend.pipeline.ranker import extract_and_rank

# Logger Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)

llm_client: AsyncOpenAI | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_client
    llm_client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, max_retries=LLM_MAX_RETRIES)
    logger.info("LLM client configured for %s", LLM_PROVIDER)
    yield


app = FastAPI(title="CiteTable API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=300)
    search_depth: int = Field(default=2, ge=1, le=3)


@app.get("/health")
async def health():
    return {"status": "ok", "version": app.version, "llm": f"{LLM_PROVIDER}/{PLANNER_MODEL}", "search": "tavily"}


@app.get("/example-queries")
async def example_queries():
    return {"examples": [
        "AI startups in healthcare",
        "top pizza places in Brooklyn",
        "open source database tools",
        "autonomous vehicle companies 2026",
        "no-code app building tools",
        "large language model providers",
    ]}


@app.post("/search")
async def search_endpoint(request: SearchRequest):
    if llm_client is None or not LLM_API_KEY:
        raise HTTPException(status_code=503, detail="LLM client not configured")

    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    async def send_progress(stage, message, progress, detail=None):
        await queue.put(
            {
                "type": "progress",
                "stage": stage,
                "message": message,
                "progress": min(progress, 1.0),
                "detail": detail
            })

    async def run():
        try:
            all_entities = []
            seen_urls: set[str] = set()
            all_sources: list[str] = []
            all_queries_used: list[str] = []

            # 1. Plan
            await send_progress("planning", "analysing query and inferring schema", 0.04)
            plan = await plan_search(llm_client, request.query)
            await send_progress("planning",
                       f"Schema: {len(plan.columns)} columns for '{plan.entity_type}'",
                       0.10, f"Columns: {', '.join(plan.columns)}")

            present_query = plan.search_queries
            completed_rounds = 0

            for round_idx in range(request.search_depth):
                completed_rounds = round_idx + 1
                rl = f"Round {round_idx+1}/{request.search_depth}"
                rs = 0.10 + round_idx * (0.85 / request.search_depth)
                rspan = 0.85 / request.search_depth
                rp = lambda f, _rs=rs, _rspan=rspan: _rs + f * _rspan

                await send_progress("searching", f"{rl}: Searching the web", rp(0.05))
                if round_idx == 0:
                    pages, present_query = await retrieve_with_backup(present_query, seen_urls)
                else:
                    pages = await retrieve(present_query, seen_urls)
                all_queries_used.extend(present_query)
                seen_urls.update(page.url for page in pages)
                all_sources.extend(page.url for page in pages)

                await send_progress("searching", f"{rl}: Read {len(pages)} new pages", rp(0.40))
                if not pages:
                    break

                # 4. Extract
                await send_progress("extracting", f"{rl}: Extracting entities", rp(0.45))
                new_raw = await extract_from_pages(llm_client, pages, plan.columns, plan.entity_type)
                await send_progress("extracting", f"{rl}: Found {len(new_raw)} entity mentions", rp(0.65))
                all_entities.extend(new_raw)

                # 5. Resolve
                await send_progress("resolving", f"{rl}: Deduplicating and merging", rp(0.70))
                all_entities = await resolver.resolve_entities(llm_client, all_entities)
                await send_progress("resolving", f"{rl}: {len(all_entities)} unique entities", rp(0.80))

                # 6. Gap Analysis
                if round_idx < request.search_depth - 1:
                    await send_progress("analyzing", "Checking data coverage", rp(0.88))
                    gap = await gap_analyzer.analyze_gaps(
                        llm_client, all_entities, plan.columns,
                        request.query, plan.entity_type
                    )
                    if gap.get("should_continue") and gap.get("gap_queries"):
                        present_query = gap["gap_queries"]
                        await send_progress("analyzing",
                                   f"Gaps found - running follow-up: {gap.get('gap_summary', '')}",
                                   rp(0.92))
                    else:
                        await send_progress("analyzing", "Coverage sufficient - stopping early", rp(0.92))
                        break

            # 7. LLM gap-fill - last-resort pass for still-missing values
            await send_progress("filling", "Filling remaining gaps from LLM knowledge", 0.95)
            all_entities = await llm_fill_gaps(
                llm_client, all_entities, plan.columns, plan.entity_type
            )

            # 8. Constraint-based ranking - mandatory: constraint-satisfying
            #    entities always surface to the top regardless of order.
            await send_progress("ranking", "Ranking results by query constraints", 0.98)
            all_entities, ranking_info = await extract_and_rank(
                llm_client, all_entities, plan.columns, request.query
            )

            await send_progress("done",
                       f"Done! {len(all_entities)} entities - "
                       f"{len(all_sources)} sources - "
                       f"{completed_rounds} round(s)",
                       1.0)

            result = EntityTable(
                query=request.query,
                entity_type=plan.entity_type,
                columns=plan.columns,
                entities=all_entities,
                sources_consulted=all_sources,
                search_queries_used=all_queries_used,
                rounds_completed=completed_rounds,
                ranking=ranking_info,
            )
            await queue.put({"type": "result", "data": result.model_dump(mode="json")})

        except Exception as exc:
            logger.exception("Pipeline error for query: %s", request.query)
            await queue.put({"type": "error", "message": str(exc)})
        finally:
            await queue.put(None)

    async def event_stream():
        task = asyncio.create_task(run())
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=4.0)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            if item is None:
                break
            yield f"data: {json.dumps(item, default=str)}\n\n"
        await task

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0",
                port=int(os.getenv("PORT", 8000)), reload=True)