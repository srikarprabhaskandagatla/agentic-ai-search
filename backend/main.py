# Srikar Prabhas Kandagatla

# Import Libraries and Modules
from __future__ import annotations
import asyncio, json, logging, os
from pathlib import Path
from contextlib import asynccontextmanager

from openai import AsyncOpenAI

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config import EXTRACT_INPUT_TOKENS, LLM_API_KEY, LLM_BASE_URL, LLM_MAX_RETRIES, LLM_PROVIDER, PLANNER_MODEL
from backend.models import EntityTable
from backend import embedder
from backend.pipeline import filler, resolver
from backend.pipeline.planner import plan_search
from backend.pipeline.retriever import retrieve, retrieve_with_backup
from backend.pipeline.extractor import extract_from_chunks
from backend.pipeline.ranker import rank

# Logger Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)

MAX_FILLS = 3
MIN_ENTITIES = 8

llm_client: AsyncOpenAI | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_client
    llm_client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, max_retries=LLM_MAX_RETRIES)
    logger.info("LLM client configured for %s", LLM_PROVIDER)
    try:
        await asyncio.to_thread(embedder.load_model)
    except Exception as exc:
        logger.warning("Embedding model failed to load: %s", exc)
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
            all_pages = []
            all_queries_used: list[str] = []

            # 1. Plan
            await send_progress("planning", "analysing query and inferring schema", 0.04)
            plan = await plan_search(llm_client, request.query)
            await send_progress("planning",
                       f"Schema: {len(plan.columns)} columns for '{plan.entity_type}'",
                       0.10, f"Columns: {', '.join(plan.columns)}")

            context = f"{request.query} {plan.entity_type}"
            agentic = request.search_depth > 1

            def keep_pages(pages):
                seen_urls.update(page.url for page in pages)
                all_sources.extend(page.url for page in pages)
                all_pages.extend(pages)

            try:
                chunk_index = embedder.ChunkIndex([], [])
            except Exception as exc:
                logger.warning("Chunk index unavailable: %s", exc)
                chunk_index = None

            # 2. Search
            await send_progress("searching", "Searching the web", 0.12)
            pages, used = await retrieve_with_backup(plan.search_queries, seen_urls)
            all_queries_used.extend(used)
            keep_pages(pages)
            await send_progress("searching", f"Read {len(pages)} pages", 0.25)

            async def read_pages(pages):
                nonlocal all_entities
                await send_progress("extracting", f"Reading {len(pages)} pages", 0.28)
                try:
                    chunks, vectors = await asyncio.to_thread(embedder.embed_pages, pages)
                    if chunk_index is not None:
                        chunk_index.add(chunks, vectors)
                except Exception as exc:
                    logger.warning("Embedding failed: %s", exc)
                    chunks, vectors = embedder.chunk_pages(pages), None
                topic = f"{plan.entity_type}: {', '.join(plan.columns)}"
                best = await asyncio.to_thread(embedder.top_chunks, chunks, vectors, topic, EXTRACT_INPUT_TOKENS)

                # 3. Extract
                await send_progress("extracting", f"Extracting entities from {len(best)} best chunks", 0.32)
                all_entities.extend(await extract_from_chunks(llm_client, best, plan.columns, plan.entity_type))

                # 4. Resolve
                await send_progress("resolving", "Merging duplicate entities", 0.45)
                all_entities = resolver.resolve_entities(all_entities, all_pages, context)
                await send_progress("resolving", f"{len(all_entities)} unique entities", 0.50)

            if pages:
                await read_pages(pages)

            unused = [q for q in plan.search_queries if q not in used]

            # When is Selected Agentic
            if agentic and len(all_entities) < MIN_ENTITIES and unused:
                await send_progress("searching", f"Only {len(all_entities)} entities, searching more", 0.52)
                try:
                    more = await retrieve(unused, seen_urls)
                except Exception as exc:
                    logger.warning("Backup search failed: %s", exc)
                    more = []
                all_queries_used.extend(unused)
                keep_pages(more)
                if more:
                    await read_pages(more)

            fill_step = 0

            async def fill_progress(message):
                nonlocal fill_step
                fill_step += 1
                await send_progress("filling", message, min(0.60 + 0.05 * fill_step, 0.92))

            # 5. Fill
            await send_progress("filling", "Filling empty cells from page text", 0.58)
            fill = await filler.fill_loop(
                llm_client, all_entities, plan.columns, plan.entity_type,
                chunk_index, seen_urls, MAX_FILLS if agentic else 1, fill_progress,
            )
            all_entities = fill.entities
            all_queries_used.extend(fill.queries)
            all_sources.extend(fill.sources)
            completed_rounds = max(fill.rounds, 1)

            # 6. Rank
            await send_progress("ranking", "Ranking rows and picking full columns", 0.98)
            model = chunk_index.model if chunk_index is not None else None
            all_entities, shown_columns, ranking_info = await asyncio.to_thread(
                rank, all_entities, plan.rules, plan.columns, request.query, model
            )

            await send_progress("done",
                       f"Done! {len(all_entities)} entities - "
                       f"{len(all_sources)} sources - "
                       f"{completed_rounds} round(s)",
                       1.0)

            result = EntityTable(
                query=request.query,
                entity_type=plan.entity_type,
                columns=shown_columns,
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


app.mount("/", StaticFiles(directory=Path(__file__).resolve().parent.parent / "frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0",
                port=int(os.getenv("PORT", 8000)), reload=True)