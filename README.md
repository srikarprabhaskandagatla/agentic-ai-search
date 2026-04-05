# CIIR Agentic AI Search

A multi-stage, agentic web research system. You type a natural language query like *"AI startups in healthcare"* and get back a clean, structured table with sourced data - every cell linked to the exact web page excerpt that proves it.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Setup Instructions](#setup-instructions)
3. [Architecture Overview](#architecture-overview)
4. [Backend](#backend)
5. [Frontend](#frontend)
6. [Tests](#tests)
7. [Output Quality](#output-quality)
8. [Design Choices and Trade-offs](#design-choices-and-trade-offs)
9. [Known Limitations](#known-limitations)

---

## What It Does

The system takes a free-form query, figures out what kind of data to extract, searches the web, scrapes pages, extracts structured entities, deduplicates them, checks for missing data, and runs follow-up searches if needed. The result is a table of up to 10 entities, each with 5 attributes, and every value linked to the web page it came from.

---

## Setup Instructions

**Requirements:** Python 3.11+, two API keys (Cerebras Cloud, Tavily).

```bash
# 1. Clone and install dependencies
git clone <repo-url>
cd ciir-agentic-search
pip install -r requirements.txt

# 2. Configure environment variables
cp backend/.env.example backend/.env
# Edit backend/.env and fill in:
#   CEREBRAS_API_KEY=<your key>
#   TAVILY_API_KEY=<your key>
#   PORT=8000

# 3. Start the server
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 4. Open the app
# Open frontend/index.html in a browser, or serve it via any static server.
# The frontend connects to http://localhost:8000 by default.
```

**Run tests:**
```bash
pytest tests/test_pipeline.py -v
```

---

## Architecture Overview

The pipeline has 7 sequential stages. Each stage feeds into the next. Stages 2-6 form a loop - after resolving entities and checking coverage, the gap analyzer decides whether to run another round of searching, scraping, and extraction.

## Pipeline

1. **Planner** - Infers schema and generates initial search queries
2. **Searcher** - Runs queries via Tavily API
3. **Scraper** - Fetches and cleans web pages (concurrent)
4. **Extractor** - LLM extracts structured entities from pages
5. **Resolver** - Deduplicates and merges entities
6. **Gap Analyzer** - Checks coverage; if gaps remain, generates targeted queries and loops back to step 2
7. **LLM Filler** - Fills any remaining missing fields → **Final EntityTable**
---

## Backend

The backend is a FastAPI application. All I/O is async. The pipeline runs as a background task, streaming progress events to the frontend over Server-Sent Events (SSE).

### File Structure

```
backend/
├── main.py              # FastAPI app, endpoints, SSE streaming, pipeline orchestration
├── models.py            # Pydantic data models used across all stages
├── .env.example         # Template for required environment variables
└── pipeline/
    ├── planner.py       # Stage 1: Schema inference
    ├── searcher.py      # Stage 2: Tavily web search
    ├── scraper.py       # Stage 3: HTTP fetch + HTML cleaning
    ├── extractor.py     # Stage 4: LLM entity extraction
    ├── resolver.py      # Stage 5: Deduplication + merging
    ├── gap_analyzer.py  # Stage 6: Coverage analysis + agentic loop decision
    ├── llm_filler.py    # Stage 7: LLM knowledge-based gap filling
    └── utils.py         # JSON extraction helpers for LLM output parsing
```

### Data Models (`models.py`)

All pipeline stages share a common set of Pydantic models.

| Model | Purpose |
|---|---|
| `CellValue` | One cell in the table. Holds the value, source URLs, confidence (0-1), and whether it was LLM-filled. |
| `Entity` | One row in the table. A dict of column name → `CellValue`, plus a unique ID. Has a `coverage()` method. |
| `SearchPlan` | Output of Stage 1. Contains entity type, column names, search queries, and reasoning. |
| `SearchResult` | One result from Tavily: URL, title, snippet. |
| `ScrapedPage` | Cleaned text from one page: URL, title, up to 8KB of content, optional error. |
| `EntityTable` | Final output. List of entities, all sources consulted, all queries used, rounds completed. |

The `llm_filled` flag on `CellValue` is important - it tells the frontend to display LLM-filled values differently from web-sourced ones.

---

### Stage 1: Planner (`pipeline/planner.py`)

**What it does:** Reads the user's query and infers a schema - what kind of entities to extract, what columns to use, and what to search for first.

**How it works:**
- Calls the Cerebras LLM (Qwen 3 235B) with a system prompt that asks it to output JSON.
- Output is a `SearchPlan` with: `entity_type`, `columns` (always starts with "name"), `search_queries` (3 diverse angles), and `rationale`.

**Example:**
- Input: `"AI startups in healthcare"`
- Output:
  ```json
  {
    "entity_type": "AI healthcare startups",
    "columns": ["name", "description", "founded_year", "headquarters", "funding_stage"],
    "search_queries": ["top AI healthcare startups 2024", "clinical AI companies list", "health tech AI funding"],
    "rationale": "Captures key company attributes relevant to startup research"
  }
  ```

**Design problem solved:** Users don't tell us what columns they want. Instead of hardcoding schemas, the planner infers them per-query. "AI startups" gets business columns; "Olympic athletes" would get sports columns.

**Fallback:** If the LLM fails or returns malformed JSON, the planner falls back to a hardcoded default schema so the rest of the pipeline doesn't break.

---

### Stage 2: Searcher (`pipeline/searcher.py`)

**What it does:** Runs all search queries and returns a deduplicated list of URLs with titles and snippets.

**How it works:**
- Calls the Tavily API asynchronously for each query.
- Each query returns ~4 results (basic search depth).
- Deduplicates by URL across all queries.

**Design problem solved:** Running one broad search misses niche sources. Running 3 diverse queries gives wider coverage. For follow-up rounds, the gap analyzer generates entity-specific queries, so subsequent searches are targeted rather than broad.

**Trade-off:** Basic search depth (not Tavily's "advanced" mode) to conserve quota. Advanced mode costs more and returns more content per result, but the scraper fetches full pages anyway, making the extra snippet content redundant.

---

### Stage 3: Scraper (`pipeline/scraper.py`)

**What it does:** Fetches web pages and strips them down to readable text.

**How it works:**
- Fetches up to 6 pages concurrently using `httpx` (async, 12s timeout per page).
- Skips binary file extensions (`.pdf`, `.mp4`, `.zip`, etc.).
- Uses a realistic `User-Agent` header to avoid bot blocks.
- Parses with BeautifulSoup (lxml backend).
- Removes boilerplate: `<script>`, `<style>`, `<nav>`, `<header>`, `<footer>`, cookie banners, ads.
- Prefers semantic content from `<article>`, `<main>`, and main content divs.
- Truncates to 8,000 characters per page.
- Caches results in memory so the same URL isn't fetched twice in one pipeline run.

**Design problem solved:** Raw HTML is full of navigation, footers, and ads - the LLM wastes context on irrelevant content. Cleaning the HTML before passing it to the LLM reduces token usage and improves extraction accuracy. The 8KB cap further reduces cost.

**Trade-off:** Truncating at 8KB means we can miss data at the bottom of long pages. But most relevant entity mentions appear near the top (e.g., in article intros, list summaries), so the trade-off is acceptable.

---

### Stage 4: Extractor (`pipeline/extractor.py`)

**What it does:** For each scraped page, calls the LLM to extract all entity mentions as structured JSON.

**How it works:**
- Sends the page content and schema columns to Cerebras.
- Asks the LLM to return a JSON array of entities. Each entity must include: the column values, a confidence score per value, and the exact snippet from the text that supports the value.
- Runs up to 8 pages concurrently.

**Output per entity:**
```json
{
  "name": "Tempus AI",
  "description": "Clinical AI company applying genomics to oncology",
  "founded_year": "2015",
  "headquarters": "Chicago, IL",
  "confidence": {"name": 1.0, "description": 0.9, "founded_year": 0.85},
  "snippets": {"founded_year": "Founded in 2015 by Eric Lefkofsky..."}
}
```

**Validation:**
- Rejects entities without a name.
- Rejects implausibly long names (more than 60 characters or 6+ words) - these are usually paragraphs the LLM misidentified as names.
- Stores snippets as part of the `CellValue` source reference, so the frontend can show proof.

**Design problem solved:** LLMs hallucinate. Requiring a supporting snippet forces the LLM to ground each value in the actual text. If there's no snippet, there's no value. This is the primary anti-hallucination mechanism.

---

### Stage 5: Resolver (`pipeline/resolver.py`)

**What it does:** Takes all entities extracted across all pages (often hundreds of mentions of the same companies) and produces one clean, merged list.

**How it works in two steps:**

**Step 1 - Fast dedup (string-based):**
- Normalizes names: lowercase, strip accents, remove punctuation, strip common suffixes (`Inc`, `LLC`, `Ltd`, `Corp`, `Co`).
- Groups entities that share a normalized name.
- Example: "OpenAI" and "OpenAI Inc." and "openai" all normalize to "openai" → same group.

**Step 2 - LLM dedup (if >5 groups remain):**
- Sends the list of normalized names to the LLM.
- Asks: which names refer to the same real-world entity?
- Returns groupings that the fast dedup missed (e.g., "Google DeepMind" and "DeepMind" being the same company in context).

**Merging:**
- For each group, picks the best value per column: the one with the highest confidence score.
- Collects all source URLs from every member of the group.
- Sorts output alphabetically, capped at 10 entities.

**Design problem solved:** Web pages mention the same company in dozens of ways. Without deduplication, the output table would have 40 rows for the same 8 companies. Two-pass dedup handles both trivial duplicates (string normalization) and semantic duplicates (LLM reasoning).

**Trade-off:** LLM dedup adds latency and cost. That's why it's only triggered when there are more than 5 groups - for small result sets, string normalization is usually sufficient.

---

### Stage 6: Gap Analyzer (`pipeline/gap_analyzer.py`)

**What it does:** This is the "agentic" component. It decides whether the current data is good enough or whether more searching is needed.

**How it works:**
- Computes per-column coverage: the fraction of entities that have a value for each column.
- **Early stop condition:** If average coverage >70% AND we have ≥8 entities, stops searching.
- Otherwise, identifies which columns have low coverage and which entities are missing data.
- Generates targeted search queries - one per entity, combining the entity name with the missing column names.

**Example targeted query:** Instead of `"AI startups headquarters"`, it generates `"Tempus AI founded year headquarters funding stage"` - a query specifically designed to fill the gaps for one entity.

**Output:**
```json
{
  "should_continue": true,
  "gap_summary": "Low coverage on: headquarters (40%), funding_stage (30%)",
  "gap_queries": ["Tempus AI founded year headquarters", "Flatiron Health funding stage CEO"],
  "coverage": {"name": 1.0, "description": 0.9, "founded_year": 0.6, "headquarters": 0.4, "funding_stage": 0.3}
}
```

**Design problem solved:** A static pipeline would run the same number of search rounds regardless of data quality. The gap analyzer makes the system adaptive - it runs more rounds only when the data is actually incomplete, and stops early when coverage is good.

**Trade-off:** The coverage threshold (70%) is a fixed heuristic. A lower threshold saves cost but accepts sparser tables. A higher threshold runs more rounds but produces denser data. 70% was chosen as a pragmatic middle ground.

---

### Stage 7: LLM Filler (`pipeline/llm_filler.py`)

**What it does:** For any cells still missing after all search rounds, fills them using the LLM's training knowledge.

**How it works:**
- For each entity with missing cells, sends the entity's known data (name + found attributes) to the LLM.
- Asks it to fill in the missing columns from what it knows.
- Marks every filled value with `llm_filled=True` and `confidence=0.5`.
- Does NOT attach web sources to these values.

**Design problem solved:** Some entities are obscure or poorly documented online. Without a filler, they'd have many empty cells in the final table. The filler is a last resort that keeps the output useful, while the `llm_filled` flag makes it transparent that these values are not web-sourced.

**Trade-off:** LLM training data has a knowledge cutoff and can be wrong about specific facts (e.g., current CEO, funding amount). The lower confidence score and visual distinction in the UI communicate this uncertainty to the user.

---

### Utilities (`pipeline/utils.py`)

**What it does:** Extracts valid JSON from LLM output.

**Why this is needed:** LLMs don't always return clean JSON. They often wrap it in markdown fences (` ```json ... ``` `), include chain-of-thought in `<think>` blocks, or append explanation text after the JSON. The utilities strip all of that before parsing.

Three functions:
- `_strip_noise(text)` - removes `<think>` blocks and markdown code fences.
- `_balanced_extract(text, open, close)` - finds the first properly balanced bracket pair, respecting string contents.
- `extract_json_obj(text)` - extracts the first JSON object.
- `extract_json_arr(text)` - extracts the first JSON array.

---

### API Endpoints (`main.py`)

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Returns status, version, LLM model, search provider. Used by frontend on load. |
| `/api/example-queries` | GET | Returns a list of example queries shown as clickable pills in the UI. |
| `/api/search` | POST | Main endpoint. Accepts `{ query, search_depth }`. Streams SSE progress events, then a final result. |

**SSE Stream Format:**

Progress events (sent during pipeline execution):
```json
{ "type": "progress", "stage": "planning", "message": "Plan: 5 columns for 'AI startups'", "progress": 0.1 }
```

Final result event:
```json
{ "type": "result", "data": { ...EntityTable... } }
```

Error event:
```json
{ "type": "error", "message": "Description of what went wrong" }
```

**Design problem solved:** A single long HTTP request with no feedback leaves users wondering if anything is happening. SSE lets the server push progress updates as each stage completes, so users see the pipeline working in real time.

---

## Frontend

The frontend is a single-page application written in plain HTML, CSS, and JavaScript - no framework, no build step. It connects to the FastAPI backend and renders the pipeline output as an interactive table.

### File Structure

```
frontend/
├── index.html      # Main HTML structure, imports CSS/JS
├── style.css       # All styles: layout, themes, animations, table, modals
├── script.js       # All client logic: SSE reading, table rendering, exports, modals
└── favicon/
    └── favicon.png
```

---

### Design and Visual Choices

**Two themes:** Dark mode uses a cosmic background with stars and a shooting star animation. Light mode uses a textured green background. The choice is saved in `localStorage` and restored on next visit.

**Why no framework:** The app has one screen, one data flow, and a clear sequence of events. React or Vue would add complexity without benefit. Vanilla JS keeps the frontend readable by anyone familiar with web basics, requires no build toolchain, and loads instantly.

---

### Search Interface

- Text area for the query (Enter submits, Shift+Enter adds a newline).
- Dropdown to select 1, 2, or 3 search rounds.
- Example query pills loaded from `/api/example-queries` - clicking one fills the text area.
- An abort button appears during active searches (cancels the SSE stream).

---

### Real-Time Progress Display

While the pipeline runs, the frontend shows:
- A spinner with the current stage name and status message.
- A progress bar advancing from 0% to 100% as stages complete.
- Per-stage detail text (e.g., "Found 12 new URLs", "Scraped 10/12 pages").

This is driven by the SSE `progress` events from the backend. Each stage name maps to a fixed progress value, so the bar advances smoothly even if stages complete quickly.

---

### Results Table

Once the result event arrives:
- Columns are generated dynamically from the schema inferred in Stage 1.
- Each row is one entity. Each cell shows the value plus small circular badges (①②③) representing sources.
- Clicking a badge opens a modal showing the source URL, page title, and the exact excerpt from that page that supports the value.
- Cells with long text show a truncated version with a "more" button.
- LLM-filled values (Stage 7) are displayed with a different visual style and a "Knowledge" label instead of source badges.

---

### Confidence Indicators

Each source badge uses a color to indicate confidence:
- Green dot: confidence ≥ 0.8 (high)
- Orange dot: confidence ≥ 0.5 (medium)
- Red dot: confidence < 0.5 (low)

This lets users quickly see which values are well-supported and which are uncertain.

---

### Data Export

Two export buttons appear once results are loaded:

**JSON export:** Downloads the full `EntityTable` as a `.json` file. Includes all entities, all cell values with confidence scores and sources, all queries used, and round count.

**CSV export:** Downloads a flat `.csv` file. One row per entity, one column per attribute. LLM-filled values are marked with `(LLM)` in parentheses. Source URLs are listed in a final column.

---

### Error Handling

- If the backend returns an error event, a banner appears with the message.
- If the SSE connection drops unexpectedly, an error is shown.
- All user-supplied text passed to HTML is escaped to prevent XSS.

---

## Tests

All tests live in `tests/test_pipeline.py`. They use `unittest` and `unittest.mock` - no real API calls are made in tests.

### What Is Tested

**Models:**
- `Entity.coverage()` returns correct fractions for partial, full, and empty entities.
- `CellValue` display formatting and `llm_filled` flag behavior.

**Planner:**
- Correctly parses a valid LLM JSON response into a `SearchPlan`.
- Ensures "name" is always the first column regardless of LLM output order.

**Scraper:**
- Binary file extensions (`.pdf`, `.mp4`, `.zip`, etc.) are skipped without fetching.
- HTML cleaning correctly strips `<nav>`, `<footer>`, `<script>`, and ads while preserving main content.

**Resolver:**
- Name normalization handles punctuation, accents, capitalization, and corporate suffixes (`Inc`, `LLC`, `Ltd`, `Corp`, `Co`).
- Fast dedup correctly groups entities with equivalent normalized names.
- Cell merging keeps the highest-confidence value per column and collects sources from all group members.

**Gap Analyzer:**
- Coverage computation returns correct fractions for realistic entity sets.
- Early-stop condition triggers correctly when coverage and entity count thresholds are met.

**LLM Filler:**
- Correctly fills missing cells and leaves existing cells unchanged.
- Marks filled values as `llm_filled=True` with confidence 0.5.
- Returns entities unchanged (no crash) if the LLM API call fails.

### How to Run

```bash
pytest tests/test_pipeline.py -v
```

No API keys needed. All LLM and HTTP calls are mocked.

---

## Output Quality

### Accuracy

Each extracted value must be grounded in a real text snippet from the source page. The LLM is explicitly told not to invent values - it must quote supporting text. Values that pass through Stage 4 can be verified by the user via the source modal in the UI.

LLM-filled values (Stage 7) are different - they come from the model's training knowledge and can be wrong, especially for specific factual details like current funding or leadership. They are clearly marked in the UI.

### Usefulness

The output is structured, not a list of links. For a query like "AI startups in healthcare", you get a table of company names, descriptions, founding years, headquarters, and funding stage - all in one place, with sources. This is meaningfully more useful than a raw search results page.

The agentic loop improves completeness: if round 1 finds companies but misses their founding years, round 2 searches specifically for those missing facts.

### Latency

A single-round search takes roughly 15-25 seconds end-to-end:
- Planning: ~2s (one LLM call)
- Searching: ~3s (3 Tavily queries)
- Scraping: ~5-8s (concurrent, but some pages are slow)
- Extraction: ~5-10s (one LLM call per page, up to 8 concurrent)
- Resolving + gap analysis: ~2-3s

Each additional round adds approximately 10-15 seconds. The SSE progress stream makes the wait tolerable - users see activity rather than a blank screen.

### Cost

The primary cost drivers are LLM calls (Cerebras) and search queries (Tavily). Per request:
- 1 planning call
- 1 extraction call per scraped page (~8-12 pages per round)
- 1 resolver LLM call (if >5 entity groups)
- 1 gap analyzer call per round
- 1 filler call per entity with gaps

With 2 rounds and a typical query, this is roughly 20-25 LLM calls. Cerebras pricing makes this inexpensive at scale. Tavily basic search is low-cost per query.

---

## Design Choices and Trade-offs

### Problem: LLMs hallucinate specific facts
**Solution:** Required supporting snippets. Every cell value must be accompanied by the exact text from the source page. No snippet = value rejected. LLM-filled values (no web source) are marked separately.

### Problem: The same company appears on dozens of pages under slightly different names
**Solution:** Two-pass deduplication. String normalization handles trivial variants. LLM deduplication handles semantic variants (e.g., "Google DeepMind" vs "DeepMind"). Only the LLM pass is triggered when the entity count is high enough to justify the cost.

### Problem: A static pipeline can't know when it has enough data
**Solution:** The gap analyzer computes per-column coverage and uses a threshold (70% coverage + 8 entities) to decide whether to continue searching. This makes the search depth adaptive rather than fixed.

### Problem: Targeted gap-filling requires better queries than broad topic searches
**Solution:** The gap analyzer generates per-entity queries that name the entity and list its missing attributes. This produces much more focused searches in round 2 and 3 than repeating the original query.

### Problem: Web pages are full of boilerplate that wastes LLM context
**Solution:** The scraper strips navigation, footers, scripts, ads, and cookie banners before passing content to the LLM. Only semantic content (article body, main content divs) is kept. Pages are also capped at 8KB.

### Problem: LLMs don't always return clean JSON
**Solution:** `utils.py` has a robust JSON extractor that handles markdown code fences, `<think>` blocks, and trailing text. Rather than assuming clean output, the pipeline always extracts from potentially noisy output.

### Problem: One search round misses niche or specialized sources
**Solution:** The planner generates 3 diverse queries per round, approaching the topic from different angles. This increases the chance of finding pages that specialize in the specific entity type.

### Trade-off: Vanilla JS vs. a framework
Chosen simplicity. One page, one data flow, no routing, no state management library needed. Vanilla JS is more readable for reviewers and requires no build environment.

### Trade-off: Schema is inferred, not user-defined
The planner picks columns based on the query. This is more convenient but less controllable. A power user might want to specify custom columns; the current design doesn't support that.

### Trade-off: Max 10 entities, 8KB per page, 3 rounds max
All three caps exist to control cost and latency. They make the system practical for real use but mean it can't produce exhaustive research outputs (e.g., 50 entities with 10 columns).

---

## Known Limitations

- **Max 10 entities in output.** The resolver caps the output at 10. Useful as a summary; not useful for exhaustive lists.
- **Max 8KB of content per page.** Data near the bottom of long pages may be missed.
- **Max 3 search rounds.** Controlled by the UI dropdown. Queries that need deep coverage may not fully saturate all columns in 3 rounds.
- **LLM-filled values are not web-sourced.** They come from the model's training data and may be outdated or wrong. The UI marks them clearly, but users should verify independently.
- **English-language focus.** The LLM and prompts are in English. Queries about entities primarily documented in other languages will produce sparse results.
- **No authentication.** The API is open. In production, rate limiting and auth would be needed.
- **No persistent caching.** The scraper caches pages in memory within a single pipeline run. There is no cross-request cache, so the same URL may be fetched again in a later search session.
- **No PDF support.** The scraper skips PDF URLs. Many authoritative sources (whitepapers, reports) are PDFs and are not scraped.
- **Binary and paywalled pages are skipped.** The scraper skips non-HTML content and will return an empty result for paywalled articles, even if Tavily surfaced them.
