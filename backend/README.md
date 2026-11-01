# End to End Pipeline Walkthrough

**Example query** `"cheap pizza in Brooklyn"` in **Agentic** mode

This document follows one request from the question to the final table. It shows the values each stage passes on. The values are realistic but made up for this example. A real run gives different pages and rows.

| Stage | File | LLM calls in this run |
|---|---|---|
| 1 Plan | `pipeline/planner.py` | 1 |
| 2 Retrieve | `pipeline/retriever.py` | 0 |
| 3 Embed and Extract | `embedder.py`, `pipeline/extractor.py` | 1 |
| 4 Resolve | `pipeline/resolver.py` | 0 |
| 5 Fill Loop | `pipeline/filler.py` | 3 |
| 6 Rank | `pipeline/ranker.py` | 0 |

Total for this run is 5 LLM calls. Fast mode would make 3.

---

## Startup

When the server starts, `main.py` does two things once.

```python
llm_client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, max_retries=5)
embedder.load_model()
```

The LLM is Groq `openai/gpt-oss-20b`. The embedding model is `BAAI/bge-small-en-v1.5`, which runs on CPU through fastembed. Loading it at startup means no search waits for it.

---

## Request

```
POST /search
query = "cheap pizza in Brooklyn"
search_depth = 2
```

`SearchRequest` checks that the query has 3 to 300 characters and that `search_depth` is 1 to 3. Depth 1 is Fast. Anything above 1 is Agentic.

```python
agentic = request.search_depth > 1
```

The endpoint returns a **Server-Sent Events stream**. Each stage sends progress events, and the last event holds the table. When nothing is sent for 4 seconds, the stream sends a keepalive line to keep the connection open.

---

## Stage 1 Plan (`pipeline/planner.py`)

**Function** `plan_search(client, query)`

One LLM call picks the columns, writes the search queries and turns the filters in the question into rank rules.

### LLM call

```python
client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        dict(role="system", content=SYSTEM_PLANNER_PROMPT),
        dict(role="user", content="Current year 2026 ... Plan a structured search for cheap pizza in Brooklyn"),
    ],
    max_tokens=1024,
    reasoning_effort="low",
    temperature=0.3,
    response_format=dict(type="json_object"),
)
```

Before the call, the pacer reserves about 2,060 tokens (the prompt plus 1,024 for the answer). After the call, it records the real usage Groq reports.

### What the LLM returns

```python
dict(
    entity_type="cheap Brooklyn pizza places",
    columns=["name", "neighborhood", "pizza_style", "signature_pie", "price_range", "phone", "highlights"],
    search_queries=[
        "best cheap pizza in Brooklyn 2026",
        "Brooklyn pizza places ranked",
        "affordable pizza slices Brooklyn",
    ],
    rules=dict(
        location=dict(column="neighborhood", names=["Brooklyn", "BK", "Williamsburg", "DUMBO", "Park Slope", "Bushwick", "Carroll Gardens", "Greenpoint"]),
        numeric=[dict(column="price_range", op="<=", value=2)],
        categorical=[dict(column="pizza_style", keywords=["pizza", "pizzeria", "slice"])],
    ),
    rationale="Compare style, price and area for cheap pizza in Brooklyn.",
)
```

### Cleanup in code

| Step | Function | Result |
|---|---|---|
| Columns to snake case, `name` first, lookup columns removed, at most 8 | `_clean_columns` | `phone` is removed |
| Exactly 3 queries, padded with the user query if needed | `_clean_queries` | the 3 queries stay |
| Each rule must use a real column, a valid operator and a number | `_clean_rules` | all 3 rules stay, place names cut to 10 |

If the call fails, or fewer than 3 columns are left, `_fallback_plan` returns a safe default plan.

### Output `SearchPlan`

```python
plan = SearchPlan(
    entity_type="cheap Brooklyn pizza places",
    columns=["name", "neighborhood", "pizza_style", "signature_pie", "price_range", "highlights"],
    search_queries=["best cheap pizza in Brooklyn 2026", "Brooklyn pizza places ranked", "affordable pizza slices Brooklyn"],
    rules=dict(location=..., numeric=[...], categorical=[...]),
    rationale="Compare style, price and area for cheap pizza in Brooklyn.",
)
```

Progress events go out with stage `planning` at 0.04 and 0.10.

---

## Stage 2 Retrieve (`pipeline/retriever.py`)

**Function** `retrieve_with_backup(queries, seen_urls)`

The first 2 queries are the main ones. The third is the backup. Both main queries run at the same time through `asyncio.gather`.

### Tavily call (per query)

```python
await client.search(
    query="best cheap pizza in Brooklyn 2026",
    max_results=4,
    search_depth="basic",
    include_raw_content=True,
    exclude_domains=["youtube.com", "pinterest.com", "instagram.com", "tiktok.com", "facebook.com"],
)
```

`include_raw_content=True` makes Tavily return the page text. There is no separate scraper.

### Turning results into pages

Each result goes through `_to_page`.

1. Take `raw_content`, or the short snippet if there is no raw content.
2. `clean_markdown` removes images, link targets, bare URLs and menu lines made only of links.
3. Pages with less than 80 characters are dropped.
4. Text is cut to 8,000 characters.

### Removing copies of the same page

`normalize_url` makes copies of one page look the same. It lowercases the host, drops `www.`, drops the trailing slash, removes `utm_` and click tracking parameters and forces https.

A link to `grubstreet.com/best-pizza` with http, `www.`, a trailing slash and a `utm_source` tag gives the same key as the clean https link. The second copy is skipped.

### Output

The 2 main queries return 8 results. One is a copy, which leaves 7 pages. That is 4 or more, and the backup query does not run.

```python
pages = [
    ScrapedPage(url="grubstreet.com/best-pizza-brooklyn", title="The Best Pizza in Brooklyn", content="Lucali in Carroll Gardens ..."),
    ScrapedPage(url="theinfatuation.com/new-york/guides/best-pizza-brooklyn", title="The Best Pizza In Brooklyn", content="..."),
    ScrapedPage(url="timeout.com/newyork/restaurants/best-pizza-in-brooklyn", title="Best pizza in Brooklyn", content="..."),
    ...
]
used = ["best cheap pizza in Brooklyn 2026", "Brooklyn pizza places ranked"]
```

If fewer than 4 pages came back, `retrieve_with_backup` would run the backup query too. If every query fails, `retrieve` raises the first error and the user sees it.

Progress events go out with stage `searching` at 0.12 and 0.25.

---

## Stage 3 Embed and Extract (`embedder.py`, `pipeline/extractor.py`)

### 3a Chunk and embed

**Functions** `embed_pages(pages)` and `ChunkIndex.add(chunks, vectors)`

`chunk_pages` splits each page into chunks of 90 words. Each chunk starts 70 words after the last one, which gives 20 words of overlap. A fact that sits on a border is still whole in one of the two chunks. Pieces with fewer than 8 words are dropped.

```python
chunks, vectors = embed_pages(pages)
len(chunks)
```

The 7 pages give 112 chunks. Each chunk becomes a 384 number vector, normalized to length 1. The vectors go into a FAISS `IndexFlatIP`. With unit vectors, inner product is the same as cosine similarity. The index lives only for this one request.

### 3b Pick the best chunks

**Function** `top_chunks(chunks, vectors, topic, 3000)`

```python
topic = "cheap Brooklyn pizza places, name, neighborhood, pizza_style, signature_pie, price_range, highlights"
```

Every chunk is scored against the topic. The best ones are taken until the 3,000 token budget is full. They are then put back in page order, which keeps the text easy to read. In this run 19 chunks fit.

### 3c Extract with one LLM call

**Function** `extract_from_chunks(client, best, columns, entity_type)`

The chunks are numbered in the prompt.

```
[CHUNK 1] The Best Pizza in Brooklyn
Lucali in Carroll Gardens makes one of the best plain pies in the city ...

[CHUNK 2] The Best Pizza in Brooklyn
Di Fara Pizza in Midwood has served square and round slices since 1965 ...
```

The pacer reserves about 5,900 tokens (the chunks, the prompt and 2,500 for the answer). The model must give the chunk number and a short exact quote for every value.

```python
dict(entities=[
    dict(chunk=1,
         name=dict(value="Lucali", snippet="Lucali in Carroll Gardens"),
         neighborhood=dict(value="Carroll Gardens", snippet="Lucali in Carroll Gardens"),
         pizza_style=dict(value="Thin crust", snippet="thin crust pies baked in a brick oven")),
    dict(chunk=2,
         name=dict(value="Di Fara Pizza", snippet="Di Fara Pizza in Midwood"),
         neighborhood=dict(value="Midwood", snippet="Di Fara Pizza in Midwood"),
         price_range=dict(value="unknown", snippet="")),
])
```

### Checks in code

| Check | Function | Example |
|---|---|---|
| Find the chunk by number, or by name if the number is wrong | `_chunk_for` | a bad number for "Lucali" is fixed by finding "lucali" in a chunk |
| Drop placeholder values | `is_missing` | `"unknown"` for Di Fara price is dropped |
| Drop rows with no name, or a name longer than 60 characters or 6 words | `_to_entity` | a sentence picked up as a name is dropped |

Each kept value becomes a `CellValue` with confidence 0.9 and a `SourceRef` holding the page URL, the page title and the quote.

### Output

23 raw rows from 19 chunks, with copies. Lucali shows up 3 times, from 3 pages.

Progress events go out with stage `extracting` at 0.28 and 0.32.

---

## Stage 4 Resolve (`pipeline/resolver.py`)

**Function** `resolve_entities(raw, all_pages, context)`

No LLM here. It merges rows that name the same place.

### Pass 1 Exact names after cleanup (`_fast_dedup`)

`_normalise_name` removes accents, lowercases, turns `&` into `and`, removes punctuation, drops a leading "the" and trailing words like Inc or LLC.

| Raw name | Clean name |
|---|---|
| L&B Spumoni Gardens | l and b spumoni gardens |
| L and B Spumoni Gardens | l and b spumoni gardens |
| The Lucali | lucali |

Rows with the same clean name join one group. 23 rows become 14.

### Pass 2 Close names (`_group` with union-find)

Every pair is tested with `_same_entity`.

1. If both rows have a website column and the domains differ, they are different places.
2. Otherwise they match if the fuzzy score is 0.92 or more (`difflib`, spaces removed).
3. Or they match if they share rare words (`_rare_overlap`).

Rare words come from `WordWeights`. A word's weight is `log(paragraphs / paragraphs with the word)`, the IDF part of BM25. Words from the question and entity type get weight 0.

```
"di fara pizza"  vs  "di fara"
fuzzy score of "difarapizza" vs "difara" is 0.71, too low
"pizza" is a topic word, weight 0
"di" and "fara" are rare, and all of the smaller name's weight is shared
match
```

```
"best pizza"  vs  "brooklyn pizza co"
"pizza" and "brooklyn" are topic words, weight 0
nothing rare is shared
no match
```

Union-find makes matches carry over. If A matches B and B matches C, all three end up in one group. 14 rows become 11.

### Merging a group (`_merge_cells`)

For each column, values are grouped by their text. The value that the most pages agree on wins. Only the sources that agree are kept.

```python
lucali_price = [
    CellValue(value="$", sources=[SourceRef(url="grubstreet.com/...")]),
    CellValue(value="$", sources=[SourceRef(url="timeout.com/...")]),
    CellValue(value="$$", sources=[SourceRef(url="theinfatuation.com/...")]),
]
```

The result is `"$"` with the two sources that say `"$"`. The page that said `"$$"` is not cited for this cell.

### Output

11 unique places, sorted by how many pages mention them, then by how many cells they have. There is no cut here. The ranker picks the top 10 later.

```python
context = "cheap pizza in Brooklyn cheap Brooklyn pizza places"
```

Progress events go out with stage `resolving` at 0.45 and 0.50.

### Agentic backup search

Agentic runs the backup query only when there are fewer than 8 rows and an unused query is left. We have 11 places, which is 8 or more. The backup query does not run. With fewer than 8, Agentic would search with the unused query and run Stage 3 and 4 again on the new pages.

---

## Stage 5 Fill Loop (`pipeline/filler.py`)

**Function** `fill_loop(client, entities, columns, entity_type, chunk_index, seen_urls, max_fills=3)`

Agentic allows up to 3 rounds. Fast allows 1.

### Round 1 Fill from pages we already have

**Find gaps** with `find_gaps`. Each empty or placeholder cell of the first 15 rows is a gap. The name column is skipped. This run has 24 gaps.

**Search FAISS** with one query per gap, 8 hits each.

```python
_query(gap)
```

For Di Fara and `signature_pie`, the query is `"Di Fara Pizza signature pie"`.

**Keep only useful chunks** with `_attach_chunks`.

| Rule | Why |
|---|---|
| Score 0.6 or more | weak matches waste tokens |
| The chunk names this row (`_mentions`) | a chunk about Lucali cannot fill a Di Fara cell |
| Not sent before for this same cell | a retry never pays for the same text twice |
| At most 2 chunks per cell | keeps the prompt small |
| Stop when the token budget is full | stays under the minute limit |

Each chunk is trimmed by `_trim` to the sentences that name the row, plus one sentence on each side. A chunk used by two cells of the same row is sent once.

**Token budget.** Fast mode has one round. There `_fill_budget` uses what is left of the minute, between 500 and 4,000 tokens. Agentic uses 4,000.

**The prompt** has the entity type, the chunks and the cells.

```python
chunk_rows = [dict(id="c1", text="Di Fara Pizza in Midwood is famous for its square slice, topped with fresh basil and olive oil.")]
cell_rows = [dict(id=0, entity="Di Fara Pizza", column="signature_pie", chunks=["c1"])]
```

**The answer**

```python
dict(cells=[
    dict(id=0, value="Square slice", chunk="c1", quote="famous for its square slice"),
    dict(id=1, value=None, chunk=None, quote=None),
])
```

**Quote checks** with `_quote_ok`. A value is kept only if all of these pass.

1. The cell id is real and the chunk was offered to that cell.
2. The quote is really in the chunk (after lowercasing and removing punctuation).
3. Every word of the value is in the quote.
4. Numbers match exactly. `"800"` is fine inside `"$800M"`, but `"80"` is not.

```
value "Square slice", quote "famous for its square slice"   kept
value "1964", quote "since 1965"                           dropped, 1964 is not in the quote
```

A kept value becomes a `CellValue` with confidence 0.8 and the chunk's page and quote as its source. Round 1 fills 9 of 24 gaps.

### Round 2 Search the web for rows with gaps

**Targeted queries** from `targeted_queries`. One per row with gaps, at most 8.

```python
[
    "Di Fara Pizza price range highlights",
    "Paulie Gee's signature pie price range",
    "Totonno's pizza style highlights",
    ...
]
```

`retrieve` runs them and skips pages we already have. 13 new pages come back. They are chunked, embedded and added to the same FAISS index. Then the same fill step runs. Round 2 fills 6 cells.

### Round 3 Same, with the entity type added

From round 3 on, the queries add the entity type, like `"Totonno's cheap Brooklyn pizza places highlights"`. This helps when a name alone is too vague. Round 3 fills 0 cells, and the loop stops.

### When the loop stops

| Stop rule | Where |
|---|---|
| No gaps are left | start of each round after the first |
| The search finds no new pages | after `retrieve` |
| The search fails | after `retrieve` |
| A round after the first fills nothing | end of the round |
| `max_fills` is reached | the loop limit |

### Output `FillResult`

```python
FillResult(
    entities=[...],
    filled=15,
    rounds=3,
    queries=["Di Fara Pizza price range highlights", ...],
    sources=["slicelife.com/...", ...],
)
```

Progress events go out with stage `filling` between 0.58 and 0.92.

---

## Stage 6 Rank (`pipeline/ranker.py`)

**Function** `rank(entities, plan.rules, plan.columns, query, model)`

No LLM here either.

### Build the checks (`_checks`)

Each rule becomes a label, a test function and the rule itself. The function is stored, not called.

```python
checks = [
    ("In Brooklyn", _location_met, location_rule),
    ("price range <= 2", _numeric_met, price_rule),
    (style_label, _categorical_met, style_rule),
]
```

The categorical label is the column name followed by the first keyword, which shows as "pizza style" and "pizza" in one pill.

| Test | How it works |
|---|---|
| `_location_met` | the location cell has one of the place names as a whole word |
| `_numeric_met` | `_to_number` reads the cell. `"$$"` is 2, `"5M"` is 5,000,000. Then it compares with the limit |
| `_categorical_met` | a keyword is anywhere in the row, as a whole word |

An empty cell never passes.

### Score each row

| Row | neighborhood | price | style | Met |
|---|---|---|---|---|
| Lucali | Carroll Gardens | $ | Thin crust pizza | 3/3 |
| Joe's Pizza | Williamsburg | $ | New York slice | 3/3 |
| L&B Spumoni Gardens | Gravesend | $ | Sicilian square pizza | 2/3 |
| Juliana's | DUMBO | $$$ | Coal oven pizzeria | 2/3 |
| Di Fara Pizza | Midwood | (empty) | Square slice | 1/3 |

L&B misses the location rule because the planner did not list Gravesend. Di Fara has no price, which counts as not met.

### Sort

```python
key = (met, relevance bucket, source count, filled cells)
```

1. Rules met.
2. Relevance. The row text and the question are embedded and compared with cosine similarity, rounded to steps of 0.1. This keeps tiny score gaps from beating the next keys.
3. Number of pages that mention the row.
4. Number of filled cells.

The top 10 are kept. One of the 11 places is cut.

### Pick columns (`pick_columns`)

A column is shown if it is `name`, a rule uses it, or every top row has it filled. If fewer than 3 columns besides `name` are left, columns that are 80% filled are added back.

| Column | Filled | Shown | Why |
|---|---|---|---|
| name | 100% | yes | always |
| neighborhood | 100% | yes | rule column |
| pizza_style | 100% | yes | rule column |
| signature_pie | 100% | yes | full |
| price_range | 90% | yes | rule column |
| highlights | 70% | no | not full and not a rule column |

### Output

```python
top, shown, info = rank(...)
shown = ["name", "neighborhood", "pizza_style", "signature_pie", "price_range"]
info = RankingInfo(
    total=3,
    labels=["In Brooklyn", "price range <= 2", style_label],
    scores=dict(lucali_id=3, joes_id=3, lb_id=2, ...),
)
```

`scores` maps each row id to its count. Ids are used, not names, because ids are always unique.

The progress event goes out with stage `ranking` at 0.98, then `done` at 1.0.

---

## Final Table `EntityTable`

```python
result = EntityTable(
    query="cheap pizza in Brooklyn",
    entity_type="cheap Brooklyn pizza places",
    columns=shown,
    entities=top,
    sources_consulted=all_sources,
    search_queries_used=all_queries_used,
    rounds_completed=3,
    ranking=info,
)
```

| Field | Value in this run |
|---|---|
| `columns` | the 5 shown columns |
| `entities` | the top 10 rows in rank order |
| `sources_consulted` | 7 pages from Stage 2 and 13 from round 2 |
| `search_queries_used` | 2 main queries and 8 targeted queries |
| `rounds_completed` | 3 |

`result.model_dump(mode="json")` goes out as the last event with type `result`. If anything fails, an event with type `error` and the message goes out instead.

---

## Token Budget (`config.py`)

The Groq free tier allows about 8,000 tokens a minute. `TokenPacer` keeps the run under 7,900.

1. `reserve(tokens)` adds up the tokens used in the last 60 seconds. If the new call would go over 7,900, it waits until old calls leave the window.
2. `settle(event, usage, stage)` swaps the guess for the real number from Groq.
3. One pacer is shared by every request in the server.

Tokens are guessed as characters divided by 3.5.

| Call | Reserved before the call |
|---|---|
| Plan | about 2,060 |
| Extract | about 5,900 |
| Fill | up to about 5,700 |

Plan and extract fit in one minute once the plan's real usage is recorded. A fill round may wait a few seconds for room. The filler also drops cells until a single prompt fits under the limit.

---

## UI Rendering

### Progress

The frontend shows 6 steps, Plan, Retrieve, Extract, Resolve, Fill and Rank. Each `progress` event moves the bar and updates the status text.

### Table

| Match | name | neighborhood | pizza style | signature pie | price range |
|---|---|---|---|---|---|
| 3/3 | Lucali | Carroll Gardens | Thin crust pizza | Plain pie with fresh basil | $ |
| 3/3 | Joe's Pizza | Williamsburg | New York slice | Cheese slice | $ |
| 2/3 | L&B Spumoni Gardens | Gravesend | Sicilian square pizza | Sicilian square | $ |
| 2/3 | Juliana's | DUMBO | Coal oven pizzeria | Margherita | $$$ |
| 1/3 | Di Fara Pizza | Midwood | Square slice | Square slice | (empty) |

Above the table, the rules show as pills next to "Ranked by". The Match badge is green at 3/3 and lighter as fewer rules are met.

### Source popup

Each value has small numbered buttons, one per source. A click opens the page title, the link and the quote.

```
Di Fara Pizza / signature pie
slicelife.com/restaurants/ny/brooklyn/di-fara-pizza
"famous for its square slice"
Open page
```

The dot next to each value shows confidence. Extracted values have 0.9 and filled values have 0.8.

The JSON button saves the full result. The CSV button saves one row per place with a last column of source links.

---

## Summary of Data Flow

```
query
  plan_search()                  1 LLM call
SearchPlan  entity_type, 6 to 8 columns, 3 queries, rules

  retrieve_with_backup()         2 Tavily queries, backup if fewer than 4 pages
list of ScrapedPage  url, title, clean text up to 8,000 characters

  embed_pages()  ChunkIndex.add()  top_chunks()
90 word chunks in FAISS, best chunks under 3,000 tokens

  extract_from_chunks()          1 LLM call
raw rows with chunk, page and quote per value

  resolve_entities()             no LLM
unique rows, values most pages agree on

  backup search if Agentic and fewer than 8 rows

  fill_loop()                    1 LLM call per round, 1 round Fast, up to 3 Agentic
empty cells filled only when the quote checks pass

  rank()                         no LLM
top 10 rows, shown columns, RankingInfo

  EntityTable sent as the last SSE event
UI shows the table, Match badges and a source for every value
```
