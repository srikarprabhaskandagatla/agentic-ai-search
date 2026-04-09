# End-to-End Pipeline Walkthrough

**Example query:** `"best pizza restaurants in San Francisco"`

This document traces a single request from the user's query to the final rendered table,
showing realistic variable values at every stage.

---

## Request

```
POST /api/search
{
  "query": "best pizza restaurants in San Francisco",
  "search_depth": 2
}
```

The `SearchRequest` model validates:
- `query` is between 3 and 300 characters
- `search_depth` is between 1 and 3

The endpoint returns a **Server-Sent Events (SSE) stream**. Progress events are emitted
throughout so the frontend can update a progress bar in real time.

---

## Stage 1 - Planner (`pipeline/planner.py`)

**Function:** `plan_search(client, query)`

The LLM receives the raw query and must infer what kind of entities to look for,
design a 5-column table schema, and generate 3 diverse search queries.

### Input

```python
query = "best pizza restaurants in San Francisco"
```

### LLM call

```python
client.chat.completions.create(
    model="qwen-3-235b-a22b-instruct-2507",
    temperature=0.3,
    max_tokens=1024,
    messages=[
        {"role": "system", "content": SYSTEM_PLANNER_PROMPT},
        {"role": "user",   "content": "best pizza restaurants in San Francisco"},
    ]
)
```

### LLM JSON response (parsed by `extract_json_obj()`)

```json
{
  "entity_type": "pizza restaurant",
  "columns": ["name", "description", "neighborhood", "price_range", "signature_pizza"],
  "search_queries": [
    "best pizza restaurants San Francisco 2024",
    "top rated SF pizza places Yelp TripAdvisor",
    "authentic Neapolitan pizza San Francisco Mission District"
  ],
  "rationale": "Pizza restaurants vary by style and neighborhood; key differentiators are location, price, and signature dish."
}
```

### Output - `SearchPlan`

```python
plan = SearchPlan(
    entity_type    = "pizza restaurant",
    columns        = ["name", "description", "neighborhood", "price_range", "signature_pizza"],
    search_queries = [
        "best pizza restaurants San Francisco 2024",
        "top rated SF pizza places Yelp TripAdvisor",
        "authentic Neapolitan pizza San Francisco Mission District",
    ],
    rationale = "Pizza restaurants vary by style and neighborhood; ..."
)
```

**Progress event emitted:**
```json
{"type": "progress", "stage": "planning", "message": "Search plan ready", "progress": 0.10}
```

---

## Stage 2 - Searcher (`pipeline/searcher.py`)

**Function:** `fetch_web_results(query, max_results=4)`

Each of the 3 queries runs as a concurrent Tavily search. Each call returns at most
4 results (`results[:max_results]`). All 3 tasks fire at once via `asyncio.gather`,
so up to 12 URLs come back in parallel, then deduped by `seen_urls`.

### Input

```python
present_query = [
    "best pizza restaurants San Francisco 2024",
    "top rated SF pizza places Yelp TripAdvisor",
    "authentic Neapolitan pizza San Francisco Mission District",
]
search_tasks = [fetch_web_results(q, max_results=4) for q in present_query]
results_per_query = await asyncio.gather(*search_tasks, return_exceptions=True)
# return_exceptions=True means if one query fails, the others still succeed
```

### Tavily call (per query)

```python
await tavily_client.search(
    query          = "best pizza restaurants San Francisco 2024",
    max_results    = 4,
    search_depth   = "basic",
    include_answer = False,
)
```

### `results_per_query` — raw output before flattening

Each inner list is the result of one query. This is a list of lists:

```python
results_per_query = [
    # query 1 results:
    [
        SearchResult(url="https://sf.eater.com/maps/best-pizza-sf",
                     title="The 12 Best Pizza Spots in San Francisco",
                     snippet="From Neapolitan pies to Detroit-style slices, SF has it all..."),
        SearchResult(url="https://www.thrillist.com/eat/san-francisco/best-pizza",
                     title="The Best Pizza in San Francisco Right Now",
                     snippet="Flour + Water Pizzeria tops many lists for its wood-fired pies..."),
        SearchResult(url="https://www.sfgate.com/food/best-pizza-sf",
                     title="SF Gate: Best Pizza in San Francisco",
                     snippet="From classic NY-style to Roman al taglio, SF delivers..."),
        SearchResult(url="https://www.tonysitaliankitchen.com",
                     title="Tony's Pizza Napoletana - North Beach SF",
                     snippet="World champion pizza maker Tony Gemignani. 13 styles of pizza."),
    ],
    # query 2 results:
    [
        SearchResult(url="https://www.yelp.com/search?find_desc=pizza&find_loc=San+Francisco",
                     title="Best Pizza in San Francisco - Yelp",
                     snippet="Tony's Pizza Napoletana, Del Popolo, Gialina..."),
        SearchResult(url="https://www.tripadvisor.com/Restaurants-g60713-c26-San_Francisco.html",
                     title="THE 10 BEST Pizza Places in San Francisco - Tripadvisor",
                     snippet="Delfina, Pizzeria Delfina, Gialina - highly rated by visitors..."),
        SearchResult(url="https://sf.eater.com/maps/best-pizza-sf",     # duplicate of query 1
                     title="The 12 Best Pizza Spots in San Francisco",
                     snippet="From Neapolitan pies to Detroit-style slices, SF has it all..."),
        SearchResult(url="https://www.timeout.com/san-francisco/restaurants/best-pizza",
                     title="Best pizza in San Francisco - Time Out",
                     snippet="Penny Roma brings Roman-style al taglio to the Mission..."),
    ],
    # query 3 results:
    [
        SearchResult(url="https://missionpizza.com",
                     title="Mission District Pizza Co. | San Francisco",
                     snippet="Award-winning Neapolitan pizzas in the heart of the Mission..."),
        SearchResult(url="https://www.delpopolosf.com",
                     title="Del Popolo - San Francisco Pizza",
                     snippet="Wood-fired Neapolitan pizza in SoMa. Seasonal ingredients."),
        SearchResult(url="https://www.gialina.com",
                     title="Gialina Pizzeria - Glen Park San Francisco",
                     snippet="Thin-crust artisan pizzas with seasonal toppings in Glen Park."),
        SearchResult(url="https://www.sfchronicle.com/food/best-pizza-sf",
                     title="SF Chronicle: Where to find the best pizza in SF",
                     snippet="Shuggie's Trash Pie is one of the most creative newcomers..."),
    ],
]
```

### Deduplication loop

After `asyncio.gather` returns, the code flattens and deduplicates:

```python
new_results: list[SearchResult] = []

for results in results_per_query:
    if isinstance(results, Exception):
        continue                         # skip failed queries
    for r in results:
        if r.url not in seen_urls:       # seen_urls starts empty in Round 1
            seen_urls.add(r.url)         # mark as visited
            new_results.append(r)        # keep it
            all_search_results.append(r) # add to global log
```

`sf.eater.com` appeared in both query 1 and query 2 results. The second occurrence
is skipped because `seen_urls` already contains it.

### Output - `new_results` (deduplicated)

```python
new_results = [
    SearchResult(url="https://sf.eater.com/maps/best-pizza-sf",
                 title="The 12 Best Pizza Spots in San Francisco",
                 snippet="From Neapolitan pies to Detroit-style slices, SF has it all..."),
    SearchResult(url="https://www.thrillist.com/eat/san-francisco/best-pizza",
                 title="The Best Pizza in San Francisco Right Now",
                 snippet="Flour + Water Pizzeria tops many lists for its wood-fired pies..."),
    SearchResult(url="https://www.sfgate.com/food/best-pizza-sf",
                 title="SF Gate: Best Pizza in San Francisco",
                 snippet="From classic NY-style to Roman al taglio, SF delivers..."),
    SearchResult(url="https://www.tonysitaliankitchen.com",
                 title="Tony's Pizza Napoletana - North Beach SF",
                 snippet="World champion pizza maker Tony Gemignani. 13 styles of pizza."),
    SearchResult(url="https://www.yelp.com/search?find_desc=pizza&find_loc=San+Francisco",
                 title="Best Pizza in San Francisco - Yelp",
                 snippet="Tony's Pizza Napoletana, Del Popolo, Gialina..."),
    SearchResult(url="https://www.tripadvisor.com/Restaurants-g60713-c26-San_Francisco.html",
                 title="THE 10 BEST Pizza Places in San Francisco - Tripadvisor",
                 snippet="Delfina, Pizzeria Delfina, Gialina - highly rated by visitors..."),
    SearchResult(url="https://www.timeout.com/san-francisco/restaurants/best-pizza",
                 title="Best pizza in San Francisco - Time Out",
                 snippet="Penny Roma brings Roman-style al taglio to the Mission..."),
    SearchResult(url="https://missionpizza.com",
                 title="Mission District Pizza Co. | San Francisco",
                 snippet="Award-winning Neapolitan pizzas in the heart of the Mission..."),
    SearchResult(url="https://www.delpopolosf.com",
                 title="Del Popolo - San Francisco Pizza",
                 snippet="Wood-fired Neapolitan pizza in SoMa. Seasonal ingredients."),
    SearchResult(url="https://www.gialina.com",
                 title="Gialina Pizzeria - Glen Park San Francisco",
                 snippet="Thin-crust artisan pizzas with seasonal toppings in Glen Park."),
    SearchResult(url="https://www.sfchronicle.com/food/best-pizza-sf",
                 title="SF Chronicle: Where to find the best pizza in SF",
                 snippet="Shuggie's Trash Pie is one of the most creative newcomers..."),
]

# State variables after Stage 2:
seen_urls        = {"https://sf.eater.com/...", "https://www.thrillist.com/...", ...}  # 11 URLs
all_queries_used = [
    "best pizza restaurants San Francisco 2024",
    "top rated SF pizza places Yelp TripAdvisor",
    "authentic Neapolitan pizza San Francisco Mission District",
]
```

**Progress event emitted:**
```json
{"type": "progress", "stage": "searching", "message": "Found 11 new URLs", "progress": 0.22}
```

---

## Stage 3 - Scraper (`pipeline/scraper.py`)

**Function:** `scrape_urls(urls)`

All 11 URLs are passed in. `asyncio.gather` fires all 11 tasks at once, but
`asyncio.Semaphore(MAX_CONCURRENT=6)` ensures only 6 are actively fetching at any
moment. As soon as one finishes, the next waiting task picks up the freed slot.

### Input

```python
urls = [r.url for r in new_results]
# ["https://sf.eater.com/maps/best-pizza-sf",
#  "https://www.thrillist.com/eat/san-francisco/best-pizza",
#  "https://www.sfgate.com/food/best-pizza-sf",
#  "https://www.tonysitaliankitchen.com",
#  "https://www.yelp.com/search?...",
#  "https://www.tripadvisor.com/...",
#  "https://www.timeout.com/...",
#  "https://missionpizza.com",
#  "https://www.delpopolosf.com",
#  "https://www.gialina.com",
#  "https://www.sfchronicle.com/food/best-pizza-sf"]
```

### Per-URL fetch (inside `_fetch_one()`)

Step-by-step for `https://sf.eater.com/maps/best-pizza-sf`:

```python
# Step 1 — check cache
if url in _url_cache:
    return _url_cache[url]   # skip fetch entirely if seen before

# Step 2 — check for binary file extensions
if _should_skip(url):        # e.g. url ends in .pdf, .mp4 etc.
    return ScrapedPage(url=url, title="", content="", error="skipped: binary")

# Step 3 — fetch under semaphore
async with semaphore:        # blocks until one of the 6 slots is free
    response = await client.get(
        url,
        timeout        = REQUEST_TIMEOUT,     # 12.0 seconds
        follow_redirects = True,
    )

# Step 4 — validate response
# checks: response.status_code == 200
# checks: "text/html" in response.headers["content-type"]

# Step 5 — clean HTML
title, text = _clean_html(response.text, url)
# _clean_html removes all JUNK_TAGS: script, style, nav, footer, header,
#   aside, iframe, noscript, form, button, svg, ads, cookie-banner
# then prefers content from <article>, <main>, or elements with
#   "content"/"main"/"article" in their id or class
# filters out lines shorter than 20 chars (unless they end with ":")
# truncates final text to MAX_CONTENT_CHARS = 8000

# Step 6 — cache and return
_url_cache[url] = ScrapedPage(url=url, title=title, content=text, error=None)
```

### Output - `list[ScrapedPage]` (only successful, non-empty pages)

Yelp and TripAdvisor block scrapers and return non-200 status, so they fail.
9 out of 11 succeed:

```python
pages = [
    ScrapedPage(
        url     = "https://sf.eater.com/maps/best-pizza-sf",
        title   = "The 12 Best Pizza Spots in San Francisco - Eater SF",
        content = "Tony's Pizza Napoletana in North Beach has been voted best pizza "
                  "in America multiple times. Located at 1570 Stockton St, it offers "
                  "13 styles including Neapolitan, New York, and Detroit. Prices range "
                  "from $18-$28 for a pie.\n\nDel Popolo is a standout in SoMa, known "
                  "for its wood-fired oven and minimalist approach. A margherita runs $22. "
                  "Gialina in Glen Park serves thin-crust pies with seasonal toppings; "
                  "reservations recommended. Average price $20-25...",
        error   = None,
    ),
    ScrapedPage(
        url     = "https://www.tonysitaliankitchen.com",
        title   = "Tony's Pizza Napoletana - North Beach SF",
        content = "Tony Gemignani, 13-time world pizza champion, serves 13 distinct "
                  "pizza styles under one roof. Signature pie: the Tony's Special, "
                  "a coal-fired New York-style pizza at $26. Open Tue-Sun, "
                  "1570 Stockton Street, North Beach neighborhood...",
        error   = None,
    ),
    ScrapedPage(
        url     = "https://missionpizza.com",
        title   = "Mission District Pizza Co.",
        content = "Authentic Neapolitan pizza baked in a 900 degree wood-fired oven. "
                  "Located at 3200 16th St in the Mission District. "
                  "Signature: the Diavola with spicy salami, $19. Price range $$.",
        error   = None,
    ),
    ScrapedPage(
        url     = "https://www.delpopolosf.com",
        title   = "Del Popolo - San Francisco Pizza",
        content = "Del Popolo serves wood-fired Neapolitan pizza from a custom-built "
                  "oven in SoMa. Known for its minimalist approach and seasonal ingredients. "
                  "Signature: Margherita with house-pulled mozzarella, $22.",
        error   = None,
    ),
    ScrapedPage(
        url     = "https://www.gialina.com",
        title   = "Gialina Pizzeria - Glen Park San Francisco",
        content = "Gialina in Glen Park offers thin-crust artisan pizzas with seasonal "
                  "toppings. The Funghi pizza with rotating wild mushrooms is a perennial "
                  "favorite. Average price $20-25. Reservations strongly recommended.",
        error   = None,
    ),
    ScrapedPage(
        url     = "https://www.thrillist.com/eat/san-francisco/best-pizza",
        title   = "The Best Pizza in San Francisco Right Now - Thrillist",
        content = "Flour + Water Pizzeria in the Mission tops many best-of lists. "
                  "Their wood-fired pies use seasonal produce from local farms. "
                  "Price range $20-$26. Don't miss the Taleggio and mushroom pizza.",
        error   = None,
    ),
    ScrapedPage(
        url     = "https://www.timeout.com/san-francisco/restaurants/best-pizza",
        title   = "Best pizza in San Francisco - Time Out",
        content = "Penny Roma has quickly become a Mission staple for Roman-style "
                  "pizza al taglio. Prices run $15-$20 per slice. "
                  "Their cacio e pepe pizza is a must-order.",
        error   = None,
    ),
    ScrapedPage(
        url     = "https://www.sfchronicle.com/food/best-pizza-sf",
        title   = "SF Chronicle: Where to find the best pizza in SF",
        content = "Shuggie's Trash Pie in the Tenderloin uses upcycled and "
                  "imperfect ingredients to create creative pies. "
                  "Price range $16-$22. The signature Trash Pie changes seasonally.",
        error   = None,
    ),
    ScrapedPage(
        url     = "https://www.sfgate.com/food/best-pizza-sf",
        title   = "SF Gate: Best Pizza in San Francisco",
        content = "From classic NY-style slices to Roman al taglio and Neapolitan pies, "
                  "San Francisco has a pizza for every style. Delfina in the Mission "
                  "has been a neighborhood staple for over 20 years. "
                  "Their sausage pizza with fennel pollen is legendary. Price: $18-$24.",
        error   = None,
    ),
]

# Fallback: for the 2 failed URLs (Yelp, TripAdvisor), if their snippet was >= 80 chars
# the scraper uses the Tavily snippet as a substitute ScrapedPage:
pages.append(ScrapedPage(
    url     = "https://www.yelp.com/search?find_desc=pizza&find_loc=San+Francisco",
    title   = "Best Pizza in San Francisco - Yelp",
    content = "Tony's Pizza Napoletana, Del Popolo, Gialina...",   # Tavily snippet
    error   = None,
))
```

**Progress event emitted:**
```json
{"type": "progress", "stage": "scraping", "message": "Scraped 9/11 pages successfully", "progress": 0.45}
```

---

## Stage 4 - Extractor (`pipeline/extractor.py`)

**Function:** `extract_from_pages(client, pages, columns, entity_type)`

All 10 pages are processed concurrently, limited by `asyncio.Semaphore(8)` — so at most
8 LLM calls run at the same time. Each page gets its own LLM call. The LLM must only
return values **explicitly present** in the page content — no hallucination allowed.

### Input (per page)

```python
columns     = ["name", "description", "neighborhood", "price_range", "signature_pizza"]
entity_type = "pizza restaurant"

# User message built from USER_TEMPLATE for the Eater SF page:
user_message = """
Extract all pizza restaurant entities from this page.
Columns: name, description, neighborhood, price_range, signature_pizza

URL: https://sf.eater.com/maps/best-pizza-sf
Title: The 12 Best Pizza Spots in San Francisco - Eater SF

CONTENT:
Tony's Pizza Napoletana in North Beach has been voted best pizza in America
multiple times. Located at 1570 Stockton St, it offers 13 styles including
Neapolitan, New York, and Detroit. Prices range from $18-$28 for a pie.

Del Popolo is a standout in SoMa, known for its wood-fired oven and minimalist
approach. A margherita runs $22.

Gialina in Glen Park serves thin-crust pies with seasonal toppings;
reservations recommended. Average price $20-25...
[truncated to 6000 chars]
"""
```

### LLM call (temperature=0.1 for near-determinism)

```python
client.chat.completions.create(
    model       = "qwen-3-235b-a22b-instruct-2507",
    temperature = 0.1,    # near-deterministic — structured extraction, not creative
    max_tokens  = 3000,
    messages    = [
        {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]
)
```

### LLM JSON response (from Eater SF page)

The LLM returns one object per restaurant found on the page. Every field has three
sub-keys: `value` (extracted text), `confidence` (0.0-1.0), and `snippet` (exact
quote from the page that supports the value). If a field is not found, `value` is `null`.

```json
{
  "entities": [
    {
      "name":            {"value": "Tony's Pizza Napoletana", "confidence": 0.99, "snippet": "Tony's Pizza Napoletana in North Beach has been voted"},
      "description":     {"value": "Multi-style pizzeria voted best pizza in America", "confidence": 0.95, "snippet": "voted best pizza in America multiple times"},
      "neighborhood":    {"value": "North Beach",  "confidence": 0.99, "snippet": "in North Beach"},
      "price_range":     {"value": "$18-$28",      "confidence": 0.97, "snippet": "Prices range from $18-$28 for a pie"},
      "signature_pizza": {"value": null,            "confidence": 0.0,  "snippet": ""}
    },
    {
      "name":            {"value": "Del Popolo",   "confidence": 0.99, "snippet": "Del Popolo is a standout in SoMa"},
      "description":     {"value": "Wood-fired minimalist Neapolitan pizza", "confidence": 0.92, "snippet": "known for its wood-fired oven and minimalist approach"},
      "neighborhood":    {"value": "SoMa",          "confidence": 0.99, "snippet": "standout in SoMa"},
      "price_range":     {"value": "$22",            "confidence": 0.95, "snippet": "A margherita runs $22"},
      "signature_pizza": {"value": null,             "confidence": 0.0,  "snippet": ""}
    },
    {
      "name":            {"value": "Gialina",       "confidence": 0.99, "snippet": "Gialina in Glen Park"},
      "description":     {"value": "Thin-crust seasonal pizzas in Glen Park", "confidence": 0.90, "snippet": "serves thin-crust pies with seasonal toppings"},
      "neighborhood":    {"value": "Glen Park",      "confidence": 0.99, "snippet": "Gialina in Glen Park"},
      "price_range":     {"value": "$20-$25",        "confidence": 0.95, "snippet": "Average price $20-25"},
      "signature_pizza": {"value": null,             "confidence": 0.0,  "snippet": ""}
    }
  ]
}
```

### Parsing loop — converting raw JSON into `Entity` objects

The code iterates over every entity the LLM returned, then over every column in the schema,
building up a `cells` dict one column at a time:

```python
entities: list[Entity] = []

for raw in data.get("entities", []):
    # raw is one restaurant dict from the LLM response.
    # For "Tony's Pizza Napoletana":
    # raw = {
    #   "name":         {"value": "Tony's Pizza Napoletana", "confidence": 0.99, "snippet": "..."},
    #   "description":  {"value": "Multi-style pizzeria...",  "confidence": 0.95, "snippet": "..."},
    #   "neighborhood": {"value": "North Beach",              "confidence": 0.99, "snippet": "..."},
    #   "price_range":  {"value": "$18-$28",                  "confidence": 0.97, "snippet": "..."},
    #   "signature_pizza": {"value": null,                    "confidence": 0.0,  "snippet": ""}
    # }

    cells: dict[str, CellValue] = {}

    for col in columns:
        # col cycles through: "name", "description", "neighborhood", "price_range", "signature_pizza"

        field = raw.get(col, {})
        # For col="neighborhood":
        # field = {"value": "North Beach", "confidence": 0.99, "snippet": "in North Beach"}
        # If the key is missing from raw entirely, field defaults to {}

        if not isinstance(field, dict):
            continue      # LLM returned something malformed (e.g. a plain string), skip it

        val = field.get("value")
        # val = "North Beach"   for neighborhood
        # val = None            for signature_pizza (LLM returned null)

        if val is None:
            continue
        # signature_pizza is skipped here — it will simply not appear in cells
        # (not stored as null — just absent entirely)

        cells[col] = CellValue(
            value      = val,
            # val = "North Beach"

            confidence = float(field.get("confidence", 0.8)),
            # confidence = 0.99

            sources    = [SourceRef(
                url     = page.url,
                # "https://sf.eater.com/maps/best-pizza-sf"

                title   = page.title,
                # "The 12 Best Pizza Spots in San Francisco - Eater SF"

                snippet = str(field.get("snippet", ""))[:400],
                # "in North Beach"  (capped at 400 chars)
            )],
        )
        # Every CellValue carries source attribution: which page + exact quote
```

After the inner loop, `cells` for Tony's Pizza Napoletana from this page:

```python
cells = {
    "name": CellValue(
        value      = "Tony's Pizza Napoletana",
        confidence = 0.99,
        sources    = [SourceRef(
            url     = "https://sf.eater.com/maps/best-pizza-sf",
            title   = "The 12 Best Pizza Spots in San Francisco - Eater SF",
            snippet = "Tony's Pizza Napoletana in North Beach has been voted",
        )],
    ),
    "description": CellValue(
        value      = "Multi-style pizzeria voted best pizza in America",
        confidence = 0.95,
        sources    = [SourceRef(url="https://sf.eater.com/maps/best-pizza-sf", ...)],
    ),
    "neighborhood": CellValue(
        value      = "North Beach",
        confidence = 0.99,
        sources    = [SourceRef(url="https://sf.eater.com/maps/best-pizza-sf",
                                snippet="in North Beach")],
    ),
    "price_range": CellValue(
        value      = "$18-$28",
        confidence = 0.97,
        sources    = [SourceRef(url="https://sf.eater.com/maps/best-pizza-sf",
                                snippet="Prices range from $18-$28 for a pie")],
    ),
    # "signature_pizza" is absent — val was None, so it was never added to cells
}
```

### Validation before appending

Two guard checks run before the entity is accepted:

```python
    if "name" not in cells:
        continue
    # If the LLM couldn't find a name (value was null), the entity is useless.
    # A row with no name can't be identified or deduplicated later.

    name_val = str(cells["name"].value)
    if len(name_val) > 60 or len(name_val.split()) > 6:
        continue
    # Rejects names that are clearly sentences, not restaurant names.
    # "The absolute best pizza you will find in all of San Francisco" (12 words) → rejected
    # "Tony's Pizza Napoletana" (3 words, 24 chars) → accepted
```

### Entity created

```python
    entities.append(Entity(id=str(uuid.uuid4()), cells=cells))
    # Each entity gets a brand-new random UUID as its id.
    # Same restaurant appearing on multiple pages = multiple Entity objects here.
    # Deduplication happens in Stage 5.
```

### Final `raw_entities` list (all pages combined)

After all 10 pages are processed and flattened (`[e for page_entities in results for e in page_entities]`),
`raw_entities` contains one Entity per restaurant per page. Tony's Pizza Napoletana appears twice
(once from the Eater SF page, once from its own website) because two different pages mentioned it.

```python
raw_entities = [
    # from Eater SF page (3 restaurants found):
    Entity(
        id    = "a1b2-c3d4",
        cells = {
            "name":        CellValue(value="Tony's Pizza Napoletana", confidence=0.99,
                               sources=[SourceRef(url="https://sf.eater.com/maps/best-pizza-sf",
                                                  title="The 12 Best Pizza Spots...",
                                                  snippet="Tony's Pizza Napoletana in North Beach has been voted")]),
            "description": CellValue(value="Multi-style pizzeria voted best pizza in America",
                               confidence=0.95,
                               sources=[SourceRef(url="https://sf.eater.com/maps/best-pizza-sf",
                                                  snippet="voted best pizza in America multiple times")]),
            "neighborhood":CellValue(value="North Beach", confidence=0.99,
                               sources=[SourceRef(url="https://sf.eater.com/maps/best-pizza-sf",
                                                  snippet="in North Beach")]),
            "price_range": CellValue(value="$18-$28", confidence=0.97,
                               sources=[SourceRef(url="https://sf.eater.com/maps/best-pizza-sf",
                                                  snippet="Prices range from $18-$28 for a pie")]),
            # "signature_pizza" absent — was null on this page
        }
    ),
    Entity(
        id    = "e5f6-g7h8",
        cells = {
            "name":        CellValue(value="Del Popolo", confidence=0.99,
                               sources=[SourceRef(url="https://sf.eater.com/maps/best-pizza-sf",
                                                  snippet="Del Popolo is a standout in SoMa")]),
            "description": CellValue(value="Wood-fired minimalist Neapolitan pizza",
                               confidence=0.92,
                               sources=[SourceRef(url="https://sf.eater.com/maps/best-pizza-sf",
                                                  snippet="known for its wood-fired oven and minimalist approach")]),
            "neighborhood":CellValue(value="SoMa", confidence=0.99,
                               sources=[SourceRef(url="https://sf.eater.com/maps/best-pizza-sf",
                                                  snippet="standout in SoMa")]),
            "price_range": CellValue(value="$22", confidence=0.95,
                               sources=[SourceRef(url="https://sf.eater.com/maps/best-pizza-sf",
                                                  snippet="A margherita runs $22")]),
        }
    ),
    Entity(
        id    = "i9j0-k1l2",
        cells = {
            "name":        CellValue(value="Gialina", confidence=0.99,
                               sources=[SourceRef(url="https://sf.eater.com/maps/best-pizza-sf",
                                                  snippet="Gialina in Glen Park")]),
            "description": CellValue(value="Thin-crust seasonal pizzas in Glen Park",
                               confidence=0.90,
                               sources=[SourceRef(url="https://sf.eater.com/maps/best-pizza-sf",
                                                  snippet="serves thin-crust pies with seasonal toppings")]),
            "neighborhood":CellValue(value="Glen Park", confidence=0.99,
                               sources=[SourceRef(url="https://sf.eater.com/maps/best-pizza-sf",
                                                  snippet="Gialina in Glen Park")]),
            "price_range": CellValue(value="$20-$25", confidence=0.95,
                               sources=[SourceRef(url="https://sf.eater.com/maps/best-pizza-sf",
                                                  snippet="Average price $20-25")]),
        }
    ),

    # from Tony's own website (different SourceRef, overlapping with a1b2-c3d4 above):
    Entity(
        id    = "m3n4-o5p6",
        cells = {
            "name":            CellValue(value="Tony's Pizza Napoletana", confidence=0.99,
                                   sources=[SourceRef(url="https://www.tonysitaliankitchen.com",
                                                      title="Tony's Pizza Napoletana - North Beach SF",
                                                      snippet="Tony Gemignani, 13-time world pizza champion")]),
            "neighborhood":    CellValue(value="North Beach", confidence=0.99,
                                   sources=[SourceRef(url="https://www.tonysitaliankitchen.com",
                                                      snippet="1570 Stockton Street, North Beach neighborhood")]),
            "price_range":     CellValue(value="$26", confidence=0.97,
                                   sources=[SourceRef(url="https://www.tonysitaliankitchen.com",
                                                      snippet="coal-fired New York-style pizza at $26")]),
            "signature_pizza": CellValue(value="Tony's Special (coal-fired NY-style)",
                                   confidence=0.95,
                                   sources=[SourceRef(url="https://www.tonysitaliankitchen.com",
                                                      snippet="Signature pie: the Tony's Special, a coal-fired New York-style pizza")]),
            # "description" absent — not found on this page
        }
    ),

    # from Mission Pizza page:
    Entity(
        id    = "q7r8-s9t0",
        cells = {
            "name":            CellValue(value="Mission District Pizza Co.", confidence=0.99,
                                   sources=[SourceRef(url="https://missionpizza.com",
                                                      snippet="Authentic Neapolitan pizza baked in a 900 degree wood-fired oven")]),
            "description":     CellValue(value="Authentic Neapolitan pizza in a 900 degree wood-fired oven",
                                   confidence=0.95,
                                   sources=[SourceRef(url="https://missionpizza.com",
                                                      snippet="Authentic Neapolitan pizza baked in a 900 degree wood-fired oven")]),
            "neighborhood":    CellValue(value="Mission District", confidence=0.99,
                                   sources=[SourceRef(url="https://missionpizza.com",
                                                      snippet="3200 16th St in the Mission District")]),
            "price_range":     CellValue(value="$$", confidence=0.80,
                                   sources=[SourceRef(url="https://missionpizza.com",
                                                      snippet="Price range $$")]),
            "signature_pizza": CellValue(value="Diavola with spicy salami", confidence=0.95,
                                   sources=[SourceRef(url="https://missionpizza.com",
                                                      snippet="Signature: the Diavola with spicy salami, $19")]),
        }
    ),

    # ... 9 more Entity objects from the remaining 6 pages
    # (Del Popolo from its own site, Gialina from its own site,
    #  Flour+Water from Thrillist, Penny Roma from Time Out,
    #  Shuggie's from SF Chronicle, Delfina from SF Gate)
]
# Total: ~14 raw entities across all pages, with duplicates
# (Tony's appears twice, Del Popolo appears twice, Gialina appears twice)
# Deduplication happens in Stage 5.
```

**Progress event emitted:**
```json
{"type": "progress", "stage": "extracting", "message": "Extracted 14 raw entities", "progress": 0.70}
```

---

## Stage 5 - Resolver (`pipeline/resolver.py`)

**Function:** `resolve_entities(client, raw_entities)`

14 raw entities came in but several are duplicates — the same restaurant found on
multiple pages. The resolver merges them into one entity per restaurant, keeping the
best data and combining sources from all pages.

### Pass 1 - Fast dedup via `_fast_dedup()`

`_normalise_name()` is applied to every entity name before grouping:

```python
def _normalise_name(name: str) -> str:
    # Step 1: Unicode NFKD normalize
    # "Tony\u2019s" (curly apostrophe) → "Tony's"
    name = unicodedata.normalize("NFKD", name)

    # Step 2: lowercase
    # "Tony's Pizza Napoletana" → "tony's pizza napoletana"

    # Step 3: remove non-word chars (keep alphanumeric + space)
    # "tony's pizza napoletana" → "tonys pizza napoletana"

    # Step 4: collapse whitespace
    # "tonys  pizza  napoletana" → "tonys pizza napoletana"

    # Step 5: strip common business suffixes
    # " inc", " llc", " ltd", " corp", " co" → removed
    return name
```

Examples:
```python
_normalise_name("Tony's Pizza Napoletana")    = "tonys pizza napoletana"
_normalise_name("Tony\u2019s Pizza Napoletana") = "tonys pizza napoletana"  # same key
_normalise_name("Del Popolo")                 = "del popolo"
_normalise_name("Flour + Water Pizzeria")     = "flour  water pizzeria"
_normalise_name("Pizzeria Delfina")           = "pizzeria delfina"
_normalise_name("Delfina")                    = "delfina"
```

The `_fast_dedup()` function builds a dict grouping entities by their normalized name:

```python
groups: dict[str, list[Entity]] = {}

for entity in raw_entities:
    name_cell = entity.cells.get("name")
    # name_cell = CellValue(value="Tony's Pizza Napoletana", ...)

    key = _normalise_name(str(name_cell.value))
    # key = "tonys pizza napoletana"

    if key not in groups:
        groups[key] = []
    groups[key].append(entity)
```

After iterating all 14 entities, `groups` looks like:

```python
groups = {
    "tonys pizza napoletana": [
        Entity(id="a1b2-c3d4", cells={"name": CellValue("Tony's Pizza Napoletana"), "description": ..., "neighborhood": ..., "price_range": ...,               }),  # from Eater SF
        Entity(id="m3n4-o5p6", cells={"name": CellValue("Tony's Pizza Napoletana"), "neighborhood": ..., "price_range": ...,  "signature_pizza": ...,           }),  # from Tony's site
    ],
    "del popolo": [
        Entity(id="e5f6-g7h8", cells={"name": CellValue("Del Popolo"), "description": ..., "neighborhood": ..., "price_range": ...,                            }),  # from Eater SF
        Entity(id="u1v2-w3x4", cells={"name": CellValue("Del Popolo"), "description": ..., "neighborhood": ..., "price_range": ..., "signature_pizza": ...,    }),  # from Del Popolo site
    ],
    "gialina": [
        Entity(id="i9j0-k1l2", cells={"name": CellValue("Gialina"), "description": ..., "neighborhood": ..., "price_range": ...,                               }),  # from Eater SF
        Entity(id="y5z6-a7b8", cells={"name": CellValue("Gialina"), "description": ..., "neighborhood": ..., "price_range": ..., "signature_pizza": ...,       }),  # from Gialina site
    ],
    "mission district pizza co": [
        Entity(id="q7r8-s9t0", cells={"name": CellValue("Mission District Pizza Co."), "description": ..., "neighborhood": ..., "price_range": ..., "signature_pizza": ...}),
    ],
    "flour  water pizzeria": [
        Entity(id="c9d0-e1f2", cells={"name": CellValue("Flour + Water Pizzeria"), "description": ..., "neighborhood": ..., "price_range": ..., "signature_pizza": ...}),
    ],
    "penny roma": [
        Entity(id="g3h4-i5j6", cells={"name": CellValue("Penny Roma"), "description": ..., "neighborhood": ..., "price_range": ...,                            }),
    ],
    "shuggies trash pie": [
        Entity(id="k7l8-m9n0", cells={"name": CellValue("Shuggie's Trash Pie"), "description": ..., "neighborhood": ..., "price_range": ..., "signature_pizza": ...}),
    ],
    "pizzeria delfina": [
        Entity(id="o1p2-q3r4", cells={"name": CellValue("Pizzeria Delfina"), "description": ..., "neighborhood": ..., "price_range": ..., "signature_pizza": ...}),
    ],
}
```

`return list(groups.values())` strips the string keys and returns just the groups.
The normalized string key was only used for matching — it is discarded.

```python
[
    [
        Entity(id="a1b2-c3d4", cells={"name": CellValue("Tony's Pizza Napoletana"), "description": ..., "neighborhood": ..., "price_range": ...,               }),  # from Eater SF
        Entity(id="m3n4-o5p6", cells={"name": CellValue("Tony's Pizza Napoletana"), "neighborhood": ..., "price_range": ...,  "signature_pizza": ...,           }),  # from Tony's site
    ],
    [
        Entity(id="e5f6-g7h8", cells={"name": CellValue("Del Popolo"), "description": ..., "neighborhood": ..., "price_range": ...,                            }),  # from Eater SF
        Entity(id="u1v2-w3x4", cells={"name": CellValue("Del Popolo"), "description": ..., "neighborhood": ..., "price_range": ..., "signature_pizza": ...,    }),  # from Del Popolo site
    ],
    [
        Entity(id="i9j0-k1l2", cells={"name": CellValue("Gialina"), "description": ..., "neighborhood": ..., "price_range": ...,                               }),  # from Eater SF
        Entity(id="y5z6-a7b8", cells={"name": CellValue("Gialina"), "description": ..., "neighborhood": ..., "price_range": ..., "signature_pizza": ...,       }),  # from Gialina site
    ],
    [
        Entity(id="q7r8-s9t0", cells={"name": CellValue("Mission District Pizza Co."), "description": ..., "neighborhood": ..., "price_range": ..., "signature_pizza": ...}),
    ],
    [
        Entity(id="c9d0-e1f2", cells={"name": CellValue("Flour + Water Pizzeria"), "description": ..., "neighborhood": ..., "price_range": ..., "signature_pizza": ...}),
    ],
    [
        Entity(id="g3h4-i5j6", cells={"name": CellValue("Penny Roma"), "description": ..., "neighborhood": ..., "price_range": ...,                            }),
    ],
    [
        Entity(id="k7l8-m9n0", cells={"name": CellValue("Shuggie's Trash Pie"), "description": ..., "neighborhood": ..., "price_range": ..., "signature_pizza": ...}),
    ],
    [
        Entity(id="o1p2-q3r4", cells={"name": CellValue("Pizzeria Delfina"), "description": ..., "neighborhood": ..., "price_range": ..., "signature_pizza": ...}),
    ],
]
```

### Merging each group via `_merge_entity_group()`

For each group, `_merge_entity_group()` is called. Here is the Tony's group in detail:

```python
# Input: two entities for Tony's Pizza Napoletana
# Entity a1b2-c3d4 (from Eater SF) has:   name, description, neighborhood, price_range
# Entity m3n4-o5p6 (from Tony's site) has: name, neighborhood, price_range, signature_pizza

# Step 1 — collect all columns that appear in ANY entity in the group
all_columns = set()
all_columns.update({"name", "description", "neighborhood", "price_range"})       # from Eater entity
all_columns.update({"name", "neighborhood", "price_range", "signature_pizza"})   # from Tony's entity
# all_columns = {"name", "description", "neighborhood", "price_range", "signature_pizza"}

# Step 2 — for each column, gather CellValues from all entities that have it
for col in all_columns:
    candidates = [e.cells[col] for e in entities if col in e.cells]

# col = "description":
#   candidates = [CellValue("Multi-style pizzeria voted best pizza in America", confidence=0.95)]
#   only Eater entity had it

# col = "price_range":
#   candidates = [
#       CellValue("$18-$28", confidence=0.97, sources=[eater_ref]),   # from Eater
#       CellValue("$26",     confidence=0.97, sources=[tonys_ref]),    # from Tony's site
#   ]
#   both entities had it — _merge_cells picks the best

# col = "signature_pizza":
#   candidates = [CellValue("Tony's Special (coal-fired NY-style)", confidence=0.95)]
#   only Tony's site entity had it
```

`_merge_cells()` selects the highest-confidence `CellValue` and aggregates all sources:

```python
def _merge_cells(cells_list: list[CellValue]) -> CellValue:
    best = max(cells_list, key=lambda c: c.confidence)
    # For "price_range": both are 0.97, so first wins: CellValue("$18-$28")

    all_sources = []
    seen_source_urls = set()
    for c in cells_list:
        for src in c.sources:
            if src.url not in seen_source_urls:
                seen_source_urls.add(src.url)
                all_sources.append(src)
    # all_sources = [eater_ref, tonys_ref]  — both URLs kept

    return CellValue(value=best.value, confidence=best.confidence,
                     sources=all_sources, llm_filled=best.llm_filled)
```

### Merged result for Tony's Pizza Napoletana

```python
Entity(
    id    = "a1b2-c3d4",   # reused from the first entity in the group
    cells = {
        "name": CellValue(
            value      = "Tony's Pizza Napoletana",
            confidence = 0.99,
            sources    = [
                SourceRef(url="https://sf.eater.com/maps/best-pizza-sf",
                          snippet="Tony's Pizza Napoletana in North Beach has been voted"),
                SourceRef(url="https://www.tonysitaliankitchen.com",
                          snippet="Tony Gemignani, 13-time world pizza champion"),
            ],
        ),
        "description": CellValue(
            value      = "Multi-style pizzeria voted best pizza in America",
            confidence = 0.95,
            sources    = [SourceRef(url="https://sf.eater.com/maps/best-pizza-sf",
                                    snippet="voted best pizza in America multiple times")],
            # only Eater had description — one source
        ),
        "neighborhood": CellValue(
            value      = "North Beach",
            confidence = 0.99,
            sources    = [
                SourceRef(url="https://sf.eater.com/maps/best-pizza-sf", snippet="in North Beach"),
                SourceRef(url="https://www.tonysitaliankitchen.com",     snippet="1570 Stockton Street, North Beach neighborhood"),
            ],
        ),
        "price_range": CellValue(
            value      = "$18-$28",    # higher confidence won (both tied, first wins)
            confidence = 0.97,
            sources    = [
                SourceRef(url="https://sf.eater.com/maps/best-pizza-sf",
                          snippet="Prices range from $18-$28 for a pie"),
                SourceRef(url="https://www.tonysitaliankitchen.com",
                          snippet="coal-fired New York-style pizza at $26"),
            ],
        ),
        "signature_pizza": CellValue(
            value      = "Tony's Special (coal-fired NY-style)",
            confidence = 0.95,
            sources    = [SourceRef(url="https://www.tonysitaliankitchen.com",
                                    snippet="Signature pie: the Tony's Special, a coal-fired New York-style pizza")],
            # only Tony's site had this — filled the gap left by the Eater entity
        ),
    }
)
```

### Pass 2 - LLM dedup (triggered because > 5 entities)

After Pass 1 there are 8 partially-merged entities. The resolver sends their names
to the LLM to catch near-duplicates that survived Pass 1 — e.g. "Pizzeria Delfina"
and "Delfina" normalize to different keys but are the same place:

```python
# Payload sent to LLM:
[
    {"index": 0, "name": "Tony's Pizza Napoletana"},
    {"index": 1, "name": "Del Popolo"},
    {"index": 2, "name": "Gialina"},
    {"index": 3, "name": "Mission District Pizza Co."},
    {"index": 4, "name": "Flour + Water Pizzeria"},
    {"index": 5, "name": "Penny Roma"},
    {"index": 6, "name": "Shuggie's Trash Pie"},
    {"index": 7, "name": "Pizzeria Delfina"},
]
```

LLM response:

```json
{
  "merge_groups": [[0], [1], [2], [3], [4], [5], [6], [7]]
}
```

All groups are singletons — the LLM found no near-duplicates. All 8 entities survive.

The entities are then sorted alphabetically by name (case-insensitive) and
capped at `MAX_ENTITIES_OUT = 10`:

### Output - `list[Entity]` after Round 1

```python
all_entities = [
    Entity(id="e5f6-g7h8", cells={"name": CellValue("Del Popolo"),               "description": ..., "neighborhood": CellValue("SoMa"),             "price_range": CellValue("$22"),     "signature_pizza": CellValue("Margherita with house-pulled mozzarella")}),
    Entity(id="c9d0-e1f2", cells={"name": CellValue("Flour + Water Pizzeria"),    "description": ..., "neighborhood": CellValue("Mission"),           "price_range": CellValue("$20-$26"), "signature_pizza": CellValue("Taleggio and mushroom pizza")}),
    Entity(id="i9j0-k1l2", cells={"name": CellValue("Gialina"),                   "description": ..., "neighborhood": CellValue("Glen Park"),         "price_range": CellValue("$20-$25"), }),
    Entity(id="q7r8-s9t0", cells={"name": CellValue("Mission District Pizza Co."), "description": ..., "neighborhood": CellValue("Mission District"),  "price_range": CellValue("$$"),      "signature_pizza": CellValue("Diavola with spicy salami")}),
    Entity(id="g3h4-i5j6", cells={"name": CellValue("Penny Roma"),                "description": ..., "neighborhood": CellValue("Mission"),           "price_range": CellValue("$15-$20"), }),
    Entity(id="o1p2-q3r4", cells={"name": CellValue("Pizzeria Delfina"),           "description": ..., "neighborhood": CellValue("Mission"),           "price_range": CellValue("$18-$24"), "signature_pizza": CellValue("Sausage pizza with fennel pollen")}),
    Entity(id="k7l8-m9n0", cells={"name": CellValue("Shuggie's Trash Pie"),        "description": ..., "neighborhood": CellValue("Tenderloin"),        "price_range": CellValue("$16-$22"), "signature_pizza": CellValue("Trash Pie with seasonal scraps")}),
    Entity(id="a1b2-c3d4", cells={"name": CellValue("Tony's Pizza Napoletana"),    "description": ..., "neighborhood": CellValue("North Beach"),        "price_range": CellValue("$18-$28"), "signature_pizza": CellValue("Tony's Special (coal-fired NY-style)")}),
]
# 8 unique entities after Round 1.
# Gialina and Penny Roma are missing "signature_pizza".
```

**Progress event emitted:**
```json
{"type": "progress", "stage": "resolving", "message": "Resolved to 8 unique entities", "progress": 0.88}
```

---

## Stage 6 - Gap Analyzer (`pipeline/gap_analyzer.py`)

**Function:** `analyze_gaps(client, entities, columns, original_query, entity_type)`

Round 1 of 2 is complete. The gap analyzer checks whether coverage is good enough
to stop, or whether a second round of searching is needed.

### Step 1 - Coverage computation (`_compute_coverage()`)

For each column, the function counts what fraction of the 8 entities have a non-null value:

```python
def _compute_coverage(entities, columns):
    coverage = {}
    for col in columns:
        filled = sum(1 for e in entities if col in e.cells and e.cells[col].value is not None)
        coverage[col] = filled / len(entities)
    return coverage

coverage = {
    "name":            8/8 = 1.00,   # all 8 have it
    "description":     6/8 = 0.75,   # Gialina and Penny Roma missing
    "neighborhood":    8/8 = 1.00,   # all 8 have it
    "price_range":     8/8 = 1.00,   # all 8 have it
    "signature_pizza": 6/8 = 0.75,   # Gialina and Penny Roma missing
}
avg_coverage = (1.00 + 0.75 + 1.00 + 1.00 + 0.75) / 5 = 0.90
```

### Step 2 - Early exit check

```python
if avg_coverage >= 0.70 and len(entities) >= 8:
    return {"should_continue": False, ...}
```

`avg_coverage = 0.90 >= 0.70` AND `len(entities) = 8 >= 8` — both conditions met.
However, this check is only a fast path. The gap analyzer also checks whether
`round_idx < request.search_depth - 1` in `main.py` before calling it at all.
Since `round_idx = 0` and `search_depth = 2`, the gap analyzer does run.

In this case it returns early:

```python
gap = {
    "should_continue": False,
    "gap_summary":     "Coverage is sufficient at 90% across all columns with 8 entities.",
    "gap_queries":     [],
    "coverage":        {"name": 1.0, "description": 0.75, "neighborhood": 1.0,
                        "price_range": 1.0, "signature_pizza": 0.75},
}
```

### Step 3 - Loop decision in `main.py`

```python
if gap.get("should_continue") and gap.get("gap_queries"):
    present_query = gap["gap_queries"]   # would feed into Round 2
else:
    break   # gap analyzer said stop — exit the loop early
```

`should_continue` is `False`, so `break` fires. The loop ends after Round 1.
Execution moves directly to Stage 7.

**Progress event emitted:**
```json
{"type": "progress", "stage": "analyzing", "message": "Coverage sufficient - stopping early", "progress": 0.92}
```

---

## Stage 7 - LLM Filler (`pipeline/llm_filler.py`)

**Function:** `llm_fill_gaps(client, entities, columns, entity_type)`

Two entities (Gialina and Penny Roma) still have `signature_pizza` and `description`
missing. The filler uses LLM training knowledge as a last resort to fill remaining nulls.

### Step 1 - Identify gaps

```python
for entity in entities:
    gaps = [col for col in columns
            if col != "name" and col not in entity.cells]
    # "name" is never filled — it must come from the web

# Gialina:    gaps = ["signature_pizza"]
# Penny Roma: gaps = ["signature_pizza"]
# All others: gaps = []  — nothing to fill
```

### Step 2 - Build payload for entities that have gaps

```python
payload = [
    {
        "name":  "Gialina",
        "known": {
            "description": "Thin-crust seasonal pizzas in Glen Park",
            "neighborhood": "Glen Park",
            "price_range":  "$20-$25",
        },
        "fill":  ["signature_pizza"],
    },
    {
        "name":  "Penny Roma",
        "known": {
            "description": "Roman-style pizza al taglio in the Mission",
            "neighborhood": "Mission",
            "price_range":  "$15-$20",
        },
        "fill":  ["signature_pizza"],
    },
]
```

### Step 3 - LLM call

```python
# Prompt sent to LLM:
"""
You are filling missing fields for pizza restaurant entities from your training knowledge.

Rules:
- Fill ONLY the fields listed in 'fill'.
- Use concise values (e.g. '2019', 'San Francisco, CA', 'Series B').
- If you genuinely don't know, use null.
- Do NOT invent plausible-sounding values you are not confident about.

Entities:
[
  {"name": "Gialina",    "known": {"description": "...", "neighborhood": "Glen Park", "price_range": "$20-$25"}, "fill": ["signature_pizza"]},
  {"name": "Penny Roma", "known": {"description": "...", "neighborhood": "Mission",   "price_range": "$15-$20"}, "fill": ["signature_pizza"]}
]

Respond with ONLY a JSON array — one object per entity, same order:
[{"name": "...", "<field>": "<value or null>", ...}, ...]
"""
```

### Step 4 - LLM response

```json
[
  {"name": "Gialina",    "signature_pizza": "Funghi pizza with rotating seasonal mushrooms"},
  {"name": "Penny Roma", "signature_pizza": "Cacio e pepe pizza al taglio"}
]
```

### Step 5 - Cells updated with `llm_filled=True`

```python
# For Gialina:
entities[2].cells["signature_pizza"] = CellValue(
    value      = "Funghi pizza with rotating seasonal mushrooms",
    confidence = 0.5,       # always 0.5 for LLM-filled values
    sources    = [],         # no web source — came from model knowledge
    llm_filled = True,       # flag so UI can mark it differently
)

# For Penny Roma:
entities[4].cells["signature_pizza"] = CellValue(
    value      = "Cacio e pepe pizza al taglio",
    confidence = 0.5,
    sources    = [],
    llm_filled = True,
)
```

**Progress event emitted:**
```json
{"type": "progress", "stage": "filling", "message": "Filled 2 gaps from LLM knowledge", "progress": 0.97}
```

---

## Final Table - `EntityTable`

Built in `main.py` after all stages complete:

```python
result = EntityTable(
    query               = "best pizza restaurants in San Francisco",
    entity_type         = "pizza restaurant",
    columns             = ["name", "description", "neighborhood", "price_range", "signature_pizza"],
    entities            = all_entities,          # 8 entities
    sources_consulted   = list(seen_urls),        # all unique URLs visited
    search_queries_used = all_queries_used,       # 3 queries used in Round 1
    rounds_completed    = 1,                      # loop exited early after Round 1
    created_at          = datetime(2026, 4, 8),
)
```

### `sources_consulted`

```python
sources_consulted = [
    "https://sf.eater.com/maps/best-pizza-sf",
    "https://www.thrillist.com/eat/san-francisco/best-pizza",
    "https://www.sfgate.com/food/best-pizza-sf",
    "https://www.tonysitaliankitchen.com",
    "https://www.yelp.com/search?find_desc=pizza&find_loc=San+Francisco",
    "https://www.tripadvisor.com/Restaurants-g60713-c26-San_Francisco.html",
    "https://www.timeout.com/san-francisco/restaurants/best-pizza",
    "https://missionpizza.com",
    "https://www.delpopolosf.com",
    "https://www.gialina.com",
    "https://www.sfchronicle.com/food/best-pizza-sf",
]
```

### `search_queries_used`

```python
search_queries_used = [
    "best pizza restaurants San Francisco 2024",
    "top rated SF pizza places Yelp TripAdvisor",
    "authentic Neapolitan pizza San Francisco Mission District",
]
```

---

## Final SSE event

```json
{
  "type": "result",
  "data": {
    "query": "best pizza restaurants in San Francisco",
    "entity_type": "pizza restaurant",
    "columns": ["name", "description", "neighborhood", "price_range", "signature_pizza"],
    "entities": [
      {
        "id": "e5f6-g7h8",
        "cells": {
          "name":            {"value": "Del Popolo",                          "confidence": 0.99, "sources": [...], "llm_filled": false},
          "description":     {"value": "Wood-fired minimalist Neapolitan pizza", "confidence": 0.92, "sources": [...], "llm_filled": false},
          "neighborhood":    {"value": "SoMa",                                "confidence": 0.99, "sources": [...], "llm_filled": false},
          "price_range":     {"value": "$22",                                 "confidence": 0.95, "sources": [...], "llm_filled": false},
          "signature_pizza": {"value": "Margherita with house-pulled mozzarella", "confidence": 0.95, "sources": [...], "llm_filled": false}
        }
      },
      {
        "id": "i9j0-k1l2",
        "cells": {
          "name":            {"value": "Gialina",                             "confidence": 0.99, "sources": [...], "llm_filled": false},
          "description":     {"value": "Thin-crust seasonal pizzas in Glen Park", "confidence": 0.90, "sources": [...], "llm_filled": false},
          "neighborhood":    {"value": "Glen Park",                           "confidence": 0.99, "sources": [...], "llm_filled": false},
          "price_range":     {"value": "$20-$25",                             "confidence": 0.95, "sources": [...], "llm_filled": false},
          "signature_pizza": {"value": "Funghi pizza with rotating seasonal mushrooms", "confidence": 0.5, "sources": [], "llm_filled": true}
        }
      }
      // ... 6 more entities
    ],
    "sources_consulted": ["https://sf.eater.com/...", "https://www.tonysitaliankitchen.com", ...],
    "search_queries_used": [
      "best pizza restaurants San Francisco 2024",
      "top rated SF pizza places Yelp TripAdvisor",
      "authentic Neapolitan pizza San Francisco Mission District"
    ],
    "rounds_completed": 1,
    "created_at": "2026-04-08T12:00:00"
  }
}
```

---

## UI Rendering

The frontend receives the SSE stream and renders two things: a live progress bar
during processing, and a final table once the result event arrives.

### Progress bar (during processing)

Each `progress` event carries a `stage`, `message`, and `progress` float (0.0-1.0).
The frontend maps these to a visual progress bar:

```
[planning   ]  0.10  "Search plan ready"
[searching  ]  0.22  "Found 11 new URLs"
[scraping   ]  0.45  "Scraped 9/11 pages successfully"
[extracting ]  0.70  "Extracted 14 raw entities"
[resolving  ]  0.88  "Resolved to 8 unique entities"
[analyzing  ]  0.92  "Coverage sufficient - stopping early"
[filling    ]  0.97  "Filled 2 gaps from LLM knowledge"
[done       ]  1.00
```

### Final rendered table

Once the `result` event arrives, the frontend builds a table from `columns` and `entities`:

| name | description | neighborhood | price_range | signature_pizza |
|---|---|---|---|---|
| Del Popolo | Wood-fired minimalist Neapolitan pizza | SoMa | $22 | Margherita with house-pulled mozzarella |
| Flour + Water Pizzeria | Seasonal wood-fired pies with local produce | Mission | $20-$26 | Taleggio and mushroom pizza |
| Gialina | Thin-crust seasonal pizzas in Glen Park | Glen Park | $20-$25 | Funghi pizza with rotating seasonal mushrooms* |
| Mission District Pizza Co. | Authentic Neapolitan pizza in 900 degree wood-fired oven | Mission District | $$ | Diavola with spicy salami |
| Penny Roma | Roman-style pizza al taglio in the Mission | Mission | $15-$20 | Cacio e pepe pizza al taglio* |
| Pizzeria Delfina | Beloved Mission neighborhood pizza institution | Mission | $18-$24 | Sausage pizza with fennel pollen |
| Shuggie's Trash Pie | Creative upcycled-ingredient pizza | Tenderloin | $16-$22 | Trash Pie with seasonal scraps |
| Tony's Pizza Napoletana | Multi-style pizzeria voted best pizza in America | North Beach | $18-$28 | Tony's Special (coal-fired NY-style) |

`*` = value filled from LLM training knowledge (`llm_filled: true`, `confidence: 0.5`)

### Per-cell source tooltip

Every non-LLM-filled cell is clickable in the UI. Clicking reveals the source attribution
stored in `CellValue.sources`:

```
Cell: "North Beach"  (Tony's Pizza Napoletana / neighborhood)

Sources:
  [1] sf.eater.com/maps/best-pizza-sf
      "Tony's Pizza Napoletana in North Beach has been voted"

  [2] tonysitaliankitchen.com
      "1570 Stockton Street, North Beach neighborhood"
```

LLM-filled cells show a different indicator:

```
Cell: "Funghi pizza with rotating seasonal mushrooms"  (Gialina / signature_pizza)

  Filled from LLM training knowledge
  Confidence: 0.5 — no web source available
```

---

## Summary of data flow

```
query: str
  plan_search()
SearchPlan { entity_type, columns[5], search_queries[3] }

  fetch_web_results()   x3 queries, 4 results each, asyncio.gather
results_per_query: list[list[SearchResult]]
  dedup by seen_urls
new_results: list[SearchResult] { url, title, snippet }

  scrape_urls()   semaphore=6 concurrent fetches, _clean_html()
list[ScrapedPage] { url, title, content max 8KB, error }

  extract_from_pages()   semaphore=8 concurrent LLM calls, temperature=0.1
list[Entity] raw, may have duplicates — one Entity per restaurant per page

  resolve_entities()   _fast_dedup() then LLM dedup if > 5 entities
list[Entity] deduplicated, merged sources, sorted, max 10

  analyze_gaps()   _compute_coverage(), LLM if coverage < 70% or entities < 8
  if should_continue=True: present_query = gap_queries, loop back to fetch_web_results()
  if should_continue=False: break out of loop

  llm_fill_gaps()   fills remaining null cells, llm_filled=True, confidence=0.5
list[Entity] complete

  EntityTable built
EntityTable { query, entity_type, columns, entities, sources_consulted,
              search_queries_used, rounds_completed, created_at }

  SSE result event streamed to frontend
UI renders progress bar during processing, final table on result event
Each cell links back to source URL and snippet, LLM-filled cells marked with *
```
