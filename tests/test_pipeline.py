# These tests use mocked HTTP and API responses - no real API keys needed.
# Run with: pytest tests/ -v

import asyncio, json, pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Planner
def _planner_client(payload):
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_llm_reply(payload))
    return mock_client


@pytest.mark.asyncio
async def test_planner_parse():
    from backend.pipeline.planner import plan_search

    client = _planner_client({
        "entity_type": "AI healthcare startups",
        "columns": ["name", "focus_area", "key_product", "funding", "headquarters", "highlights"],
        "search_queries": ["top AI healthcare startups", "healthcare AI companies list", "best health AI startups"],
        "rationale": "Topic attributes",
    })
    plan = await plan_search(client, "AI startups in healthcare")

    assert plan.entity_type == "AI healthcare startups"
    assert plan.columns == ["name", "focus_area", "key_product", "funding", "headquarters", "highlights"]
    assert len(plan.search_queries) == 3


@pytest.mark.asyncio
async def test_planner_name_always_first():
    from backend.pipeline.planner import plan_search

    client = _planner_client({"entity_type": "pizza", "columns": ["pizza_style", "Name", "neighborhood"], "search_queries": ["q"]})
    plan = await plan_search(client, "pizza places Brooklyn")
    assert plan.columns == ["name", "pizza_style", "neighborhood"]


@pytest.mark.asyncio
async def test_planner_drops_lookup_columns_and_cleans_names():
    from backend.pipeline.planner import plan_search

    client = _planner_client({"entity_type": "pizza", "search_queries": ["q"],
                              "columns": ["name", "Pizza Style", "phone", "Address", "opening_hours", "price range", "pizza_style"]})
    plan = await plan_search(client, "pizza places Brooklyn")
    assert plan.columns == ["name", "pizza_style", "price_range"]


@pytest.mark.asyncio
async def test_planner_caps_columns_at_eight():
    from backend.pipeline.planner import plan_search

    client = _planner_client({"entity_type": "x", "search_queries": ["q"], "columns": ["name"] + [f"col_{i}" for i in range(12)]})
    plan = await plan_search(client, "anything")
    assert len(plan.columns) == 8


@pytest.mark.asyncio
async def test_planner_pads_queries_to_three_without_repeats():
    from backend.pipeline.planner import plan_search

    client = _planner_client({"entity_type": "x", "columns": ["name", "a", "b"], "search_queries": ["pizza Brooklyn"]})
    plan = await plan_search(client, "pizza Brooklyn")
    assert plan.search_queries == ["pizza Brooklyn", "top pizza Brooklyn", "best pizza Brooklyn"]


@pytest.mark.asyncio
async def test_planner_falls_back_on_api_failure():
    from backend.pipeline.planner import plan_search

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("429"))
    plan = await plan_search(client, "pizza Brooklyn")

    assert plan.columns[0] == "name"
    assert "highlights" in plan.columns
    assert len(plan.search_queries) == 3
    assert not any("2024" in q for q in plan.search_queries)


@pytest.mark.asyncio
async def test_planner_falls_back_when_too_few_columns():
    from backend.pipeline.planner import plan_search

    client = _planner_client({"entity_type": "x", "columns": ["name", "phone"], "search_queries": ["q"]})
    plan = await plan_search(client, "pizza Brooklyn")
    assert plan.rationale == "Fallback plan."


LONG_TEXT = "Hippocratic AI, a Palo Alto startup building a safety-focused LLM for healthcare, was founded in 2023."


def test_retriever_prefers_raw_content():
    from backend.pipeline.retriever import _to_page

    page = _to_page({"url": "https://a.com", "title": "A", "raw_content": LONG_TEXT, "content": "short snippet"})
    assert page.content == LONG_TEXT


def test_retriever_falls_back_to_snippet():
    from backend.pipeline.retriever import _to_page

    page = _to_page({"url": "https://a.com", "title": "A", "raw_content": None, "content": LONG_TEXT})
    assert page.content == LONG_TEXT


def test_retriever_skips_empty_pages():
    from backend.pipeline.retriever import _to_page

    assert _to_page({"url": "https://a.com", "raw_content": "", "content": "too short"}) is None
    assert _to_page({"url": "", "raw_content": LONG_TEXT}) is None


def test_retriever_cuts_long_pages():
    from backend.pipeline.retriever import _to_page, MAX_CONTENT_CHARS

    page = _to_page({"url": "https://a.com", "raw_content": "x" * (MAX_CONTENT_CHARS * 2)})
    assert len(page.content) == MAX_CONTENT_CHARS


@pytest.mark.asyncio
async def test_retriever_dedupes_and_skips_seen_urls():
    from backend.pipeline import retriever

    def result(url):
        return {"url": url, "title": url, "raw_content": LONG_TEXT}

    mock_client = AsyncMock()
    mock_client.search = AsyncMock(side_effect=[
        {"results": [result("https://a.com"), result("https://b.com")]},
        {"results": [result("https://b.com"), result("https://c.com")]},
    ])
    with patch.object(retriever, "_client", return_value=mock_client):
        pages = await retriever.retrieve(["q1", "q2"], seen_urls={"https://a.com"})

    assert [p.url for p in pages] == ["https://b.com", "https://c.com"]


@pytest.mark.asyncio
async def test_retriever_survives_one_failed_query():
    from backend.pipeline import retriever

    mock_client = AsyncMock()
    mock_client.search = AsyncMock(side_effect=[
        RuntimeError("rate limited"),
        {"results": [{"url": "https://a.com", "raw_content": LONG_TEXT}]},
    ])
    with patch.object(retriever, "_client", return_value=mock_client):
        pages = await retriever.retrieve(["q1", "q2"], seen_urls=set())

    assert [p.url for p in pages] == ["https://a.com"]


def test_normalize_url_merges_copies_of_one_page():
    from backend.pipeline.retriever import normalize_url

    copies = [
        "https://techcrunch.com/hippocratic-ai-raises",
        "http://techcrunch.com/hippocratic-ai-raises/",
        "https://www.TechCrunch.com/hippocratic-ai-raises?utm_source=twitter&utm_medium=social",
        "https://techcrunch.com/hippocratic-ai-raises#comments",
        "https://techcrunch.com/hippocratic-ai-raises?fbclid=abc123",
    ]
    assert len({normalize_url(u) for u in copies}) == 1


def test_normalize_url_keeps_real_query_params():
    from backend.pipeline.retriever import normalize_url

    assert normalize_url("https://a.com/item?id=1") != normalize_url("https://a.com/item?id=2")
    assert normalize_url("https://a.com/item?id=1&utm_source=x") == normalize_url("https://a.com/item?id=1")


def test_normalize_url_keeps_path_case():
    from backend.pipeline.retriever import normalize_url

    assert normalize_url("https://a.com/Docs") != normalize_url("https://a.com/docs")
    assert normalize_url("https://a.com") == normalize_url("https://a.com/")


@pytest.mark.asyncio
async def test_retriever_skips_near_duplicate_urls():
    from backend.pipeline import retriever

    mock_client = AsyncMock()
    mock_client.search = AsyncMock(side_effect=[
        {"results": [{"url": "https://a.com/page", "raw_content": LONG_TEXT}]},
        {"results": [
            {"url": "http://www.a.com/page/?utm_source=x", "raw_content": LONG_TEXT},
            {"url": "https://b.com/page", "raw_content": LONG_TEXT},
        ]},
    ])
    with patch.object(retriever, "_client", return_value=mock_client):
        pages = await retriever.retrieve(["q1", "q2"], seen_urls={"https://c.com/"})

    assert [p.url for p in pages] == ["https://a.com/page", "https://b.com/page"]


@pytest.mark.asyncio
async def test_retriever_skips_seen_urls_written_differently():
    from backend.pipeline import retriever

    mock_client = AsyncMock()
    mock_client.search = AsyncMock(return_value={"results": [{"url": "https://www.a.com/page/", "raw_content": LONG_TEXT}]})
    with patch.object(retriever, "_client", return_value=mock_client):
        pages = await retriever.retrieve(["q1"], seen_urls={"https://a.com/page"})

    assert pages == []


@pytest.mark.asyncio
async def test_retriever_excludes_low_value_domains():
    from backend.pipeline import retriever

    mock_client = AsyncMock()
    mock_client.search = AsyncMock(return_value={"results": []})
    with patch.object(retriever, "_client", return_value=mock_client):
        await retriever.retrieve(["q1"], seen_urls=set())

    assert mock_client.search.call_args.kwargs["exclude_domains"] == retriever.EXCLUDED_DOMAINS
    assert "youtube.com" in retriever.EXCLUDED_DOMAINS


@pytest.mark.asyncio
async def test_retriever_raises_when_all_queries_fail():
    from backend.pipeline import retriever

    mock_client = AsyncMock()
    mock_client.search = AsyncMock(side_effect=RuntimeError("bad key"))
    with patch.object(retriever, "_client", return_value=mock_client):
        with pytest.raises(RuntimeError):
            await retriever.retrieve(["q1", "q2"], seen_urls=set())


@pytest.mark.asyncio
async def test_retrieve_with_backup_skips_backup_when_enough_pages():
    from backend.pipeline import retriever

    pages = [retriever.ScrapedPage(url=f"https://{c}.com", content=LONG_TEXT) for c in "abcd"]
    with patch.object(retriever, "retrieve", AsyncMock(return_value=pages)) as mock_retrieve:
        got, used = await retriever.retrieve_with_backup(["q1", "q2", "q3"], set())

    assert len(got) == 4
    assert used == ["q1", "q2"]
    mock_retrieve.assert_called_once()


@pytest.mark.asyncio
async def test_retrieve_with_backup_runs_backup_when_few_pages():
    from backend.pipeline import retriever

    first = [retriever.ScrapedPage(url="https://a.com", content=LONG_TEXT)]
    second = [retriever.ScrapedPage(url="https://b.com", content=LONG_TEXT)]
    with patch.object(retriever, "retrieve", AsyncMock(side_effect=[first, second])) as mock_retrieve:
        got, used = await retriever.retrieve_with_backup(["q1", "q2", "q3"], set())

    assert [p.url for p in got] == ["https://a.com", "https://b.com"]
    assert used == ["q1", "q2", "q3"]
    assert mock_retrieve.call_args_list[1].args[0] == ["q3"]
    assert "https://a.com" in mock_retrieve.call_args_list[1].args[1]


def _llm_reply(payload):
    choice = MagicMock()
    choice.message.content = json.dumps(payload)
    response = MagicMock()
    response.choices = [choice]
    return response


def _pages(n):
    from backend.models import ScrapedPage
    return [ScrapedPage(url=f"https://p{i}.com", title=f"P{i}", content=LONG_TEXT) for i in range(1, n + 1)]


@pytest.mark.asyncio
async def test_extractor_groups_pages_into_batches():
    from backend.pipeline import extractor

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_llm_reply({"entities": []}))
    await extractor.extract_from_pages(mock_client, _pages(7), ["name"], "startups")

    assert mock_client.chat.completions.create.call_count == 3


@pytest.mark.asyncio
async def test_extractor_links_entity_to_its_page():
    from backend.pipeline import extractor

    reply = {"entities": [
        {"page": 1, "name": {"value": "Alpha", "confidence": 1.0, "snippet": "Alpha"}},
        {"page": 2, "name": {"value": "Beta", "confidence": 1.0, "snippet": "Beta"}},
    ]}
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_llm_reply(reply))
    entities = await extractor.extract_from_pages(mock_client, _pages(2), ["name"], "startups")

    sources = {e.get_name(): e.cells["name"].sources[0].url for e in entities}
    assert sources == {"Alpha": "https://p1.com", "Beta": "https://p2.com"}


@pytest.mark.asyncio
async def test_extractor_drops_bad_page_numbers():
    from backend.pipeline import extractor

    reply = {"entities": [
        {"page": 5, "name": {"value": "Ghost"}},
        {"page": "x", "name": {"value": "Typo"}},
        {"name": {"value": "NoPage"}},
        {"page": 1, "name": {"value": "Real"}},
    ]}
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_llm_reply(reply))
    entities = await extractor.extract_from_pages(mock_client, _pages(2), ["name"], "startups")

    assert [e.get_name() for e in entities] == ["Real"]


@pytest.mark.asyncio
async def test_extractor_keeps_other_batches_when_one_fails():
    from backend.pipeline import extractor

    reply = {"entities": [{"page": 1, "name": {"value": "Alpha"}}]}
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=[RuntimeError("429"), _llm_reply(reply)])
    entities = await extractor.extract_from_pages(mock_client, _pages(4), ["name"], "startups")

    assert [e.get_name() for e in entities] == ["Alpha"]


@pytest.mark.asyncio
async def test_extractor_skips_short_pages():
    from backend.pipeline import extractor
    from backend.models import ScrapedPage

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_llm_reply({"entities": []}))
    await extractor.extract_from_pages(mock_client, [ScrapedPage(url="https://a.com", content="short")], ["name"], "startups")

    mock_client.chat.completions.create.assert_not_called()


# Resolver
def test_normalise_name(): # Name normalisation should collapse common variations
    from backend.pipeline.resolver import _normalise_name

    assert _normalise_name("OpenAI Inc.") == _normalise_name("openai inc")
    assert _normalise_name("Tempus AI, LLC") == _normalise_name("tempus ai")
    assert _normalise_name("Google LLC") == _normalise_name("google")


def test_fast_dedup_merges_same_names(): # Fast dedup should merge entities with the same normalised name
    from backend.pipeline.resolver import _fast_dedup, _merge_entity_group
    from backend.models import Entity, CellValue, SourceRef
    import uuid

    make_entity = lambda name, val, url: Entity(
        id=str(uuid.uuid4()),
        cells={
            "name": CellValue(value=name, confidence=0.99,
                              sources=[SourceRef(url=url, title="", snippet="")]),
            "description": CellValue(value=val, confidence=0.8,
                                     sources=[SourceRef(url=url, title="", snippet="")]),
        }
    )

    entities = [
        make_entity("OpenAI", "AI research lab", "https://page1.com"),
        make_entity("OpenAI Inc", "Maker of GPT-4", "https://page2.com"),
        make_entity("Anthropic", "AI safety company", "https://page3.com"),
    ]

    groups = _fast_dedup(entities)
    # Should have 2 groups: OpenAI (2 variants) + Anthropic
    assert len(groups) == 2
    merged = [_merge_entity_group(g) for g in groups]
    names = {str(e.cells["name"].value) for e in merged}
    assert "Anthropic" in names


def _entity(name, url="https://p1.com", **cols):
    from backend.models import Entity, CellValue, SourceRef
    import uuid
    cells = {"name": CellValue(value=name, sources=[SourceRef(url=url)])}
    for col, val in cols.items():
        cells[col] = CellValue(value=val, sources=[SourceRef(url=url)])
    return Entity(id=str(uuid.uuid4()), cells=cells)


def _topic_pages():
    from backend.models import ScrapedPage
    filler = "\n".join(f"Healthcare AI startup number {i} is building health tools for hospitals." for i in range(20))
    return [ScrapedPage(url="https://list.com", content=filler + "\nHippocratic AI builds safety focused models for nurses.")]


def test_normalise_name_handles_the_and_ampersand():
    from backend.pipeline.resolver import _normalise_name

    assert _normalise_name("The Hippocratic AI, Inc.") == "hippocratic ai"
    assert _normalise_name("Johnson & Johnson") == "johnson and johnson"
    assert _normalise_name("Co") == "co"


def test_resolver_merges_spacing_variants():
    from backend.pipeline.resolver import resolve_entities

    out = resolve_entities([_entity("OpenAI"), _entity("Open AI", "https://p2.com")], [], "")
    assert len(out) == 1


def test_resolver_merges_names_sharing_a_rare_word():
    from backend.pipeline.resolver import resolve_entities

    out = resolve_entities(
        [_entity("Hippocratic AI"), _entity("Hippocratic", "https://p2.com")],
        _topic_pages(), "AI healthcare startups",
    )
    assert len(out) == 1


def test_resolver_keeps_names_sharing_only_a_topic_word():
    from backend.pipeline.resolver import resolve_entities

    out = resolve_entities(
        [_entity("Bayesian Health"), _entity("Health Inc", "https://p2.com")],
        _topic_pages(), "AI healthcare startups",
    )
    assert len(out) == 2


def test_resolver_different_domains_block_merge():
    from backend.pipeline.resolver import resolve_entities

    out = resolve_entities(
        [_entity("Acme Health", website="acme.com"), _entity("Acme Healths", "https://p2.com", website="https://acmehealth.io")],
        [], "",
    )
    assert len(out) == 2


def test_resolver_same_domain_merges_longer_name():
    from backend.pipeline.resolver import resolve_entities

    out = resolve_entities(
        [_entity("Stripe", website="stripe.com"), _entity("Stripe Payments", "https://p2.com", website="https://www.stripe.com/")],
        [], "",
    )
    assert len(out) == 1


def test_resolver_merge_is_transitive():
    from backend.pipeline.resolver import resolve_entities

    out = resolve_entities(
        [_entity("Hippocratic AI", website="hippocratic.ai"), _entity("Hippocratic", "https://p2.com"),
         _entity("Hipocratic", "https://p3.com")],
        _topic_pages(), "AI healthcare startups",
    )
    assert len(out) == 1


def test_resolver_cites_only_sources_that_agree():
    from backend.pipeline.resolver import resolve_entities

    out = resolve_entities(
        [_entity("Alpha", "https://p1.com", founded="2019"), _entity("Alpha", "https://p2.com", founded="2019"),
         _entity("Alpha", "https://p3.com", founded="2020")],
        [], "",
    )
    cell = out[0].cells["founded"]
    assert cell.value == "2019"
    assert {src.url for src in cell.sources} == {"https://p1.com", "https://p2.com"}


def test_resolver_orders_by_source_count_not_alphabet():
    from backend.pipeline.resolver import resolve_entities

    out = resolve_entities(
        [_entity("Alpha"), _entity("Zebra", "https://p1.com"), _entity("Zebra", "https://p2.com")],
        [], "",
    )
    assert [e.get_name() for e in out] == ["Zebra", "Alpha"]


def test_resolver_keeps_more_than_ten():
    from backend.pipeline.resolver import resolve_entities

    names = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India", "Juliet", "Kilo", "Lima"]
    out = resolve_entities([_entity(n) for n in names], [], "")
    assert len(out) == 12


class _FakeEmbedder:
    vocab = ["hippocratic", "founded", "2023", "palo", "alto", "pizza", "brooklyn", "funding"]

    def _vec(self, text):
        words = text.lower().split()
        return [float(sum(w.startswith(v) for w in words)) + 0.01 for v in self.vocab]

    def embed(self, texts):
        return [self._vec(t) for t in texts]

    def query_embed(self, texts):
        return [self._vec(t) for t in texts]


def test_chunk_pages_splits_long_page_with_overlap():
    from backend.pipeline.embedder import chunk_pages, CHUNK_WORDS, OVERLAP_WORDS
    from backend.models import ScrapedPage

    words = [f"w{i}" for i in range(200)]
    chunks = chunk_pages([ScrapedPage(url="https://a.com", content=" ".join(words))])

    assert len(chunks) == 3
    assert all(c.url == "https://a.com" for c in chunks)
    assert chunks[1].text.split()[0] == f"w{CHUNK_WORDS - OVERLAP_WORDS}"
    assert chunks[-1].text.split()[-1] == "w199"


def test_chunk_pages_skips_tiny_pages():
    from backend.pipeline.embedder import chunk_pages
    from backend.models import ScrapedPage

    assert chunk_pages([ScrapedPage(url="https://a.com", content="too short")]) == []


def test_chunk_index_finds_the_right_page():
    from backend.pipeline.embedder import embed_pages, ChunkIndex
    from backend.models import ScrapedPage

    fake = _FakeEmbedder()
    pages = [
        ScrapedPage(url="https://health.com", content="Hippocratic AI was founded in 2023 in Palo Alto and builds models for nurses and hospitals."),
        ScrapedPage(url="https://food.com", content="The best pizza in Brooklyn is found at small family shops near the river every night."),
    ]
    chunks, vectors = embed_pages(pages, model=fake)
    index = ChunkIndex(chunks, [vectors], model=fake)

    hits = index.search(["Hippocratic founded", "Brooklyn pizza"], k=1)
    assert hits[0][0][0].url == "https://health.com"
    assert hits[1][0][0].url == "https://food.com"


def test_chunk_index_empty_is_safe():
    from backend.pipeline.embedder import ChunkIndex

    index = ChunkIndex([], [], model=_FakeEmbedder())
    assert len(index) == 0
    assert index.search(["anything"]) == [[]]


class _FakeIndex:
    def __init__(self, hits_by_entity):
        self.hits_by_entity = hits_by_entity
        self.queries = []

    def __len__(self):
        return 1

    def search(self, queries, k=3):
        self.queries.extend(queries)
        return [next((h for name, h in self.hits_by_entity.items() if q.startswith(name)), []) for q in queries]


HIPPO_TEXT = "Hippocratic AI was established in 2023 and is headquartered in Palo Alto, California."


def _chunk(text=HIPPO_TEXT, url="https://hippo.com"):
    from backend.pipeline.embedder import Chunk
    return Chunk(url=url, title="Hippo", text=text)


def _fill_client(cells):
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_llm_reply({"cells": cells}))
    return mock_client


def test_find_gaps_skips_name_and_filled_cells():
    from backend.pipeline.filler import find_gaps

    gaps = find_gaps([_entity("Hippocratic AI", founded="2023")], ["name", "founded", "headquarters"])
    assert [(g.entity.get_name(), g.column) for g in gaps] == [("Hippocratic AI", "headquarters")]


@pytest.mark.asyncio
async def test_fill_puts_value_with_source_and_quote():
    from backend.pipeline.filler import fill_from_chunks

    entity = _entity("Hippocratic AI")
    client = _fill_client([{"id": 0, "value": "2023", "chunk": "c1", "quote": "was established in 2023"}])
    out, filled = await fill_from_chunks(client, [entity], ["name", "founded"], "startups",
                                         _FakeIndex({"Hippocratic AI": [(_chunk(), 0.9)]}))

    cell = out[0].cells["founded"]
    assert filled == 1
    assert cell.value == "2023"
    assert cell.sources[0].url == "https://hippo.com"
    assert cell.sources[0].snippet == "was established in 2023"
    assert cell.llm_filled is False


@pytest.mark.asyncio
async def test_fill_drops_value_when_quote_not_in_chunk():
    from backend.pipeline.filler import fill_from_chunks

    client = _fill_client([{"id": 0, "value": "2019", "chunk": "c1", "quote": "was founded in 2019"}])
    out, filled = await fill_from_chunks(client, [_entity("Hippocratic AI")], ["name", "founded"], "startups",
                                         _FakeIndex({"Hippocratic AI": [(_chunk(), 0.9)]}))

    assert filled == 0
    assert "founded" not in out[0].cells


@pytest.mark.asyncio
async def test_fill_drops_value_not_inside_its_quote():
    from backend.pipeline.filler import fill_from_chunks

    client = _fill_client([{"id": 0, "value": "2024", "chunk": "c1", "quote": "was established in 2023"}])
    _, filled = await fill_from_chunks(client, [_entity("Hippocratic AI")], ["name", "founded"], "startups",
                                       _FakeIndex({"Hippocratic AI": [(_chunk(), 0.9)]}))
    assert filled == 0


@pytest.mark.asyncio
async def test_fill_skips_chunks_that_do_not_name_the_entity():
    from backend.pipeline.filler import fill_from_chunks

    other = _chunk("Bayesian Health was established in 2018 in New York.", "https://bayes.com")
    client = _fill_client([])
    _, filled = await fill_from_chunks(client, [_entity("Hippocratic AI")], ["name", "founded"], "startups",
                                       _FakeIndex({"Hippocratic AI": [(other, 0.95)]}))

    assert filled == 0
    client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_fill_skips_low_score_chunks():
    from backend.pipeline.filler import fill_from_chunks

    client = _fill_client([])
    await fill_from_chunks(client, [_entity("Hippocratic AI")], ["name", "founded"], "startups",
                           _FakeIndex({"Hippocratic AI": [(_chunk(), 0.3)]}))
    client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_fill_rejects_chunk_not_offered_to_that_cell():
    from backend.pipeline.filler import fill_from_chunks

    client = _fill_client([{"id": 0, "value": "2023", "chunk": "c9", "quote": "was established in 2023"}])
    _, filled = await fill_from_chunks(client, [_entity("Hippocratic AI")], ["name", "founded"], "startups",
                                       _FakeIndex({"Hippocratic AI": [(_chunk(), 0.9)]}))
    assert filled == 0


@pytest.mark.asyncio
async def test_fill_ignores_bad_cell_ids_and_nulls():
    from backend.pipeline.filler import fill_from_chunks

    client = _fill_client([
        {"id": 7, "value": "2023", "chunk": "c1", "quote": "was established in 2023"},
        {"id": "0", "value": "2023", "chunk": "c1", "quote": "was established in 2023"},
        {"id": 0, "value": None, "chunk": None, "quote": None},
    ])
    _, filled = await fill_from_chunks(client, [_entity("Hippocratic AI")], ["name", "founded"], "startups",
                                       _FakeIndex({"Hippocratic AI": [(_chunk(), 0.9)]}))
    assert filled == 0


@pytest.mark.asyncio
async def test_fill_sends_shared_chunk_once():
    from backend.pipeline.filler import fill_from_chunks

    client = _fill_client([
        {"id": 0, "value": "2023", "chunk": "c1", "quote": "was established in 2023"},
        {"id": 1, "value": "Palo Alto, California", "chunk": "c1", "quote": "headquartered in Palo Alto, California"},
    ])
    out, filled = await fill_from_chunks(client, [_entity("Hippocratic AI")], ["name", "founded", "headquarters"], "startups",
                                         _FakeIndex({"Hippocratic AI": [(_chunk(), 0.9)]}))

    prompt = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert prompt.count(HIPPO_TEXT) == 1
    assert filled == 2
    assert out[0].cells["headquarters"].value == "Palo Alto, California"


@pytest.mark.asyncio
async def test_fill_survives_api_failure():
    from backend.pipeline.filler import fill_from_chunks

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("429"))
    out, filled = await fill_from_chunks(client, [_entity("Hippocratic AI")], ["name", "founded"], "startups",
                                         _FakeIndex({"Hippocratic AI": [(_chunk(), 0.9)]}))
    assert filled == 0
    assert "founded" not in out[0].cells


@pytest.mark.asyncio
async def test_fill_does_nothing_without_index_or_gaps():
    from backend.pipeline.filler import fill_from_chunks

    client = _fill_client([])
    await fill_from_chunks(client, [_entity("Hippocratic AI")], ["name", "founded"], "startups", None)
    await fill_from_chunks(client, [_entity("Hippocratic AI", founded="2023")], ["name", "founded"], "startups",
                           _FakeIndex({"Hippocratic AI": [(_chunk(), 0.9)]}))
    client.chat.completions.create.assert_not_called()


# Gap Analyser
def test_coverage_computation(): # Coverage should be 0 when no entities have a value for a column
    from backend.pipeline.gap_analyzer import _compute_coverage
    from backend.models import Entity, CellValue, SourceRef
    import uuid

    entities = [
        Entity(id=str(uuid.uuid4()), cells={
            "name": CellValue(value="A", confidence=1.0, sources=[]),
            # no founded_year
        }),
        Entity(id=str(uuid.uuid4()), cells={
            "name": CellValue(value="B", confidence=1.0, sources=[]),
            "founded_year": CellValue(value="2020", confidence=0.9, sources=[]),
        }),
    ]

    cov = _compute_coverage(entities, ["name", "founded_year"])
    assert cov["name"] == 1.0
    assert cov["founded_year"] == 0.5


# Models
def test_entity_coverage(): # Entity.coverage() should return fraction of filled columns
    from backend.models import Entity, CellValue
    import uuid

    e = Entity(
        id=str(uuid.uuid4()),
        cells={
            "name": CellValue(value="Test Co", confidence=1.0, sources=[]),
            "description": CellValue(value="A company", confidence=0.9, sources=[]),
        }
    )
    columns = ["name", "description", "founded_year", "headquarters"]
    assert e.coverage(columns) == 0.5  # 2 of 4 filled


def test_cell_value_display(): # CellValue.display_value should handle None gracefully
    from backend.models import CellValue

    assert CellValue(value=None, sources=[]).display_value == ""
    assert CellValue(value="2020", sources=[]).display_value == "2020"
    assert CellValue(value=42, sources=[]).display_value == "42"


def test_cell_value_llm_filled_default():# CellValue.llm_filled should default to False
    from backend.models import CellValue

    c = CellValue(value="Series B", confidence=0.9, sources=[])
    assert c.llm_filled is False


# LLM Filler (Stage 7)
@pytest.mark.asyncio
async def test_llm_filler_fills_missing_cells(): # LLM filler should populate null cells and mark them llm_filled=True
    from backend.pipeline.llm_filler import llm_fill_gaps
    from backend.models import Entity, CellValue
    import uuid

    entity = Entity(
        id=str(uuid.uuid4()),
        cells={
            "name": CellValue(value="Abridge", confidence=1.0, sources=[]),
            "description": CellValue(value="Clinical documentation AI", confidence=0.9, sources=[]),
            # founded_year and headquarters deliberately missing
        }
    )

    llm_response = json.dumps([
        {"name": "Abridge", "founded_year": "2018", "headquarters": "Pittsburgh, PA"}
    ])

    mock_choice = MagicMock()
    mock_choice.message.content = llm_response

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    columns = ["name", "description", "founded_year", "headquarters"]
    result = await llm_fill_gaps(mock_client, [entity], columns, "AI startups")

    assert len(result) == 1
    e = result[0]
    assert e.cells["founded_year"].value == "2018"
    assert e.cells["founded_year"].llm_filled is True
    assert e.cells["founded_year"].confidence == 0.5
    assert e.cells["headquarters"].value == "Pittsburgh, PA"
    assert e.cells["headquarters"].llm_filled is True
    # Already-filled cells should be unchanged
    assert e.cells["description"].value == "Clinical documentation AI"
    assert e.cells["description"].llm_filled is False


@pytest.mark.asyncio
async def test_llm_filler_skips_null_responses():
    from backend.pipeline.llm_filler import llm_fill_gaps
    from backend.models import Entity, CellValue
    import uuid

    entity = Entity(
        id=str(uuid.uuid4()),
        cells={"name": CellValue(value="Healthcare AI", confidence=1.0, sources=[])}
    )

    llm_response = json.dumps([
        {"name": "Healthcare AI", "founded_year": None, "headquarters": None}
    ])

    mock_choice = MagicMock()
    mock_choice.message.content = llm_response
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    columns = ["name", "founded_year", "headquarters"]
    result = await llm_fill_gaps(mock_client, [entity], columns, "AI startups")

    e = result[0]
    assert "founded_year" not in e.cells or e.cells.get("founded_year") is None or e.cells["founded_year"].value is None


@pytest.mark.asyncio
async def test_llm_filler_skips_entities_with_no_gaps():
    from backend.pipeline.llm_filler import llm_fill_gaps
    from backend.models import Entity, CellValue
    import uuid

    entity = Entity(
        id=str(uuid.uuid4()),
        cells={
            "name": CellValue(value="Abridge", confidence=1.0, sources=[]),
            "founded_year": CellValue(value="2018", confidence=0.9, sources=[]),
        }
    )

    mock_client = AsyncMock()
    columns = ["name", "founded_year"]
    result = await llm_fill_gaps(mock_client, [entity], columns, "AI startups")

    # API should never be called - nothing to fill
    mock_client.chat.completions.create.assert_not_called()
    assert result[0].cells["founded_year"].value == "2018"


@pytest.mark.asyncio
async def test_llm_filler_graceful_on_api_failure():
    from backend.pipeline.llm_filler import llm_fill_gaps
    from backend.models import Entity, CellValue
    import uuid

    entity = Entity(
        id=str(uuid.uuid4()),
        cells={"name": CellValue(value="Abridge", confidence=1.0, sources=[])}
    )

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("API down"))

    columns = ["name", "founded_year"]
    result = await llm_fill_gaps(mock_client, [entity], columns, "AI startups")

    assert result == [entity]  # unchanged
    assert "founded_year" not in result[0].cells


@pytest.mark.asyncio
async def test_llm_filler_strips_markdown_fences():
    from backend.pipeline.llm_filler import llm_fill_gaps
    from backend.models import Entity, CellValue
    import uuid

    entity = Entity(
        id=str(uuid.uuid4()),
        cells={"name": CellValue(value="Abridge", confidence=1.0, sources=[])}
    )

    fenced = "```json\n" + json.dumps([{"name": "Abridge", "founded_year": "2018"}]) + "\n```"
    mock_choice = MagicMock()
    mock_choice.message.content = fenced
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    columns = ["name", "founded_year"]
    result = await llm_fill_gaps(mock_client, [entity], columns, "AI startups")

    assert result[0].cells["founded_year"].value == "2018"