# These tests use mocked HTTP and API responses - no real API keys needed.
# Run with: pytest tests/ -v

import asyncio, json, pytest
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.fixture(autouse=True)
def _fresh_pacer():
    from backend.config import pacer
    old = pacer.limit
    pacer.events.clear()
    pacer.limit = 10 ** 9
    yield
    pacer.events.clear()
    pacer.limit = old


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


def test_is_missing_catches_placeholders():
    from backend.utils import is_missing

    for value in [None, "", "Unknown", "N/A", "not specified.", "  undisclosed "]:
        assert is_missing(value)
    for value in ["2019", "Palo Alto", 0, "Series B"]:
        assert not is_missing(value)


def test_clean_markdown_removes_images_links_and_menus():
    from backend.utils import clean_markdown

    raw = "\n".join([
        "[Skip to content](#content)",
        "[Home](https://a.com) | [Blog](https://a.com/blog)",
        "![logo](https://a.com/logo.png)",
        "## 1. Abridge",
        "Abridge raised **$250M** led by [a16z](https://a16z.com), see https://abridge.com now.",
        "| Tempus | $14B |",
    ])
    assert clean_markdown(raw).splitlines() == [
        "1. Abridge",
        "Abridge raised $250M led by a16z, see now.",
        "Tempus $14B",
    ]


def test_find_gaps_treats_placeholders_as_empty():
    from backend.pipeline.filler import find_gaps

    gaps = find_gaps([_entity("Alpha", city="unknown", focus="Notes")], ["name", "city", "focus"])
    assert [g.column for g in gaps] == ["city"]


def _chunks(*texts):
    from backend.embedder import Chunk
    return [Chunk(url=f"https://p{i}.com", title=f"P{i}", text=t) for i, t in enumerate(texts, start=1)]


def _extract_client(reply):
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_llm_reply(reply))
    return mock_client


@pytest.mark.asyncio
async def test_extractor_links_entity_to_its_chunk():
    from backend.pipeline.extractor import extract_from_chunks

    client = _extract_client({"entities": [
        {"chunk": 1, "name": {"value": "Alpha", "snippet": "Alpha"}},
        {"chunk": 2, "name": {"value": "Beta", "snippet": "Beta"}},
    ]})
    entities = await extract_from_chunks(client, _chunks("Alpha is great", "Beta is good"), ["name"], "startups")

    assert {e.get_name(): e.cells["name"].sources[0].url for e in entities} == {"Alpha": "https://p1.com", "Beta": "https://p2.com"}
    assert client.chat.completions.create.call_count == 1


@pytest.mark.asyncio
async def test_extractor_finds_chunk_by_name_or_drops():
    from backend.pipeline.extractor import extract_from_chunks

    client = _extract_client({"entities": [
        {"chunk": 9, "name": {"value": "Hippocratic AI"}},
        {"chunk": "x", "name": {"value": "Nowhere Inc"}},
        {"name": {"value": "Abridge"}},
    ]})
    entities = await extract_from_chunks(client, _chunks("Abridge builds scribes", LONG_TEXT), ["name"], "startups")

    assert [(e.get_name(), e.cells["name"].sources[0].url) for e in entities] == [
        ("Hippocratic AI", "https://p2.com"), ("Abridge", "https://p1.com")]


@pytest.mark.asyncio
async def test_extractor_drops_placeholder_values():
    from backend.pipeline.extractor import extract_from_chunks

    client = _extract_client({"entities": [{"chunk": 1, "name": {"value": "Alpha"}, "city": {"value": "Not specified"},
                                            "funding": {"value": "unknown"}, "focus": {"value": "Clinical notes"}}]})
    entities = await extract_from_chunks(client, _chunks("Alpha does clinical notes"), ["name", "city", "funding", "focus"], "startups")
    assert set(entities[0].cells) == {"name", "focus"}


@pytest.mark.asyncio
async def test_extractor_no_chunks_no_call():
    from backend.pipeline.extractor import extract_from_chunks

    client = _extract_client({"entities": []})
    assert await extract_from_chunks(client, [], ["name"], "startups") == []
    client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_extractor_survives_api_failure():
    from backend.pipeline.extractor import extract_from_chunks

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("429"))
    assert await extract_from_chunks(client, _chunks("Alpha"), ["name"], "startups") == []


@pytest.mark.asyncio
async def test_extractor_cuts_chunks_to_token_budget():
    from backend.pipeline import extractor
    from backend.config import estimate_tokens

    client = _extract_client({"entities": []})
    texts = ["word " * 400 for _ in range(30)]
    await extractor.extract_from_chunks(client, _chunks(*texts), ["name"], "startups")

    user = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert estimate_tokens(user) <= extractor.EXTRACT_INPUT_TOKENS + 100
    assert user.count("[CHUNK") < 30


def test_top_chunks_picks_best_within_budget_in_page_order():
    from backend.embedder import top_chunks, embed_pages
    from backend.models import ScrapedPage
    from backend.config import estimate_tokens

    fake = _FakeEmbedder()
    pages = [ScrapedPage(url=f"https://p{i}.com", content=text) for i, text in enumerate([
        "The weather today is sunny and warm with light wind across the region all day.",
        "Hippocratic AI was founded in 2023 in Palo Alto and builds models for nurses.",
        "Brooklyn pizza shops stay open late and serve slices to long lines every night.",
    ])]
    chunks, vectors = embed_pages(pages, model=fake)
    budget = estimate_tokens(chunks[1].text) + 1
    picked = top_chunks(chunks, vectors, "Hippocratic founded 2023", budget, model=fake)
    assert [c.url for c in picked] == ["https://p1.com"]


def test_top_chunks_without_vectors_keeps_order():
    from backend.embedder import top_chunks

    picked = top_chunks(_chunks("a " * 10, "b " * 10, "c " * 10), None, "q", 12)
    assert [c.url for c in picked] == ["https://p1.com", "https://p2.com"]


def test_pacer_waits_when_minute_is_full():
    from backend.config import TokenPacer

    now = [0.0]
    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    pacer = TokenPacer(1000, clock=lambda: now[0], sleep=fake_sleep)

    async def go():
        await pacer.reserve(800)
        now[0] = 10.0
        await pacer.reserve(400)

    asyncio.run(go())
    assert slept and abs(slept[0] - 50.0) < 0.01


def test_pacer_settle_uses_real_usage():
    from backend.config import TokenPacer

    pacer = TokenPacer(1000, clock=lambda: 0.0)
    event = asyncio.run(pacer.reserve(900))
    pacer.settle(event, MagicMock(total_tokens=300, prompt_tokens=200, completion_tokens=100), "test")
    assert pacer.available() == 700


@pytest.mark.asyncio
async def test_fill_trims_chunk_to_sentences_about_the_entity():
    from backend.pipeline.filler import fill_from_chunks

    text = ("Weather was nice. Markets went up. Rates stayed flat. Hippocratic AI was established in 2023. "
            "It is based in Palo Alto. Bayesian Health sells sepsis tools. Pizza is great.")
    client = _fill_client([])
    await fill_from_chunks(client, [_entity("Hippocratic AI")], ["name", "founded"], "startups",
                           _FakeIndex({"Hippocratic AI": [(_chunk(text), 0.9)]}))
    prompt = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "Hippocratic AI was established in 2023" in prompt
    assert "Weather was nice" not in prompt and "Pizza is great" not in prompt


@pytest.mark.asyncio
async def test_fill_never_resends_a_chunk_for_the_same_cell():
    from backend.pipeline.filler import fill_from_chunks

    index = _FakeIndex({"Hippocratic AI": [(_chunk(), 0.9)]})
    client = _fill_client([])
    tried = {}
    entity = _entity("Hippocratic AI")
    await fill_from_chunks(client, [entity], ["name", "founded"], "startups", index, tried=tried)
    await fill_from_chunks(client, [entity], ["name", "founded"], "startups", index, tried=tried)
    assert client.chat.completions.create.call_count == 1


@pytest.mark.asyncio
async def test_fill_respects_token_budget():
    from backend.pipeline.filler import fill_from_chunks

    big = [(_chunk(f"Hippocratic AI fact number {i}. " + "filler words here " * 60, f"https://h{i}.com"), 0.9) for i in range(8)]
    client = _fill_client([])
    await fill_from_chunks(client, [_entity("Hippocratic AI")], ["name", "founded", "city"], "startups",
                           _FakeIndex({"Hippocratic AI": big}), token_budget=400)
    prompt = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert prompt.count("Hippocratic AI fact number") == 1



@pytest.mark.asyncio
async def test_fill_cuts_the_call_to_fit_the_token_limit():
    from backend.pipeline.filler import fill_from_chunks, _call_tokens

    hits = [(_chunk(f"Hippocratic AI fact number {i}. " + "filler words here " * 30, f"https://h{i}.com"), 0.9) for i in range(2)]
    client = _fill_client([])
    columns = ["name"] + [f"col_{i}" for i in range(6)]
    await fill_from_chunks(client, [_entity("Hippocratic AI")], columns, "startups",
                           _FakeIndex({"Hippocratic AI": hits}), token_limit=1800)
    prompt = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert _call_tokens(prompt) <= 1800
    assert prompt.count('"column"') < 6

# Resolver
def test_normalise_name(): # Name normalisation should collapse common variations
    from backend.pipeline.resolver import _normalise_name

    assert _normalise_name("OpenAI Inc.") == _normalise_name("openai inc")
    assert _normalise_name("Tempus AI, LLC") == _normalise_name("tempus ai")
    assert _normalise_name("Google LLC") == _normalise_name("google")
    assert _normalise_name("The Hippocratic AI, Inc.") == "hippocratic ai"
    assert _normalise_name("Johnson & Johnson") == "johnson and johnson"
    assert _normalise_name("Co") == "co"


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
    from backend.embedder import chunk_pages, CHUNK_WORDS, OVERLAP_WORDS
    from backend.models import ScrapedPage

    words = [f"w{i}" for i in range(200)]
    chunks = chunk_pages([ScrapedPage(url="https://a.com", content=" ".join(words))])

    assert len(chunks) == 3
    assert all(c.url == "https://a.com" for c in chunks)
    assert chunks[1].text.split()[0] == f"w{CHUNK_WORDS - OVERLAP_WORDS}"
    assert chunks[-1].text.split()[-1] == "w199"


def test_chunk_pages_skips_tiny_pages():
    from backend.embedder import chunk_pages
    from backend.models import ScrapedPage

    assert chunk_pages([ScrapedPage(url="https://a.com", content="too short")]) == []


def test_chunk_index_finds_the_right_page():
    from backend.embedder import embed_pages, ChunkIndex
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
    from backend.embedder import ChunkIndex

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
    from backend.embedder import Chunk
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


def test_targeted_queries_one_per_entity_with_missing_columns():
    from backend.pipeline.filler import find_gaps, targeted_queries

    gaps = find_gaps([_entity("Hippocratic AI"), _entity("Abridge", key_product="Scribe")],
                     ["name", "focus_area", "key_product"])
    assert targeted_queries(gaps, "startups", 2) == ["Hippocratic AI focus area key product", "Abridge focus area"]
    assert targeted_queries(gaps, "startups", 3)[0] == "Hippocratic AI startups focus area key product"


def test_targeted_queries_capped():
    from backend.pipeline.filler import find_gaps, targeted_queries, MAX_TARGETED

    gaps = find_gaps([_entity(f"Company {i}") for i in range(12)], ["name", "focus_area"])
    assert len(targeted_queries(gaps, "startups", 2)) == MAX_TARGETED


def test_chunk_index_add_grows_the_index():
    from backend.embedder import embed_pages, ChunkIndex
    from backend.models import ScrapedPage

    fake = _FakeEmbedder()
    index = ChunkIndex([], [], model=fake)
    chunks, vectors = embed_pages([ScrapedPage(url="https://a.com", content="Hippocratic AI was founded in 2023 in Palo Alto for nurses.")], model=fake)
    index.add(chunks, vectors)

    assert len(index) == 1
    assert index.search(["Hippocratic founded"], k=1)[0][0][0].url == "https://a.com"


class _LoopIndex(_FakeIndex):
    def __init__(self):
        super().__init__({})
        self.added = []

    def add(self, chunks, vectors):
        self.added.extend(chunks)
        for chunk in chunks:
            self.hits_by_entity["Hippocratic AI"] = [(chunk, 0.9)]


@pytest.mark.asyncio
async def test_fill_loop_fast_mode_fills_once_and_never_searches():
    from backend.pipeline import filler

    client = _fill_client([])
    with patch.object(filler, "retrieve", AsyncMock()) as mock_retrieve:
        result = await filler.fill_loop(client, [_entity("Hippocratic AI")], ["name", "founded"], "startups",
                                        _LoopIndex(), set(), max_fills=1)

    assert result.rounds == 1
    mock_retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_fill_loop_searches_and_fills_in_round_two():
    from backend.pipeline import filler
    from backend.models import ScrapedPage

    page = ScrapedPage(url="https://hippo.com", title="Hippo", content=HIPPO_TEXT)
    client = _fill_client([{"id": 0, "value": "2023", "chunk": "c1", "quote": "was established in 2023"}])
    seen = set()
    with patch.object(filler, "retrieve", AsyncMock(return_value=[page])) as mock_retrieve, \
         patch.object(filler, "embed_pages", lambda pages: ([_chunk()], MagicMock(size=1))):
        result = await filler.fill_loop(client, [_entity("Hippocratic AI")], ["name", "founded"], "startups",
                                        _LoopIndex(), seen, max_fills=3)

    assert result.entities[0].cells["founded"].value == "2023"
    assert result.filled == 1
    assert result.rounds == 2
    assert result.queries == ["Hippocratic AI founded"]
    assert result.sources == ["https://hippo.com"]
    assert "https://hippo.com" in seen
    assert mock_retrieve.call_count == 1


@pytest.mark.asyncio
async def test_fill_loop_stops_when_a_round_fills_nothing():
    from backend.pipeline import filler
    from backend.models import ScrapedPage

    page = ScrapedPage(url="https://other.com", content=HIPPO_TEXT)
    client = _fill_client([{"id": 0, "value": None, "chunk": None, "quote": None}])
    with patch.object(filler, "retrieve", AsyncMock(return_value=[page])) as mock_retrieve, \
         patch.object(filler, "embed_pages", lambda pages: ([_chunk()], MagicMock(size=1))):
        result = await filler.fill_loop(client, [_entity("Hippocratic AI")], ["name", "founded"], "startups",
                                        _LoopIndex(), set(), max_fills=3)

    assert result.rounds == 2
    assert mock_retrieve.call_count == 1


@pytest.mark.asyncio
async def test_fill_loop_stops_when_nothing_is_empty():
    from backend.pipeline import filler

    client = _fill_client([])
    with patch.object(filler, "retrieve", AsyncMock()) as mock_retrieve:
        result = await filler.fill_loop(client, [_entity("Hippocratic AI", founded="2023")], ["name", "founded"],
                                        "startups", _LoopIndex(), set(), max_fills=3)

    assert result.rounds == 1
    mock_retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_fill_loop_stops_when_search_finds_no_new_pages():
    from backend.pipeline import filler

    client = _fill_client([])
    with patch.object(filler, "retrieve", AsyncMock(return_value=[])) as mock_retrieve:
        result = await filler.fill_loop(client, [_entity("Hippocratic AI")], ["name", "founded"], "startups",
                                        _LoopIndex(), set(), max_fills=3)

    assert result.rounds == 1
    assert mock_retrieve.call_count == 1


@pytest.mark.asyncio
async def test_fill_loop_survives_search_failure():
    from backend.pipeline import filler

    client = _fill_client([])
    with patch.object(filler, "retrieve", AsyncMock(side_effect=RuntimeError("tavily down"))):
        result = await filler.fill_loop(client, [_entity("Hippocratic AI")], ["name", "founded"], "startups",
                                        _LoopIndex(), set(), max_fills=3)
    assert result.rounds == 1


@pytest.mark.asyncio
async def test_fill_loop_without_index_does_nothing():
    from backend.pipeline import filler

    client = _fill_client([])
    result = await filler.fill_loop(client, [_entity("Hippocratic AI")], ["name", "founded"], "startups",
                                    None, set(), max_fills=3)
    assert result.rounds == 0
    client.chat.completions.create.assert_not_called()


def test_quote_check_accepts_number_with_unit():
    from backend.pipeline.filler import _quote_ok
    from backend.embedder import Chunk

    chunk = Chunk(url="https://a.com", title="", text="Abridge raised $800M in total funding last year.")
    assert _quote_ok("800", "raised $800M in total funding", chunk)
    assert not _quote_ok("80", "raised $800M in total funding", chunk)
    assert not _quote_ok("900", "raised $800M in total funding", chunk)


@pytest.mark.asyncio
async def test_planner_keeps_valid_rules_and_drops_bad_ones():
    from backend.pipeline.planner import plan_search

    client = _planner_client({
        "entity_type": "pizza", "search_queries": ["q"],
        "columns": ["name", "neighborhood", "price_range", "pizza_style"],
        "rules": {
            "location": {"column": "Neighborhood", "names": ["Brooklyn", "BK"]},
            "numeric": [{"column": "price_range", "op": "<=", "value": "2"},
                        {"column": "missing_col", "op": ">", "value": 1},
                        {"column": "price_range", "op": "about", "value": 1}],
            "categorical": [{"column": "pizza_style", "keywords": ["pizza", ""]},
                            {"column": "pizza_style", "keywords": []}],
        },
    })
    plan = await plan_search(client, "cheap pizza in Brooklyn")
    assert plan.rules == {
        "location": {"column": "neighborhood", "names": ["Brooklyn", "BK"]},
        "numeric": [{"column": "price_range", "op": "<=", "value": 2.0}],
        "categorical": [{"column": "pizza_style", "keywords": ["pizza"]}],
    }



@pytest.mark.asyncio
async def test_planner_rules_must_come_from_the_query():
    from backend.pipeline.planner import plan_search

    client = _planner_client({
        "entity_type": "pizza", "search_queries": ["q"],
        "columns": ["name", "neighborhood", "pizza_style"],
        "rules": {
            "location": {"column": "neighborhood", "names": ["Back Bay", "North End", "Boston"]},
            "categorical": [{"column": "pizza_style", "keywords": ["Neapolitan"]}],
        },
    })
    plan = await plan_search(client, "best pizza places in Boston")
    assert plan.rules["location"]["names"] == ["Boston", "Back Bay", "North End"]
    assert plan.rules["categorical"] == []


@pytest.mark.asyncio
async def test_planner_drops_location_not_in_query():
    from backend.pipeline.planner import plan_search

    client = _planner_client({
        "entity_type": "pizza", "search_queries": ["q"], "columns": ["name", "neighborhood", "style"],
        "rules": {"location": {"column": "neighborhood", "names": ["Back Bay", "North End"]}},
    })
    plan = await plan_search(client, "best pizza places in Boston")
    assert plan.rules["location"] is None

@pytest.mark.asyncio
async def test_planner_rules_default_when_missing():
    from backend.pipeline.planner import plan_search

    client = _planner_client({"entity_type": "x", "columns": ["name", "a", "b"], "search_queries": ["q"]})
    plan = await plan_search(client, "anything")
    assert plan.rules == {"location": None, "numeric": [], "categorical": []}


PIZZA_RULES = {
    "location": {"column": "neighborhood", "names": ["Brooklyn", "BK"]},
    "numeric": [{"column": "price_range", "op": "<=", "value": 2.0}],
    "categorical": [{"column": "style", "keywords": ["pizza", "pizzeria"]}],
}
PIZZA_COLS = ["name", "neighborhood", "price_range", "style"]


def test_rank_scores_rules_and_sorts():
    from backend.pipeline.ranker import rank

    rows = [
        _entity("Joes Pizza", neighborhood="Manhattan", price_range="$", style="New York pizza"),
        _entity("Lucali", neighborhood="Carroll Gardens, Brooklyn", price_range="$$", style="Thin crust pizza"),
        _entity("Juliana", neighborhood="Brooklyn", price_range="$$$", style="Coal oven pizzeria"),
    ]
    top, _, info = rank(rows, PIZZA_RULES, PIZZA_COLS, "cheap pizza in Brooklyn")

    assert [e.get_name() for e in top] == ["Lucali", "Joes Pizza", "Juliana"]
    assert info.total == 3
    assert info.scores[top[0].id] == 3
    assert info.labels == ["In Brooklyn", "price range <= 2", "style: pizza"]


def test_rank_uses_whole_words_only():
    from backend.pipeline.ranker import rank

    rules = {"location": {"column": "market", "names": ["US"]}, "numeric": [], "categorical": []}
    rows = [_entity("Canva", market="Australia"), _entity("Stripe", market="US and Europe")]
    top, _, info = rank(rows, rules, ["name", "market"], "q")
    assert {e.get_name(): info.scores[e.id] for e in top} == {"Stripe": 1, "Canva": 0}


def test_rank_numbers_with_units_and_dollar_signs():
    from backend.pipeline.ranker import _to_number

    assert _to_number("$$") == 2
    assert _to_number("$800M") == 800_000_000
    assert _to_number("Raised $1.5 billion") == 1_500_000_000
    assert _to_number("Founded in 2019") == 2019
    assert _to_number("Series B") is None


def test_rank_empty_rule_cell_counts_as_not_met():
    from backend.pipeline.ranker import rank

    rows = [_entity("Ghost", style="pizza"), _entity("Lucali", neighborhood="Brooklyn", price_range="$$", style="pizza")]
    top, _, info = rank(rows, PIZZA_RULES, PIZZA_COLS, "q")
    assert info.scores[top[1].id] == 1
    assert top[0].get_name() == "Lucali"


def test_rank_tie_broken_by_sources_then_fullness():
    from backend.pipeline.ranker import rank
    from backend.models import SourceRef

    one = _entity("One", "https://a.com", style="x")
    two = _entity("Two", "https://a.com", style="x")
    two.cells["style"].sources.append(SourceRef(url="https://b.com"))
    full = _entity("Full", "https://a.com", style="x", extra="y")
    top, _, _ = rank([one, full, two], {}, ["name", "style", "extra"], "q")
    assert [e.get_name() for e in top] == ["Two", "Full", "One"]


def test_rank_relevance_breaks_ties_by_meaning():
    from backend.pipeline.ranker import rank

    rows = [_entity("Pizza Place", style="brooklyn pizza"), _entity("Hippocratic AI", style="hippocratic founded 2023")]
    top, _, _ = rank(rows, {}, ["name", "style"], "Hippocratic founded", model=_FakeEmbedder())
    assert top[0].get_name() == "Hippocratic AI"


def test_rank_keeps_top_ten():
    from backend.pipeline.ranker import rank, TOP_N

    top, _, info = rank([_entity(f"E{i}", style="x") for i in range(15)], {}, ["name", "style"], "q")
    assert len(top) == TOP_N
    assert len(info.scores) == TOP_N


def test_pick_columns_keeps_full_and_rule_columns():
    from backend.pipeline.ranker import pick_columns

    rows = [_entity("A", style="x", funding="1", city="Boston", hq="a"),
            _entity("B", style="y", hq="b"),
            _entity("C", style="z", city="NYC", hq="c")]
    cols = pick_columns(rows, ["name", "style", "funding", "city", "hq"], keep={"city"})
    assert cols == ["name", "style", "city", "hq"]


def test_pick_columns_falls_back_to_80_percent():
    from backend.pipeline.ranker import pick_columns

    rows = [_entity(f"E{i}", a="1", b="1" if i < 9 else "", c="1" if i < 8 else "", d="1" if i < 5 else "") for i in range(10)]
    assert pick_columns(rows, ["name", "a", "b", "c", "d"], keep=set()) == ["name", "a", "b", "c"]



def test_pick_columns_hides_empty_columns_and_shows_left_over_data():
    from backend.pipeline.ranker import pick_columns

    rows = [_entity("A", city="Boston", price="$$", note="great"), _entity("B"), _entity("C")]
    cols = pick_columns(rows, ["name", "style", "city", "price", "note"], keep={"style"})
    assert cols == ["name", "city", "price", "note"]

def test_rank_without_rules_or_entities_is_safe():
    from backend.pipeline.ranker import rank

    top, cols, info = rank([], {}, ["name"], "q")
    assert top == [] and cols == ["name"] and info.total == 0
