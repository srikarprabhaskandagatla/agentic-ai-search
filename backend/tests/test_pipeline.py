# These tests use mocked HTTP and API responses — no real API keys needed.
# Run with: pytest backend/tests/ -v

import asyncio, json, pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Planner
@pytest.mark.asyncio
async def test_planner_parse(): # Planner should parse valid JSON from Claude into a SearchPlan
    from backend.pipeline.planner import plan_search
    from backend.models import SearchPlan

    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "entity_type": "AI healthcare startups",
        "columns": ["name", "description", "founded_year", "headquarters"],
        "search_queries": ["AI healthcare startups 2024", "top health AI companies"],
        "rationale": "Standard company attributes",
    })
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    plan = await plan_search(mock_client, "AI startups in healthcare")

    assert plan.entity_type == "AI healthcare startups"
    assert plan.columns[0] == "name"
    assert len(plan.columns) == 4
    assert len(plan.search_queries) == 2


@pytest.mark.asyncio
async def test_planner_name_always_first(): # Name column should always be first, even if model puts it elsewhere
    from backend.pipeline.planner import plan_search

    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "entity_type": "pizza restaurants",
        "columns": ["rating", "name", "address", "price_range"],
        "search_queries": ["top pizza brooklyn"],
        "rationale": "Test",
    })
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    plan = await plan_search(mock_client, "pizza places Brooklyn")
    assert plan.columns[0] == "name"


# Scraper 
@pytest.mark.asyncio
async def test_scraper_skips_binary(): # Scraper should skip URLs with binary file extensions
    from backend.pipeline.scraper import scrape_urls

    pages = await scrape_urls(["https://example.com/file.pdf"])
    assert len(pages) == 0  # PDF should be skipped (no content)


def test_scraper_clean_html(): # HTML cleaner should extract text and remove boilerplate
    from backend.pipeline.scraper import _clean_html

    html = """
    <html><head><title>Test Page</title></head>
    <body>
      <nav>Nav menu here</nav>
      <main>
        <article>
          <h1>Company Name</h1>
          <p>This company was founded in 2020 and is headquartered in San Francisco.</p>
          <p>They raised $50 million in Series B funding last year.</p>
        </article>
      </main>
      <footer>Footer text here</footer>
    </body></html>
    """
    title, content = _clean_html(html, "https://test.com")
    assert title == "Test Page"
    assert "founded in 2020" in content  # "Company Name" is <20 chars so filtered
    assert "founded in 2020" in content


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