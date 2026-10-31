from __future__ import annotations
import math, re, unicodedata, logging
from collections import Counter
from difflib import SequenceMatcher
from urllib.parse import urlsplit

from ..models import Entity, CellValue, ScrapedPage, SourceRef

logger = logging.getLogger(__name__)

SUFFIXES = ("inc", "llc", "ltd", "corp", "corporation", "co", "company", "plc", "gmbh", "limited")
WEBSITE_COLUMNS = ("website", "url", "domain", "homepage")
FUZZY_CUTOFF = 0.92
DOMAIN_NAME_CUTOFF = 0.6
CONTAINMENT_CUTOFF = 0.8
RARE_SHARE = 0.6
MIN_PARAGRAPH_CHARS = 40


def _normalise_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower().replace("&", " and ")
    name = re.sub(r"[^\w\s]", " ", name)
    words = name.split()
    if words and words[0] == "the":
        words = words[1:]
    while len(words) > 1 and words[-1] in SUFFIXES:
        words = words[:-1]
    return " ".join(words)


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _domain(entity: Entity) -> str:
    for col, cell in entity.cells.items():
        if col.lower() not in WEBSITE_COLUMNS or cell.value is None:
            continue
        text = str(cell.value).strip().lower()
        if " " in text or "." not in text:
            continue
        host = urlsplit(text if "//" in text else f"//{text}").netloc
        return host.removeprefix("www.")
    return ""


def _source_count(entity: Entity) -> int:
    return len({src.url for cell in entity.cells.values() for src in cell.sources})


# IDF part of BM25 only, since names are too short for TF and length to matter.
class WordWeights:
    def __init__(self, pages: list[ScrapedPage], context: str):
        paragraphs = [
            part for page in pages for part in re.split(r"\n+", page.content)
            if len(part) >= MIN_PARAGRAPH_CHARS
        ]
        self.total = max(len(paragraphs), 1)
        self.counts = Counter(word for part in paragraphs for word in set(_words(part)))
        self.common = set(_words(context))
        self.max_weight = math.log(self.total) if self.total > 1 else 1.0

    def __call__(self, word: str) -> float:
        if word in self.common:
            return 0.0
        return math.log(self.total / max(self.counts.get(word, 0), 1))


def _rare_overlap(a: str, b: str, weight: WordWeights) -> bool:
    words_a, words_b = set(a.split()), set(b.split())
    shared = sum(weight(w) for w in words_a & words_b)
    smaller = min(sum(weight(w) for w in words_a), sum(weight(w) for w in words_b))
    if smaller <= 0:
        return False
    return shared / smaller >= CONTAINMENT_CUTOFF and shared >= RARE_SHARE * weight.max_weight


def _fuzzy(a: str, b: str) -> float:
    return SequenceMatcher(None, a.replace(" ", ""), b.replace(" ", "")).ratio()


def _same_entity(a: Entity, b: Entity, weight: WordWeights) -> bool:
    name_a, name_b = _normalise_name(a.get_name()), _normalise_name(b.get_name())
    domain_a, domain_b = _domain(a), _domain(b)
    if domain_a and domain_b:
        if domain_a != domain_b:
            return False
        return _fuzzy(name_a, name_b) >= DOMAIN_NAME_CUTOFF or _rare_overlap(name_a, name_b, weight)
    return _fuzzy(name_a, name_b) >= FUZZY_CUTOFF or _rare_overlap(name_a, name_b, weight)


def _merge_cells(cells_list: list[CellValue]) -> CellValue:
    groups: dict[str, list[CellValue]] = {}
    for cell in cells_list:
        groups.setdefault(str(cell.value).strip().lower(), []).append(cell)

    def support(group: list[CellValue]) -> tuple[int, float]:
        urls = {src.url for cell in group for src in cell.sources}
        return len(urls), max(cell.confidence for cell in group)

    winner = max(groups.values(), key=support)
    best = max(winner, key=lambda c: c.confidence)
    sources: list[SourceRef] = []
    seen_urls: set[str] = set()
    for cell in winner:
        for src in cell.sources:
            if src.url not in seen_urls:
                seen_urls.add(src.url)
                sources.append(src)
    return CellValue(value=best.value, confidence=best.confidence, sources=sources)


def _merge_entity_group(entities: list[Entity]) -> Entity:
    if len(entities) == 1:
        return entities[0]
    columns: list[str] = []
    for e in entities:
        columns.extend(col for col in e.cells if col not in columns)
    merged_cells = {
        col: _merge_cells([e.cells[col] for e in entities if col in e.cells])
        for col in columns
    }
    return Entity(id=entities[0].id, cells=merged_cells)


def _fast_dedup(entities: list[Entity]) -> list[list[Entity]]:
    groups: dict[str, list[Entity]] = {}
    for entity in entities:
        name_cell = entity.cells.get("name")
        if not name_cell or name_cell.value is None:
            continue
        groups.setdefault(_normalise_name(str(name_cell.value)), []).append(entity)
    return list(groups.values())


def _group(entities: list[Entity], weight: WordWeights) -> list[list[Entity]]:
    parent = list(range(len(entities)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            if find(i) != find(j) and _same_entity(entities[i], entities[j], weight):
                parent[find(j)] = find(i)

    groups: dict[int, list[Entity]] = {}
    for i, entity in enumerate(entities):
        groups.setdefault(find(i), []).append(entity)
    return list(groups.values())


def resolve_entities(raw_entities: list[Entity], pages: list[ScrapedPage], context: str = "") -> list[Entity]:
    if not raw_entities:
        return []

    logger.info("Resolving %d raw entities", len(raw_entities))
    weight = WordWeights(pages, context)
    exact = [_merge_entity_group(g) for g in _fast_dedup(raw_entities)]
    resolved = [_merge_entity_group(g) for g in _group(exact, weight)]
    resolved.sort(key=lambda e: (_source_count(e), len(e.cells)), reverse=True)
    logger.info("Resolved to %d unique entities", len(resolved))
    return resolved
