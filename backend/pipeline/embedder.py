from __future__ import annotations
import logging, re
from dataclasses import dataclass

import faiss
import numpy as np

from ..config import EMBED_MODEL
from ..models import ScrapedPage

logger = logging.getLogger(__name__)

CHUNK_WORDS = 90
OVERLAP_WORDS = 20
MIN_CHUNK_WORDS = 8

_model = None


@dataclass
class Chunk:
    url: str
    title: str
    text: str


def load_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding(EMBED_MODEL)
        logger.info("Embedding model %s loaded", EMBED_MODEL)
    return _model


def chunk_pages(pages: list[ScrapedPage]) -> list[Chunk]:
    chunks: list[Chunk] = []
    step = CHUNK_WORDS - OVERLAP_WORDS
    for page in pages:
        words = re.sub(r"\s+", " ", page.content).split(" ")
        for start in range(0, len(words), step):
            part = words[start:start + CHUNK_WORDS]
            if len(part) >= MIN_CHUNK_WORDS:
                chunks.append(Chunk(url=page.url, title=page.title, text=" ".join(part)))
            if start + CHUNK_WORDS >= len(words):
                break
    return chunks


def _matrix(vectors) -> np.ndarray:
    matrix = np.asarray(list(vectors), dtype="float32")
    if matrix.size:
        faiss.normalize_L2(matrix)
    return matrix


def embed_pages(pages: list[ScrapedPage], model=None) -> tuple[list[Chunk], np.ndarray]:
    model = model or load_model()
    chunks = chunk_pages(pages)
    if not chunks:
        return [], np.zeros((0, 0), dtype="float32")
    vectors = _matrix(model.embed([chunk.text for chunk in chunks]))
    logger.info("Embedded %d chunks from %d pages", len(chunks), len(pages))
    return chunks, vectors


class ChunkIndex:
    def __init__(self, chunks: list[Chunk], vectors: list[np.ndarray], model=None):
        self.model = model or load_model()
        self.chunks = chunks
        filled = [v for v in vectors if v.size]
        self.index = None
        if filled:
            matrix = np.vstack(filled)
            self.index = faiss.IndexFlatIP(matrix.shape[1])
            self.index.add(matrix)

    def __len__(self) -> int:
        return self.index.ntotal if self.index else 0

    def search(self, queries: list[str], k: int = 3) -> list[list[tuple[Chunk, float]]]:
        if not queries or not len(self):
            return [[] for _ in queries]
        scores, ids = self.index.search(_matrix(self.model.query_embed(queries)), k)
        return [
            [(self.chunks[i], float(s)) for s, i in zip(row_scores, row_ids) if i != -1]
            for row_scores, row_ids in zip(scores, ids)
        ]
